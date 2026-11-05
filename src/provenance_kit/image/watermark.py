"""Invisible image watermarks with read-back verification.

:class:`ImageWatermarker` embeds an identifier (packed by a
:class:`~provenance_kit.payload.PayloadCodec` into exactly the backend's
capacity) and detects it again. The model sits behind
:class:`ImageWatermarkBackend`; :class:`TrustMarkBackend` (Adobe TrustMark,
extra ``image-watermark``) is the implementation provided.

**Verified embedding.** A mark that is fine in the watermarked pixels can
still be destroyed by the encoding the application publishes (WebP or JPEG at
some quality, resizing). :meth:`ImageWatermarker.embed_verified` therefore
embeds, runs the caller's encoder (image in, bytes out), decodes those bytes
as untrusted media, and reads the mark back. If the payload is not recovered
exactly, it retries with the next strength. The result says which strength
worked, or that none did; it does not raise for "not verified", so the
caller can publish anyway and flag the asset::

    marker = ImageWatermarker(TrustMarkBackend(model_dir="/srv/models/trustmark"))


    def to_webp(image):
        buffer = io.BytesIO()
        image.save(buffer, "WEBP", quality=80)
        return buffer.getvalue()


    result = marker.embed_verified(image, identifier=123_456, encoder=to_webp)
    publish(result.encoded)
    if not result.verified:
        flag_for_review(...)

**Shape of the API.** The video and audio watermark modules expose module
functions over raw payload bits (``embed_verified(frames, bits, *,
encoder=...)``), because their payload layout depends on choices made per
call (a repetition code, a codec). Images use a class instead: the
TrustMark capacity is fixed by the backend's BCH mode, so
:class:`ImageWatermarker` binds a backend, a
:class:`~provenance_kit.payload.PayloadCodec` and limits once, and its
methods take integer identifiers (:meth:`~ImageWatermarker.embed_bits`
takes raw bits). The encoder callable has the same keyword, ``encoder``,
in all three, and backend failures raise the same
:class:`~provenance_kit.errors.WatermarkBackendError`.

**Payload and false positives.** The default codec is
``PayloadCodec(payload_bits, version_bits=4, check_bits=8)``: for TrustMark
``BCH_5`` that leaves 49 identifier bits. TrustMark's BCH decoder accepts
random bits as a valid codeword now and then (an unmarked image decodes in
roughly 0.2% of reads in ``BCH_5`` and up to several percent in ``BCH_3``,
by the code's coverage; the decoder's outputs are not uniform, so the real
rate differs). The backend rejects marks in another encoding, and the 8
check bits reject about 255 in 256 of the rest, so trust a detection when
``detection.found and detection.payload.valid``.

**Measured robustness.** See ``docs/image.md`` for the strengths, PSNR and
verification results measured through WebP and JPEG encoders; the default
strength schedule is :data:`DEFAULT_STRENGTHS`.

Thread safety: :class:`ImageWatermarker` is as thread-safe as its backend
(:class:`TrustMarkBackend` is); it keeps no mutable state.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from provenance_kit.bits import BitsLike, normalize_bits
from provenance_kit.detection import WatermarkDetection
from provenance_kit.errors import WatermarkBackendError
from provenance_kit.image import _modes
from provenance_kit.image._trustmark import (
    MODEL_BASE_URL,
    MODEL_DIR_ENV,
    MODEL_FILES,
    ModelFile,
    TrustMarkBackend,
    TrustMarkDetection,
    TrustMarkEncoding,
    TrustMarkVariant,
    default_model_dir,
    download_trustmark_models,
)
from provenance_kit.image.loading import load_image
from provenance_kit.limits import MediaLimits
from provenance_kit.payload import PayloadCodec

__all__ = [
    "DEFAULT_CHECK_BITS",
    "DEFAULT_STRENGTHS",
    "DEFAULT_VERSION_BITS",
    "MODEL_BASE_URL",
    "MODEL_DIR_ENV",
    "MODEL_FILES",
    "EmbedAttempt",
    "ImageEncoder",
    "ImageWatermarkBackend",
    "ImageWatermarker",
    "ModelFile",
    "TrustMarkBackend",
    "TrustMarkDetection",
    "TrustMarkEncoding",
    "TrustMarkVariant",
    "VerifiedEmbedding",
    "WatermarkBackendError",
    "default_model_dir",
    "download_trustmark_models",
    "psnr",
]

#: Strengths :meth:`ImageWatermarker.embed_verified` tries, in order.
DEFAULT_STRENGTHS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)

#: Version bits of the default payload codec.
DEFAULT_VERSION_BITS = 4

#: Check bits of the default payload codec.
DEFAULT_CHECK_BITS = 8

#: Encodes a watermarked image the way the application will publish it.
ImageEncoder = Callable[[Image.Image], bytes]


class ImageWatermarkBackend(Protocol):
    """What :class:`ImageWatermarker` needs from an invisible-watermark model."""

    @property
    def payload_bits(self) -> int:
        """The number of payload bits one mark carries."""
        ...

    def embed(self, image: Image.Image, bits: str, *, strength: float) -> Image.Image:
        """Returns a watermarked copy of ``image`` carrying ``bits``."""
        ...

    def extract(self, image: Image.Image) -> WatermarkDetection:
        """Looks for a mark; returns ``found=False`` (not an error) when there is none."""
        ...


@dataclass(frozen=True, slots=True)
class EmbedAttempt:
    """One strength tried by :meth:`ImageWatermarker.embed_verified`.

    Attributes:
        strength: The strength used.
        recovered: Whether the read-back returned exactly the embedded bits.
        detection: The read-back result (payload decoded with the codec).
        psnr: PSNR in dB of the watermarked image against the original,
            before encoding (``inf`` when identical).
        encoded_size: The size of the encoder's output in bytes.
    """

    strength: float
    recovered: bool
    detection: WatermarkDetection
    psnr: float
    encoded_size: int


@dataclass(frozen=True, slots=True)
class VerifiedEmbedding:
    """The outcome of :meth:`ImageWatermarker.embed_verified`.

    Attributes:
        verified: Whether some strength survived the encoder.
        strength: The strength that survived, or ``None``.
        image: The watermarked image of the successful attempt, or of the
            last attempt when none survived (left out of ``repr``).
        encoded: The encoder's output for that image: what to publish (left
            out of ``repr``, so that logging a result does not log the media).
        attempts: Every attempt, in order.
    """

    verified: bool
    strength: float | None
    image: Image.Image = dataclasses.field(repr=False)
    encoded: bytes = dataclasses.field(repr=False)
    attempts: tuple[EmbedAttempt, ...]


def psnr(original: Image.Image, marked: Image.Image) -> float:
    """Peak signal-to-noise ratio in dB between two images of the same size.

    Both are compared in 8-bit RGB (alpha ignored).

    Args:
        original: The reference image.
        marked: The image to compare.

    Returns:
        The PSNR; ``inf`` when the images are identical.

    Raises:
        ValueError: If the sizes differ.
    """
    if original.size != marked.size:
        raise ValueError("images must have the same size")
    a = np.asarray(_modes.to_rgb(original), dtype=np.float64)
    b = np.asarray(_modes.to_rgb(marked), dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    return math.inf if mse == 0.0 else 10.0 * math.log10(255.0**2 / mse)


class ImageWatermarker:
    """Embeds and detects identifiers with an invisible-watermark backend.

    Args:
        backend: The model. ``None`` builds ``TrustMarkBackend()`` with its
            defaults (variant Q, ``BCH_5``, CPU, models already in
            :func:`default_model_dir`; no download).
        codec: How identifiers become bits; its ``total_bits`` must equal
            ``backend.payload_bits``. ``None`` uses
            ``PayloadCodec(backend.payload_bits, version_bits=4, check_bits=8)``.
        limits: Limits for images handled here, including the encoder output
            decoded by :meth:`embed_verified`; ``None`` uses
            ``MediaLimits()``.
        strength: The default strength for :meth:`embed`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the codec does not match the backend's capacity or the
            strength is invalid.
        BackendUnavailableError: If ``backend`` is ``None`` and the extra is
            missing.
        ModelLoadError: If ``backend`` is ``None`` and the models are not
            provisioned.
    """

    def __init__(
        self,
        backend: ImageWatermarkBackend | None = None,
        *,
        codec: PayloadCodec | None = None,
        limits: MediaLimits | None = None,
        strength: float = 1.0,
    ) -> None:
        """Configures the watermarker (see the class docstring)."""
        self._backend: ImageWatermarkBackend = (
            backend if backend is not None else TrustMarkBackend()
        )
        capacity = self._backend.payload_bits
        if codec is None:
            codec = PayloadCodec(
                capacity, version_bits=DEFAULT_VERSION_BITS, check_bits=DEFAULT_CHECK_BITS
            )
        if not isinstance(codec, PayloadCodec):
            raise TypeError("codec must be a PayloadCodec")
        if codec.total_bits != capacity:
            raise ValueError(
                f"the codec has {codec.total_bits} bits but the backend carries {capacity}"
            )
        self._codec = codec
        self._limits = limits if limits is not None else MediaLimits()
        if not isinstance(self._limits, MediaLimits):
            raise TypeError("limits must be a MediaLimits")
        self._strength = _check_strength(strength)

    @property
    def backend(self) -> ImageWatermarkBackend:
        """The model backend."""
        return self._backend

    @property
    def codec(self) -> PayloadCodec:
        """The payload codec."""
        return self._codec

    @property
    def payload_bits(self) -> int:
        """The backend's capacity in bits."""
        return self._codec.total_bits

    def embed(
        self,
        image: Image.Image,
        identifier: int,
        *,
        version: int = 0,
        strength: float | None = None,
    ) -> Image.Image:
        """Embeds an identifier without verification.

        Args:
            image: The image to mark (decoded; checked against the limits).
            identifier: The identifier, up to ``codec.max_identifier``.
            version: The payload format version.
            strength: Overrides the default strength.

        Returns:
            The watermarked image (same size; see the backend for the mode).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If the identifier, version or strength is invalid.
            PayloadError: If the identifier does not fit.
            MediaLimitExceededError: If the image exceeds the limits.
            WatermarkBackendError: If the model fails or returns malformed
                output.
        """
        return self.embed_bits(
            image, self._codec.encode(identifier, version=version), strength=strength
        )

    def embed_bits(
        self, image: Image.Image, bits: BitsLike, *, strength: float | None = None
    ) -> Image.Image:
        """Embeds raw payload bits (``payload_bits`` of them).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If the bits or the strength are invalid.
            MediaLimitExceededError: If the image exceeds the limits.
            WatermarkBackendError: If the model fails or returns something
                other than an image of the input's size.
        """
        self._check_image(image)
        text = normalize_bits(bits, length=self.payload_bits)
        level = self._strength if strength is None else _check_strength(strength)
        return self._embed(image, text, level)

    def detect(self, image: Image.Image) -> WatermarkDetection:
        """Looks for a mark and decodes its payload.

        Args:
            image: The decoded image (checked against the limits).

        Returns:
            The backend's detection with ``payload`` decoded by the codec
            when a mark was found. Trust it when ``found`` and
            ``payload.valid``.

        Raises:
            TypeError: If ``image`` is not a Pillow image.
            MediaLimitExceededError: If the image exceeds the limits.
            WatermarkBackendError: If the model fails, or the backend
                returns a payload of the wrong length.
        """
        self._check_image(image)
        detection = self._backend.extract(image)
        if not detection.found or detection.bits is None:
            return detection
        if len(detection.bits) != self.payload_bits:
            raise WatermarkBackendError("the backend returned a payload of the wrong length")
        return dataclasses.replace(detection, payload=self._codec.decode(detection.bits))

    def detect_encoded(self, data: bytes | bytearray | memoryview) -> WatermarkDetection:
        """Decodes untrusted image bytes (first frame) and looks for a mark.

        Raises:
            TypeError: If ``data`` is not bytes-like.
            MediaError: If the bytes cannot be decoded within the limits.
            WatermarkBackendError: If the model fails or returns malformed
                output.
        """
        return self.detect(load_image(data, limits=self._limits, mode="rgb").image)

    def embed_verified(
        self,
        image: Image.Image,
        identifier: int,
        *,
        encoder: ImageEncoder,
        version: int = 0,
        strengths: Sequence[float] = DEFAULT_STRENGTHS,
    ) -> VerifiedEmbedding:
        """Embeds, encodes and reads back, escalating the strength until it survives.

        For each strength in order: embed, call ``encoder`` on the result,
        decode the bytes with :func:`~provenance_kit.image.loading.load_image`
        (as untrusted media, with this watermarker's limits), detect, and
        stop at the first read-back that returns exactly the embedded bits.

        Args:
            image: The image to mark.
            identifier: The identifier to embed.
            encoder: The application's encoder: watermarked image in, encoded
                bytes out (for example WebP at quality 80). Its exceptions
                propagate unchanged. Same keyword as the video and audio
                ``embed_verified`` functions.
            version: The payload format version.
            strengths: Strengths to try, in order (usually increasing).

        Returns:
            The outcome; ``verified`` is false when no strength survived (the
            image and bytes of the last attempt are still returned).

        Raises:
            TypeError: If an argument has the wrong type or ``encoder`` does
                not return bytes.
            ValueError: If the identifier, version or a strength is invalid,
                or ``strengths`` is empty.
            PayloadError: If the identifier does not fit.
            MediaLimitExceededError: If the image exceeds the limits.
            MediaError: If the encoder's output cannot be decoded.
            WatermarkBackendError: If the model fails or returns malformed
                output.
        """
        if not callable(encoder):
            raise TypeError("encoder must be callable")
        levels = _check_strengths(strengths)
        bits = self._codec.encode(identifier, version=version)
        self._check_image(image)
        attempts: list[EmbedAttempt] = []
        marked = image
        encoded = b""
        for level in levels:
            marked = self._embed(image, bits, level)
            output = encoder(marked)
            if not isinstance(output, (bytes, bytearray, memoryview)):
                raise TypeError("encoder must return bytes")
            encoded = bytes(output)
            detection = self.detect_encoded(encoded)
            recovered = detection.found and detection.bits == bits
            attempts.append(
                EmbedAttempt(
                    strength=level,
                    recovered=recovered,
                    detection=detection,
                    psnr=psnr(image, marked),
                    encoded_size=len(encoded),
                )
            )
            if recovered:
                return VerifiedEmbedding(
                    verified=True,
                    strength=level,
                    image=marked,
                    encoded=encoded,
                    attempts=tuple(attempts),
                )
        return VerifiedEmbedding(
            verified=False,
            strength=None,
            image=marked,
            encoded=encoded,
            attempts=tuple(attempts),
        )

    def _embed(self, image: Image.Image, bits: str, strength: float) -> Image.Image:
        marked: object = self._backend.embed(image, bits, strength=strength)
        if not isinstance(marked, Image.Image) or marked.size != image.size:
            raise WatermarkBackendError(
                "the backend returned something other than an image of the input's size"
            )
        return marked

    def _check_image(self, image: Image.Image) -> None:
        candidate: object = image
        if not isinstance(candidate, Image.Image):
            raise TypeError(f"image must be a PIL.Image.Image, got {type(candidate).__name__}")
        self._limits.check_dimensions(*image.size)


def _check_strength(strength: object) -> float:
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise TypeError("strength must be a number")
    value = float(strength)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("strength must be a positive finite number")
    return value


def _check_strengths(strengths: object) -> tuple[float, ...]:
    if isinstance(strengths, (str, bytes)) or not isinstance(strengths, Sequence):
        raise TypeError("strengths must be a sequence of numbers")
    levels = tuple(_check_strength(s) for s in strengths)
    if not levels:
        raise ValueError("strengths must not be empty")
    return levels
