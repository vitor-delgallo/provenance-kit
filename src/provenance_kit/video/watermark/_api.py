"""High-level video watermark functions (re-exported by the package)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from provenance_kit._checks import check_number
from provenance_kit._typing import BytesLike, StrPath
from provenance_kit.bits import hamming_distance, normalize_bits
from provenance_kit.limits import MediaLimits
from provenance_kit.payload import DecodedPayload, PayloadCodec
from provenance_kit.tools import ExternalTools
from provenance_kit.video.frames import FrameSampling, iter_frames
from provenance_kit.video.watermark.base import (
    Frame,
    VideoWatermarkBackend,
    VideoWatermarkDetection,
)
from provenance_kit.video.watermark.repetition import RepetitionCode

__all__ = [
    "DEFAULT_MAX_P_VALUE",
    "DEFAULT_MIN_MEAN_ABS_SCORE",
    "FrameEncoder",
    "VerifiedVideoEmbedding",
    "detect_watermark",
    "embed_verified",
    "embed_watermark",
]

#: Default decision threshold with a repetition code: the watermark is
#: reported as found when unmarked content would agree this much with
#: probability at most 1e-4.
DEFAULT_MAX_P_VALUE = 1e-4

#: Fallback decision without a repetition code and without check bits: the
#: mean magnitude of the frame-averaged channel scores must reach this. It is
#: a heuristic (measured on VideoSeal: about 0.05 to 0.25 on unmarked frames,
#: 0.4 to 1.7 on marked ones depending on the encoding); prefer a repetition
#: code or a codec with check bits.
DEFAULT_MIN_MEAN_ABS_SCORE = 0.5

#: The encoder callable of :func:`embed_verified`: watermarked frames in,
#: encoded media bytes out (for example by piping the frames into ffmpeg).
FrameEncoder = Callable[[Iterator[Frame]], BytesLike]


@dataclass(frozen=True, slots=True)
class VerifiedVideoEmbedding:
    """The result of :func:`embed_verified`.

    Attributes:
        encoded: What the encoder returned, whether or not it verified.
        detection: The detection run on the decoded ``encoded`` bytes.
        bit_errors: Payload bits that differ after the round trip (the
            repetition code's data bits, or the 256 channel bits).
        verified: True when the watermark was found and the payload came back
            without errors (and, with a codec, decodes as valid).
        frames_embedded: How many frames were watermarked and handed to the
            encoder.
    """

    encoded: bytes = field(repr=False)
    detection: VideoWatermarkDetection
    bit_errors: int
    verified: bool
    frames_embedded: int


def _default_backend() -> VideoWatermarkBackend:
    from provenance_kit.video.watermark._videoseal import VideoSealBackend  # noqa: PLC0415

    return VideoSealBackend()


def _payload_and_channel(
    bits: str,
    backend: VideoWatermarkBackend,
    repetition: RepetitionCode | None,
) -> tuple[str, str]:
    if repetition is None:
        payload = normalize_bits(bits, length=backend.payload_bits)
        return payload, payload
    _check_repetition(repetition, backend)
    payload = normalize_bits(bits, length=repetition.data_bits)
    return payload, repetition.encode(payload)


def _check_repetition(repetition: RepetitionCode, backend: VideoWatermarkBackend) -> None:
    if not isinstance(repetition, RepetitionCode):
        raise TypeError("repetition must be a RepetitionCode")
    if repetition.total_bits != backend.payload_bits:
        raise ValueError(
            f"the repetition code spans {repetition.total_bits} bits but the backend "
            f"carries {backend.payload_bits}"
        )


def _check_codec(codec: PayloadCodec | None, payload_size: int) -> None:
    if codec is None:
        return
    if not isinstance(codec, PayloadCodec):
        raise TypeError("codec must be a PayloadCodec")
    if codec.total_bits != payload_size:
        raise ValueError(
            f"the codec expects {codec.total_bits} bits but the payload has {payload_size}"
        )


def embed_watermark(
    frames: Iterable[Frame],
    bits: str,
    *,
    backend: VideoWatermarkBackend | None = None,
    repetition: RepetitionCode | None = None,
) -> Iterator[Frame]:
    """Watermarks decoded frames.

    Args:
        frames: RGB ``uint8`` frames shaped ``(height, width, 3)``, in
            presentation order. Consumed lazily.
        bits: The payload: ``repetition.data_bits`` bits with a repetition
            code, else ``backend.payload_bits`` bits. Typically
            ``codec.encode(identifier)``.
        backend: The model. Defaults to ``VideoSealBackend()`` (which loads a
            pre-provisioned model and never downloads).
        repetition: An optional repetition code (see
            :class:`~provenance_kit.video.watermark.RepetitionCode`).

    Returns:
        The watermarked frames, produced as the iterator is consumed.

    Raises:
        TypeError: If ``repetition`` has the wrong type.
        ValueError: If ``bits`` has the wrong length or the repetition code
            does not fit the backend.
        BackendUnavailableError: If the default backend cannot run.
        ModelLoadError: If the default backend's model cannot be loaded.
    """
    backend = _default_backend() if backend is None else backend
    _, channel = _payload_and_channel(bits, backend, repetition)
    return backend.embed_frames(frames, channel)


def detect_watermark(
    frames: Iterable[Frame],
    *,
    backend: VideoWatermarkBackend | None = None,
    codec: PayloadCodec | None = None,
    repetition: RepetitionCode | None = None,
    max_p_value: float = DEFAULT_MAX_P_VALUE,
    min_mean_abs_score: float = DEFAULT_MIN_MEAN_ABS_SCORE,
) -> VideoWatermarkDetection:
    """Looks for a watermark in decoded frames.

    The channel scores are averaged over all frames (memory stays constant),
    then decoded:

    * with a repetition code of two or more copies, by soft majority vote; the
      watermark is ``found`` when the copies' agreement has a p-value of at
      most ``max_p_value``, and ``confidence`` is ``1 - p_value``;
    * otherwise the signs give the bits; with a codec that has check bits the
      watermark is ``found`` when the payload decodes as valid, else when
      the mean score magnitude reaches ``min_mean_abs_score`` (a heuristic);
      ``confidence`` is ``None``.

    Args:
        frames: RGB ``uint8`` frames shaped ``(height, width, 3)``; any subset
            of the video works (more frames average out more noise).
        backend: The model. Defaults to ``VideoSealBackend()``.
        codec: Decodes the payload bits into ``payload`` when given. Its
            ``total_bits`` must equal the payload size.
        repetition: The repetition code used when embedding, if any.
        max_p_value: Decision threshold with a repetition code, in (0, 1].
        min_mean_abs_score: Fallback decision threshold (see above).

    Returns:
        The detection result. Never raises because nothing was found.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the codec or the repetition code does not fit, or a
            threshold is out of range.
        BackendUnavailableError: If the default backend cannot run.
        ModelLoadError: If the default backend's model cannot be loaded.
    """
    check_number("max_p_value", max_p_value, minimum=0.0, maximum=1.0, exclusive_minimum=True)
    check_number("min_mean_abs_score", min_mean_abs_score, minimum=0.0)
    backend = _default_backend() if backend is None else backend
    if repetition is not None:
        _check_repetition(repetition, backend)
    payload_size = repetition.data_bits if repetition is not None else backend.payload_bits
    _check_codec(codec, payload_size)

    total: npt.NDArray[np.float64] | None = None
    count = 0
    for scores in backend.score_frames(frames):
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if values.size != backend.payload_bits or not np.all(np.isfinite(values)):
            raise ValueError("the backend returned malformed scores")
        total = values if total is None else total + values
        count += 1
    if total is None:
        return VideoWatermarkDetection(found=False, frames=0)

    mean = total / count
    mean_abs = float(np.abs(mean).mean())
    agreement: float | None = None
    p_value: float | None = None
    if repetition is not None:
        decoded = repetition.decode(mean)
        bits = decoded.bits
        if repetition.copies >= 2:
            agreement, p_value = decoded.agreement, decoded.p_value
    else:
        bits = "".join("1" if value > 0 else "0" for value in mean)

    payload: DecodedPayload | None = codec.decode(bits) if codec is not None else None
    if p_value is not None:
        found = p_value <= max_p_value
        confidence: float | None = 1.0 - p_value
    elif codec is not None and codec.check_bits and payload is not None:
        found = payload.valid
        confidence = None
    else:
        found = mean_abs >= min_mean_abs_score
        confidence = None
    return VideoWatermarkDetection(
        found=found,
        confidence=confidence,
        bits=bits if found else None,
        payload=payload if found else None,
        frames=count,
        decoded_bits=bits,
        agreement=agreement,
        p_value=p_value,
        mean_abs_score=mean_abs,
    )


def embed_verified(
    frames: Iterable[Frame],
    bits: str,
    *,
    encoder: FrameEncoder,
    backend: VideoWatermarkBackend | None = None,
    repetition: RepetitionCode | None = None,
    codec: PayloadCodec | None = None,
    verify_sampling: FrameSampling | str = FrameSampling.ALL,
    verify_fps: float = 1.0,
    verify_max_frames: int | None = None,
    max_p_value: float = DEFAULT_MAX_P_VALUE,
    tools: ExternalTools | None = None,
    limits: MediaLimits | None = None,
    temp_dir: StrPath | None = None,
    timeout_seconds: float | None = None,
) -> VerifiedVideoEmbedding:
    """Watermarks frames, encodes them with the caller's encoder and reads the mark back.

    The watermarked frames are streamed into ``encoder`` (bounded memory),
    its output is decoded with :func:`~provenance_kit.video.frames.iter_frames`
    and the watermark is detected, so the verification covers the real
    encoding the application will publish (ARCHITECTURE.md, decision 2).

    Args:
        frames: RGB ``uint8`` frames shaped ``(height, width, 3)``.
        bits: The payload (see :func:`embed_watermark`).
        encoder: Watermarked frames in, encoded media bytes out.
        backend: The model. Defaults to ``VideoSealBackend()``.
        repetition: An optional repetition code.
        codec: When given, the read-back payload must also decode as valid.
        verify_sampling: Which decoded frames to score (default: all).
        verify_fps: Sampling rate for ``FrameSampling.FPS``.
        verify_max_frames: Score at most this many decoded frames.
        max_p_value: Detection threshold with a repetition code.
        tools: Where ffmpeg/ffprobe are, for decoding the encoder's output.
        limits: Limits applied when decoding the encoder's output.
        temp_dir: Directory for a private temporary file, if one is needed.
        timeout_seconds: Timeout of each tool run.

    Returns:
        The encoded bytes and the verification outcome. Not verifying is not
        an error: check ``verified``.

    Raises:
        TypeError: If an argument has the wrong type, or the encoder does not
            return bytes.
        ValueError: If ``bits``, the codec or the repetition code do not fit.
        MediaError: If the encoder's output cannot be decoded.
        BackendUnavailableError: If a backend or binary is missing.
        ModelLoadError: If the default backend's model cannot be loaded.
        Exception: Whatever ``encoder`` raises.
    """
    if not callable(encoder):
        raise TypeError("encoder must be callable")
    backend = _default_backend() if backend is None else backend
    payload, channel = _payload_and_channel(bits, backend, repetition)
    _check_codec(codec, len(payload))

    counter = [0]

    def counted(source: Iterator[Frame]) -> Iterator[Frame]:
        for frame in source:
            counter[0] += 1
            yield frame

    encoded = encoder(counted(backend.embed_frames(frames, channel)))
    if not isinstance(encoded, (bytes, bytearray, memoryview)):
        raise TypeError("the encoder must return bytes")
    encoded_bytes = bytes(encoded)
    decoded = iter_frames(
        encoded_bytes,
        sampling=verify_sampling,
        fps=verify_fps,
        max_frames=verify_max_frames,
        tools=tools,
        limits=limits,
        temp_dir=temp_dir,
        timeout_seconds=timeout_seconds,
    )
    detection = detect_watermark(
        (frame.image for frame in decoded),
        backend=backend,
        codec=codec,
        repetition=repetition,
        max_p_value=max_p_value,
    )
    read_back = detection.decoded_bits
    bit_errors = len(payload) if read_back is None else hamming_distance(read_back, payload)
    verified = detection.found and bit_errors == 0
    if codec is not None:
        verified = verified and detection.payload is not None and detection.payload.valid
    return VerifiedVideoEmbedding(
        encoded=encoded_bytes,
        detection=detection,
        bit_errors=bit_errors,
        verified=verified,
        frames_embedded=counter[0],
    )
