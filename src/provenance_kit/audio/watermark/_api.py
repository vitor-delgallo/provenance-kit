"""High-level audio watermark functions (re-exported by the package)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from provenance_kit._typing import BytesLike, StrPath
from provenance_kit.audio._checks import check_int, check_number
from provenance_kit.audio.pcm import decode_pcm
from provenance_kit.audio.watermark.base import (
    AudioWatermarkBackend,
    AudioWatermarkDetection,
    Samples,
)
from provenance_kit.bits import hamming_distance, normalize_bits
from provenance_kit.limits import MediaLimits
from provenance_kit.payload import PayloadCodec
from provenance_kit.tools import ExternalTools

__all__ = [
    "DEFAULT_DETECTION_THRESHOLD",
    "AudioEncoder",
    "VerifiedAudioEmbedding",
    "detect_watermark",
    "embed_verified",
    "embed_watermark",
]

#: Default decision threshold on the detection probability (AudioSeal's own
#: default). Measured on unmarked synthetic signals: 0.0 to 0.27; on marked ones
#: after AAC in M4A, MP3 files and Ogg Opus: 0.995 to 1.0; after encodings that
#: shift the audio without recording the delay (raw ADTS AAC, MP3 written to a
#: pipe): 0.36 to 1.0 (see docs/audio.md).
DEFAULT_DETECTION_THRESHOLD = 0.5

#: The encoder callable of :func:`embed_verified`: watermarked PCM and its
#: sample rate in, encoded media bytes out.
AudioEncoder = Callable[[Samples, int], BytesLike]


@dataclass(frozen=True, slots=True)
class VerifiedAudioEmbedding:
    """The result of :func:`embed_verified`.

    Attributes:
        encoded: What the encoder returned, whether or not it verified.
        detection: The detection run on the decoded ``encoded`` bytes.
        bit_errors: Payload bits that differ after the round trip.
        verified: True when the watermark was found and the 16 bits came back
            exactly (and, with a codec, decode as valid).
    """

    encoded: bytes = field(repr=False)
    detection: AudioWatermarkDetection
    bit_errors: int
    verified: bool


def _default_backend() -> AudioWatermarkBackend:
    from provenance_kit.audio.watermark._audioseal import AudioSealBackend  # noqa: PLC0415

    return AudioSealBackend()


def _check_codec(codec: PayloadCodec | None, payload_bits: int) -> None:
    if codec is None:
        return
    if not isinstance(codec, PayloadCodec):
        raise TypeError("codec must be a PayloadCodec")
    if codec.total_bits != payload_bits:
        raise ValueError(
            f"the codec expects {codec.total_bits} bits but the backend carries {payload_bits}"
        )


def embed_watermark(
    samples: Samples,
    sample_rate: int,
    bits: str,
    *,
    backend: AudioWatermarkBackend | None = None,
) -> Samples:
    """Watermarks decoded audio.

    Args:
        samples: ``float32`` PCM shaped ``(channels, samples)``.
        sample_rate: The sample rate in hertz (any; the backend resamples
            internally as its model requires).
        bits: Exactly ``backend.payload_bits`` bits (16 for AudioSeal),
            typically ``codec.encode(identifier)``.
        backend: The model. Defaults to ``AudioSealBackend()`` (which loads
            pre-provisioned checkpoints and never downloads).

    Returns:
        The watermarked PCM, same shape and rate.

    Raises:
        TypeError: If ``samples`` has the wrong type.
        ValueError: If the shape, rate or bits are invalid.
        BackendUnavailableError: If the default backend cannot run.
        ModelLoadError: If the default backend's models cannot be loaded.
    """
    backend = _default_backend() if backend is None else backend
    payload = normalize_bits(bits, length=backend.payload_bits)
    check_int("sample_rate", sample_rate, minimum=1)
    return backend.embed(samples, sample_rate, payload)


def detect_watermark(
    samples: Samples,
    sample_rate: int,
    *,
    backend: AudioWatermarkBackend | None = None,
    codec: PayloadCodec | None = None,
    threshold: float = DEFAULT_DETECTION_THRESHOLD,
) -> AudioWatermarkDetection:
    """Looks for a watermark in decoded audio.

    The decision is the detector's: ``found`` when its detection probability
    reaches ``threshold``. The 16 bits are thresholded at 0.5; with only 16
    bits there is no room for error correction (ARCHITECTURE.md, decision 8),
    so give the codec check bits and read ``payload.valid``.

    Args:
        samples: ``float32`` PCM shaped ``(channels, samples)``.
        sample_rate: The sample rate in hertz.
        backend: The model. Defaults to ``AudioSealBackend()``.
        codec: Decodes the bits into ``payload`` when given; its
            ``total_bits`` must equal ``backend.payload_bits``.
        threshold: Decision threshold on the detection probability, in
            ``[0, 1]``.

    Returns:
        The detection result. Never raises because nothing was found.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the shape, rate, codec or threshold is invalid.
        BackendUnavailableError: If the default backend cannot run.
        ModelLoadError: If the default backend's models cannot be loaded.
    """
    check_number("threshold", threshold, minimum=0.0, maximum=1.0)
    check_int("sample_rate", sample_rate, minimum=1)
    backend = _default_backend() if backend is None else backend
    _check_codec(codec, backend.payload_bits)
    scores = backend.score(samples, sample_rate)
    if len(scores.bit_probabilities) != backend.payload_bits:
        raise ValueError("the backend returned the wrong number of bit probabilities")
    bits = "".join("1" if p > 0.5 else "0" for p in scores.bit_probabilities)
    found = scores.detection_probability >= threshold
    bit_confidence = sum(abs(2.0 * p - 1.0) for p in scores.bit_probabilities) / len(bits)
    payload = codec.decode(bits) if codec is not None and found else None
    return AudioWatermarkDetection(
        found=found,
        confidence=scores.detection_probability,
        bits=bits if found else None,
        payload=payload,
        decoded_bits=bits,
        bit_confidence=bit_confidence,
    )


def embed_verified(
    samples: Samples,
    sample_rate: int,
    bits: str,
    *,
    encoder: AudioEncoder,
    backend: AudioWatermarkBackend | None = None,
    codec: PayloadCodec | None = None,
    threshold: float = DEFAULT_DETECTION_THRESHOLD,
    tools: ExternalTools | None = None,
    limits: MediaLimits | None = None,
    temp_dir: StrPath | None = None,
    timeout_seconds: float | None = None,
) -> VerifiedAudioEmbedding:
    """Watermarks audio, encodes it with the caller's encoder and reads the mark back.

    The encoder's output is decoded with
    :func:`~provenance_kit.audio.pcm.decode_pcm` (at its own rate and layout)
    and the watermark is detected, so the verification covers the real
    encoding the application will publish (ARCHITECTURE.md, decision 2).

    Args:
        samples: ``float32`` PCM shaped ``(channels, samples)``.
        sample_rate: The sample rate in hertz.
        bits: The payload (``backend.payload_bits`` bits).
        encoder: Watermarked PCM and its rate in, encoded bytes out.
        backend: The model. Defaults to ``AudioSealBackend()``.
        codec: When given, the read-back payload must also decode as valid.
        threshold: Detection threshold.
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
        ValueError: If the samples, rate, bits or codec are invalid.
        MediaError: If the encoder's output cannot be decoded.
        BackendUnavailableError: If a backend or binary is missing.
        ModelLoadError: If the default backend's models cannot be loaded.
        Exception: Whatever ``encoder`` raises.
    """
    if not callable(encoder):
        raise TypeError("encoder must be callable")
    backend = _default_backend() if backend is None else backend
    payload = normalize_bits(bits, length=backend.payload_bits)
    _check_codec(codec, backend.payload_bits)
    marked = embed_watermark(samples, sample_rate, payload, backend=backend)
    encoded = encoder(marked, sample_rate)
    if not isinstance(encoded, (bytes, bytearray, memoryview)):
        raise TypeError("the encoder must return bytes")
    encoded_bytes = bytes(encoded)
    decoded = decode_pcm(
        encoded_bytes,
        tools=tools,
        limits=limits,
        temp_dir=temp_dir,
        timeout_seconds=timeout_seconds,
    )
    detection = detect_watermark(
        decoded.samples,
        decoded.sample_rate,
        backend=backend,
        codec=codec,
        threshold=threshold,
    )
    read_back = detection.decoded_bits or ""
    bit_errors = (
        hamming_distance(read_back, payload) if len(read_back) == len(payload) else len(payload)
    )
    verified = detection.found and bit_errors == 0
    if codec is not None:
        verified = verified and detection.payload is not None and detection.payload.valid
    return VerifiedAudioEmbedding(
        encoded=encoded_bytes, detection=detection, bit_errors=bit_errors, verified=verified
    )
