"""High-level audio watermark functions (re-exported by the package)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from provenance_kit._checks import check_int, check_number
from provenance_kit._typing import BytesLike, StrPath
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
    "MAX_ALIGNMENT_CANDIDATES",
    "AlignmentSearch",
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

#: The most offsets one :class:`AlignmentSearch` may try (bounds its cost).
MAX_ALIGNMENT_CANDIDATES = 4096

_MAX_ALIGNMENT_OFFSET = 192_000  # one second at the highest supported rate


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignmentSearch:
    """An opt-in search over sample offsets before detection.

    AudioSeal is sensitive to sample-level time shifts (a shift of 3 samples
    at 16 kHz breaks detection), and some pipelines shift the audio without
    recording it: raw ADTS AAC and MP3 written to a pipe start with the
    encoder delay (1024 samples for ffmpeg's AAC encoder, about 1105 for
    LAME). The search tries dropping ``0, step_samples, 2 * step_samples,
    ...`` up to ``max_offset_samples`` leading samples (samples at the
    signal's own rate), scores the first ``window_seconds`` after each
    offset, and keeps the offset whose bits are the most confident (highest
    mean ``|2p - 1|``; ties: higher detection probability, then smaller
    offset). If even that offset stays below ``min_bit_confidence``, nothing
    stands out and the signal is detected unshifted (offset 0). The detection
    probability alone cannot pick the offset: it saturates near 1 over a wide
    range of misalignments, while the bit confidence peaks near the true
    delay.

    Measured (``docs/audio.md``): on broadband content after raw ADTS AAC and
    MP3 to a pipe, the search lands within 10 samples of the encoder delay
    and all 16 bits come back (without it: 8 to 12 of 16). Best excerpt
    confidence 0.36 to 0.44 there, and 0.04 to 0.05 on unmarked audio, hence
    the default floor of 0.2: without it, the near-random best offset of an
    unmarked tonal signal raised its detection probability from 0.25 to
    0.62. On sparse tonal content the bit confidence is not a reliable guide
    (the 16 bits do not survive there even when aligned).

    Cost: ``max_offset_samples // step_samples + 1`` backend calls on a
    ``window_seconds`` excerpt, plus the final detection. With the defaults
    that is 1,025 calls on 0.5 s of audio: about 20 to 40 s per search with
    AudioSeal on the reference CPU (44.1 kHz input; about 20 to 40 ms per
    call, most of it the detector, the rest resampling to 16 kHz). The cost
    is linear in the number of offsets: narrow the range when the pipeline's
    delay is known, or raise the step (the detector tolerates about one
    sample of error at 16 kHz, so a step of 2 at 44.1 or 48 kHz is safe).

    Attributes:
        max_offset_samples: The largest offset tried (default 2048: covers
            the AAC and MP3 encoder delays; about 46 ms at 44.1 kHz).
        step_samples: The distance between tried offsets (default 2).
        window_seconds: The length of the excerpt scored at each offset
            (default 0.5 s).
        min_bit_confidence: The excerpt bit confidence an offset needs to be
            used (default 0.2); below it the signal is detected unshifted.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a field is out of range, or the search would try more
            than :data:`MAX_ALIGNMENT_CANDIDATES` offsets.
    """

    max_offset_samples: int = 2048
    step_samples: int = 2
    window_seconds: float = 0.5
    min_bit_confidence: float = 0.2

    def __post_init__(self) -> None:
        """Validates the fields."""
        floor = check_number(
            "min_bit_confidence", self.min_bit_confidence, minimum=0.0, maximum=1.0
        )
        object.__setattr__(self, "min_bit_confidence", floor)
        check_int(
            "max_offset_samples", self.max_offset_samples, minimum=0, maximum=_MAX_ALIGNMENT_OFFSET
        )
        check_int("step_samples", self.step_samples, minimum=1)
        seconds = check_number("window_seconds", self.window_seconds, minimum=0.1, maximum=60.0)
        object.__setattr__(self, "window_seconds", seconds)
        if self.candidates > MAX_ALIGNMENT_CANDIDATES:
            raise ValueError(
                f"the search would try {self.candidates} offsets; at most "
                f"{MAX_ALIGNMENT_CANDIDATES} are allowed (raise step_samples)"
            )

    @property
    def candidates(self) -> int:
        """How many offsets the search tries."""
        return self.max_offset_samples // self.step_samples + 1

    def offsets(self) -> range:
        """Returns the offsets tried, in order."""
        return range(0, self.max_offset_samples + 1, self.step_samples)


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
    alignment: AlignmentSearch | None = None,
) -> AudioWatermarkDetection:
    """Looks for a watermark in decoded audio.

    The decision is the detector's: ``found`` when its detection probability
    reaches ``threshold``. The 16 bits are thresholded at 0.5; with only 16
    bits there is no room for error correction (ARCHITECTURE.md, decision 8),
    so give the codec check bits and read ``payload.valid``.

    With ``alignment``, an :class:`AlignmentSearch` first chooses how many
    leading samples to drop (for audio shifted by an unsignaled encoder
    delay); the result's ``offset_samples`` says which. It multiplies the
    cost (see :class:`AlignmentSearch`).

    Args:
        samples: ``float32`` PCM shaped ``(channels, samples)``.
        sample_rate: The sample rate in hertz.
        backend: The model. Defaults to ``AudioSealBackend()``.
        codec: Decodes the bits into ``payload`` when given; its
            ``total_bits`` must equal ``backend.payload_bits``.
        threshold: Decision threshold on the detection probability, in
            ``[0, 1]``.
        alignment: An optional search over sample offsets (opt-in; off by
            default).

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
    if alignment is not None and not isinstance(alignment, AlignmentSearch):
        raise TypeError("alignment must be an AlignmentSearch or None")
    backend = _default_backend() if backend is None else backend
    _check_codec(codec, backend.payload_bits)
    offset: int | None = None
    if alignment is not None:
        offset = _best_offset(samples, sample_rate, backend, alignment)
        samples = samples[:, offset:]
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
        offset_samples=offset,
    )


def _bit_confidence(probabilities: tuple[float, ...]) -> float:
    return sum(abs(2.0 * p - 1.0) for p in probabilities) / max(len(probabilities), 1)


def _best_offset(
    samples: Samples,
    sample_rate: int,
    backend: AudioWatermarkBackend,
    alignment: AlignmentSearch,
) -> int:
    """Returns the offset whose excerpt has the most confident bits."""
    shape = getattr(samples, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[1] < 1:
        raise ValueError("samples must be shaped (channels, samples) with at least one sample")
    length = int(shape[1])
    window = max(1, round(alignment.window_seconds * sample_rate))
    best_offset = 0
    best_key: tuple[float, float, int] | None = None
    for offset in alignment.offsets():
        if offset >= length:
            break
        scores = backend.score(samples[:, offset : offset + window], sample_rate)
        key = (_bit_confidence(scores.bit_probabilities), scores.detection_probability, -offset)
        if best_key is None or key > best_key:
            best_key, best_offset = key, offset
    if best_key is None or best_key[0] < alignment.min_bit_confidence:
        # No offset stands out: shifting by a near-random offset could only
        # add false positives, so detect the signal as it is.
        return 0
    return best_offset


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
