"""The audio watermark backend Protocol, scores and detection result."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from provenance_kit.detection import WatermarkDetection

__all__ = ["AudioScores", "AudioWatermarkBackend", "AudioWatermarkDetection", "Samples"]

#: Decoded audio: ``float32``, shape ``(channels, samples)``, values in [-1, 1].
Samples = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class AudioScores:
    """What a backend's detector reports about a signal.

    Attributes:
        detection_probability: The detector's score that the audio is
            watermarked, in ``[0, 1]`` (AudioSeal: the fraction of its
            per-sample decisions that say "watermarked").
        bit_probabilities: The probability that each payload bit is 1.

    Raises:
        ValueError: If a value is outside ``[0, 1]``.
    """

    detection_probability: float
    bit_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validates the fields."""
        values = (self.detection_probability, *self.bit_probabilities)
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
            raise ValueError("probabilities must be between 0 and 1")
        object.__setattr__(self, "bit_probabilities", tuple(self.bit_probabilities))


class AudioWatermarkBackend(Protocol):
    """What the audio watermark functions need from a model.

    Backends work on decoded PCM in memory at any sample rate (they resample
    internally as their model requires) and never encode (decision 2).
    """

    @property
    def payload_bits(self) -> int:
        """The number of payload bits the model carries (16 for AudioSeal)."""
        ...

    def embed(self, samples: Samples, sample_rate: int, bits: str) -> Samples:
        """Watermarks audio.

        Args:
            samples: ``float32`` PCM shaped ``(channels, samples)``.
            sample_rate: The sample rate in hertz.
            bits: Exactly :attr:`payload_bits` bits.

        Returns:
            The watermarked PCM: same shape, rate and dtype.
        """
        ...

    def score(self, samples: Samples, sample_rate: int) -> AudioScores:
        """Runs the detector.

        Args:
            samples: ``float32`` PCM shaped ``(channels, samples)``.
            sample_rate: The sample rate in hertz.

        Returns:
            The detection probability and the per-bit probabilities.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioWatermarkDetection(WatermarkDetection):
    """The outcome of looking for an audio watermark.

    ``confidence`` is the detector's ``detection_probability``; ``found`` is
    ``confidence >= threshold``. ``bits`` (16 for AudioSeal) and ``payload``
    are set only when ``found``.

    ``found`` alone is **not evidence** of a watermark: unmarked periodic
    signals fool the AudioSeal detector (impulse trains at 50 Hz score up to
    1.0; see ``docs/audio.md``). AudioSeal has no room for error correction
    (decision 8), so the evidence is the codec's check bits: rely on
    :attr:`verified`.

    Attributes:
        decoded_bits: The thresholded bits, also when ``found`` is false
            (meaningless on unmarked audio).
        bit_confidence: Mean of ``|2p - 1|`` over the bit probabilities: 1.0
            when every bit is certain, 0.0 when all are coin flips.
        offset_samples: With an alignment search, the number of leading
            samples dropped before detection: the offset the search chose,
            or 0 when no offset stood out or the best one was not
            corroborated (see :class:`~provenance_kit.audio.watermark.AlignmentSearch`);
            ``None`` without a search.
    """

    decoded_bits: str | None = None
    bit_confidence: float = 0.0
    offset_samples: int | None = None

    @property
    def verified(self) -> bool:
        """Whether the detection is corroborated by the payload's check bits.

        True only when ``found`` is true **and** the bits were decoded with a
        :class:`~provenance_kit.payload.PayloadCodec` that has check bits,
        and they match (``payload.check_ok``) and the version is known.
        Without a codec, or with a codec that has no check bits, this is
        always false: nothing but the detector's own score would back the
        result.

        It is strong evidence, not proof. With ``check_bits=c``, an unmarked
        signal that the detector finds passes with probability ``2**-c``,
        and up to ``2 * 2**-c`` with an
        :class:`~provenance_kit.audio.watermark.AlignmentSearch` (the
        shifted and the unshifted payload each get a chance). The bits are
        deterministic, so this is a property of the signal, not of the call:
        an unmarked 2 s, 170 Hz band-limited pulse train is found and passes
        4 check bits (identifier 3919) at 16, 44.1 and 48 kHz alike. The 16
        bits are shared: 8 check bits (1 in 256) leave 256 identifiers, 4
        check bits (1 in 16) leave 4,096. Measured rates and the
        recommended setup are in ``docs/audio.md``.
        """
        payload = self.payload
        return self.found and payload is not None and payload.check_ok is True and payload.valid
