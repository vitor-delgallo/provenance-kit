"""A repetition code with majority vote for the VideoSeal payload.

VideoSeal carries 256 bits and applies no error correction, so after lossy
encoding a few bits come back flipped. :class:`RepetitionCode` trades capacity
for robustness (ARCHITECTURE.md, decision 8): ``copies`` copies of a
``data_bits``-bit payload fill the 256 channel bits, and decoding takes a
**soft majority vote**: the detector's scores for the copies of each bit are
summed (after undoing the whitening) and the sign decides. With scores whose
magnitude reflects certainty this beats a plain vote; with ``±1`` scores it
is the plain vote (ties decode as 0).

Layout of the ``total_bits`` channel bits, before whitening::

    | copy 0 (data_bits) | copy 1 (data_bits) | ... | copy n-1 | padding |

``padding = total_bits - copies * data_bits`` bits (zero for powers of two)
are fixed and ignored by the decoder.

Whitening (on by default): every channel bit is XORed with a fixed
pseudo-random mask (derived from SHAKE-256 of a constant label, the same for
everyone: it is not a key). Each copy therefore lands on a different bit
pattern, which has two effects. Low-entropy payloads (small identifiers are
mostly zeros) do not become long runs of equal bits. And on unmarked content,
whose scores follow the content rather than any payload, the copies of a bit
agree no more often than independent fair coins. That makes the copies'
agreement a calibrated detection statistic: :meth:`RepetitionCode.decode`
returns the probability (``p_value``) that unmarked content agrees at least
as much as observed, which :func:`~provenance_kit.video.watermark.detect_watermark`
uses for its decision.

Pair the code with a :class:`~provenance_kit.payload.PayloadCodec` of
``data_bits`` total bits, for example::

    code = RepetitionCode(copies=4)  # 64 data bits
    codec = PayloadCodec(code.data_bits, version_bits=4, check_bits=12)

The bit accuracy obtained after real codecs is measured in
``tests/video/test_watermark_integration.py`` and reported in
``docs/video.md``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from provenance_kit.bits import BitsLike, normalize_bits

__all__ = ["RepetitionCode", "RepetitionDecoded"]

_MASK_LABEL = b"provenance-kit/video-watermark/repetition-mask/v1"


@dataclass(frozen=True, slots=True)
class RepetitionDecoded:
    """The result of decoding channel scores with a :class:`RepetitionCode`.

    Attributes:
        bits: The decoded data bits (``data_bits`` characters).
        agreement: Fraction of the copy bits (each copy's score sign) that
            agree with the decoded bit, in ``[0.0, 1.0]`` (1.0: every copy of
            every bit agreed). Because bits are decoded by the soft sum, a
            few confident copies can outvote many unsure ones, so this can
            fall below 0.5.
        p_value: The detection statistic: the probability that ``copies``
            independent fair bits per data bit agree among themselves at
            least as much as observed, where a data bit's agreement is the
            size of its hard majority (``max(ones, copies - ones)`` over the
            copies' score signs). Small values mean a watermark is present.
            It is computed from the hard majority, not from :attr:`agreement`,
            because that is the quantity whose null distribution is exact.
            1.0 when there is a single copy (no information).
    """

    bits: str
    agreement: float
    p_value: float


@dataclass(frozen=True, slots=True)
class RepetitionCode:
    """Repeats a payload ``copies`` times across the backend's channel bits.

    Attributes:
        copies: Number of copies, from 1 (no repetition) to ``total_bits``.
        total_bits: The backend's channel capacity (256 for VideoSeal).
        whiten: XOR the channel bits with a fixed pseudo-random mask (see the
            module docstring). Embedder and detector must agree on it.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``copies`` is out of range.
    """

    copies: int = 4
    total_bits: int = 256
    whiten: bool = True

    def __post_init__(self) -> None:
        """Validates the fields."""
        for name in ("copies", "total_bits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
        if not isinstance(self.whiten, bool):
            raise TypeError("whiten must be a bool")
        if not 1 <= self.total_bits <= 4096:
            raise ValueError("total_bits must be between 1 and 4096")
        if not 1 <= self.copies <= self.total_bits:
            raise ValueError("copies must be between 1 and total_bits")

    @property
    def data_bits(self) -> int:
        """The payload size: ``total_bits // copies``."""
        return self.total_bits // self.copies

    @property
    def padding_bits(self) -> int:
        """Channel bits left over after the copies (fixed, ignored)."""
        return self.total_bits - self.copies * self.data_bits

    def encode(self, bits: BitsLike) -> str:
        """Builds the channel bits for a payload.

        Args:
            bits: Exactly :attr:`data_bits` payload bits.

        Returns:
            :attr:`total_bits` channel bits.

        Raises:
            TypeError: If ``bits`` is not a bit sequence.
            ValueError: If ``bits`` has the wrong length or symbols.
        """
        payload = normalize_bits(bits, length=self.data_bits)
        plain = payload * self.copies + "0" * self.padding_bits
        if not self.whiten:
            return plain
        mask = _mask(self.total_bits)
        return "".join("1" if (c == "1") != m else "0" for c, m in zip(plain, mask, strict=True))

    def decode(self, scores: Sequence[float] | npt.NDArray[np.floating]) -> RepetitionDecoded:
        """Decodes channel scores by soft majority vote.

        Args:
            scores: One score per channel bit (:attr:`total_bits` values);
                positive means 1, negative means 0, and the magnitude is the
                certainty. Hard bits work too, as ``+1``/``-1`` values.

        Returns:
            The decoded payload, the copies' agreement and its p-value.

        Raises:
            ValueError: If ``scores`` has the wrong length or non-finite values.
        """
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if values.size != self.total_bits:
            raise ValueError(f"expected {self.total_bits} scores, got {values.size}")
        if not np.all(np.isfinite(values)):
            raise ValueError("scores must be finite")
        if self.whiten:
            signs = np.where(np.array(_mask(self.total_bits), dtype=bool), -1.0, 1.0)
            values = values * signs
        copies = values[: self.copies * self.data_bits].reshape(self.copies, self.data_bits)
        decoded = copies.sum(axis=0) > 0
        hard = copies > 0
        # Agreement with the decoded (soft-vote) bits, as documented.
        agreement = int(np.count_nonzero(hard == decoded[None, :])) / hard.size
        # The detection statistic uses the hard majority of each bit's copies,
        # whose distribution under the null hypothesis is known exactly.
        ones = hard.sum(axis=0)
        majority_agree = int(np.maximum(ones, self.copies - ones).sum())
        bits = "".join("1" if bit else "0" for bit in decoded)
        return RepetitionDecoded(
            bits=bits,
            agreement=agreement,
            p_value=_agreement_p_value(self.copies, self.data_bits, majority_agree),
        )


@lru_cache(maxsize=16)
def _mask(total_bits: int) -> tuple[bool, ...]:
    stream = hashlib.shake_256(_MASK_LABEL).digest((total_bits + 7) // 8)
    bits = format(int.from_bytes(stream, "big"), f"0{8 * len(stream)}b")
    return tuple(c == "1" for c in bits[:total_bits])


@lru_cache(maxsize=64)
def _agreement_distribution(copies: int, data_bits: int) -> tuple[float, ...]:
    """Distribution of the summed per-bit agreement under the null hypothesis.

    For one data bit with ``copies`` independent fair copy bits, the agreement
    is ``max(m, copies - m)`` with ``m ~ Binomial(copies, 1/2)``; the total
    over ``data_bits`` independent bits is the ``data_bits``-fold convolution.
    """
    single = np.zeros(copies + 1, dtype=np.float64)
    for ones in range(copies + 1):
        single[max(ones, copies - ones)] += math.comb(copies, ones) / 2.0**copies
    total = np.array([1.0])
    power = single
    remaining = data_bits
    while remaining:  # exponentiation by squaring keeps the convolutions few
        if remaining & 1:
            total = np.convolve(total, power)
        remaining >>= 1
        if remaining:
            power = np.convolve(power, power)
    return tuple(float(v) for v in total)


def _agreement_p_value(copies: int, data_bits: int, total_agree: int) -> float:
    if copies == 1:
        return 1.0
    distribution = _agreement_distribution(copies, data_bits)
    tail = float(np.sum(distribution[total_agree:]))
    return min(max(tail, 0.0), 1.0)
