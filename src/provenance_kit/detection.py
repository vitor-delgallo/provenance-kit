"""The common shape of a watermark detection result.

Every invisible-watermark backend (image, video, audio) returns a
:class:`WatermarkDetection`, or a frozen subclass that adds backend-specific
fields (for example per-frame scores). Detection never raises because nothing
was found: ``found=False`` says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from provenance_kit.payload import DecodedPayload

__all__ = ["WatermarkDetection"]


@dataclass(frozen=True, slots=True, kw_only=True)
class WatermarkDetection:
    """The outcome of looking for an invisible watermark.

    Subclasses must also be ``@dataclass(frozen=True, slots=True,
    kw_only=True)`` so that new fields can be added in any order.

    Attributes:
        found: The backend's decision that a watermark is present.
        confidence: The backend's detection score in ``[0.0, 1.0]`` (its
            meaning is backend-specific and documented by the backend), or
            ``None`` if the backend gives no score.
        bits: The extracted payload bits as a ``"0"``/``"1"`` string, when the
            backend extracted any. Present only when ``found`` is true.
        payload: ``bits`` decoded with the caller's
            :class:`~provenance_kit.payload.PayloadCodec`, when one was given.
            Check :attr:`DecodedPayload.valid
            <provenance_kit.payload.DecodedPayload.valid>` before trusting it.

    Raises:
        TypeError: If ``found`` is not a bool.
        ValueError: If ``confidence`` is outside ``[0, 1]``, or ``bits`` or
            ``payload`` is set while ``found`` is false.
    """

    found: bool
    confidence: float | None = None
    bits: str | None = None
    payload: DecodedPayload | None = None

    def __post_init__(self) -> None:
        """Validates the fields."""
        if not isinstance(self.found, bool):
            raise TypeError("found must be a bool")
        if self.confidence is not None and not (
            math.isfinite(self.confidence) and 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not self.found and (self.bits is not None or self.payload is not None):
            raise ValueError("bits and payload are only set when a watermark was found")
        if self.bits is not None and (not self.bits or set(self.bits) - {"0", "1"}):
            raise ValueError("bits must be a non-empty string of '0' and '1'")

    @classmethod
    def not_found(cls, confidence: float | None = None) -> WatermarkDetection:
        """Returns a "no watermark" result.

        Args:
            confidence: The detector's score, if it produced one.

        Returns:
            A result with ``found=False``.
        """
        return cls(found=False, confidence=confidence)
