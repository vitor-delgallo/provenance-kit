"""The video watermark backend Protocol and the detection result."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from provenance_kit.detection import WatermarkDetection

__all__ = ["Frame", "VideoWatermarkBackend", "VideoWatermarkDetection"]

#: A decoded video frame: ``uint8``, shape ``(height, width, 3)``, RGB.
Frame = npt.NDArray[np.uint8]


class VideoWatermarkBackend(Protocol):
    """What the video watermark functions need from a model.

    Backends work on decoded frames in memory and never encode (decision 2).
    Both methods take and return iterators so that a backend can batch frames
    internally and keep memory bounded on long videos.
    """

    @property
    def payload_bits(self) -> int:
        """The number of channel bits the model embeds (256 for VideoSeal)."""
        ...

    def embed_frames(self, frames: Iterable[Frame], bits: str) -> Iterator[Frame]:
        """Watermarks frames with ``bits``.

        Args:
            frames: RGB ``uint8`` frames shaped ``(height, width, 3)``, in
                presentation order.
            bits: Exactly :attr:`payload_bits` channel bits.

        Returns:
            The watermarked frames, same order, shapes and dtype.
        """
        ...

    def score_frames(self, frames: Iterable[Frame]) -> Iterator[npt.NDArray[np.float32]]:
        """Scores the channel bits in each frame.

        Args:
            frames: RGB ``uint8`` frames shaped ``(height, width, 3)``.

        Returns:
            One ``float32`` array of :attr:`payload_bits` scores per frame:
            positive means bit 1, negative bit 0, larger magnitude means more
            certain.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoWatermarkDetection(WatermarkDetection):
    """The outcome of looking for a video watermark.

    ``confidence`` is ``1 - p_value`` when a repetition code with two or more
    copies was used, else ``None``. ``bits`` are the decoded payload bits (the
    repetition code's data bits, or the raw channel bits without one), set
    only when ``found`` is true; ``payload`` is set when a codec was given.

    Attributes:
        frames: Number of frames scored.
        decoded_bits: The decoded payload bits, also when ``found`` is false
            (useful to inspect a near miss; do not trust them then).
        agreement: Fraction of copy bits agreeing with the decoded bits
            (repetition code only), else ``None``.
        p_value: Probability that the copies of unmarked content agree among
            themselves (each data bit's hard majority) at least as much as
            observed (repetition code only), else ``None``; see
            :class:`~provenance_kit.video.watermark.RepetitionDecoded`.
        mean_abs_score: Mean magnitude of the frame-averaged channel scores:
            a weak, uncalibrated strength indicator (VideoSeal: about 0.05 to
            0.25 on unmarked frames, above 1 on freshly marked ones).
    """

    frames: int
    decoded_bits: str | None = None
    agreement: float | None = None
    p_value: float | None = None
    mean_abs_score: float = 0.0
