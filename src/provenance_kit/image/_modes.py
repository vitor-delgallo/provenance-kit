"""Pillow mode conversions shared by the image modules (internal).

Pillow images come in many modes. The image modules work in 8-bit RGB (or
RGBA when they keep transparency). These helpers convert every mode Pillow
decodes into those, with two rules Pillow's own ``convert`` does not follow:

* 16- and 32-bit integer modes (``I;16*``, ``I``) are scaled from the 16-bit
  range (0 to 65535) down to 8 bits, instead of being clipped at 255.
* ``F`` (32-bit float) is clipped to 0 to 255, Pillow's convention.

Alpha is dropped by :func:`to_rgb` (the colour values under transparent
pixels are kept as they are, which is what Meta's PDQ tools do) and kept by
:func:`to_rgba`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["has_alpha", "to_luma", "to_rgb", "to_rgba"]

_SIXTEEN_BIT_MODES = frozenset({"I;16", "I;16L", "I;16B", "I;16N", "I"})
_ALPHA_MODES = frozenset({"RGBA", "RGBa", "LA", "La", "PA"})


def has_alpha(image: Image.Image) -> bool:
    """Whether the image has an alpha channel or palette/colour-key transparency."""
    return image.mode in _ALPHA_MODES or "transparency" in image.info


def _wide_to_luma(image: Image.Image) -> Image.Image:
    array = np.asarray(image)
    if image.mode == "F":
        scaled: NDArray[np.uint8] = np.clip(np.nan_to_num(array), 0, 255).astype(np.uint8)
    else:
        scaled = (np.clip(array.astype(np.int64), 0, 65535) >> 8).astype(np.uint8)
    return Image.fromarray(scaled)


def to_luma(image: Image.Image) -> Image.Image:
    """Converts to 8-bit greyscale (``L``), dropping alpha.

    Raises:
        ValueError: If Pillow cannot convert the mode.
    """
    if image.mode == "L":
        return image
    if image.mode in _SIXTEEN_BIT_MODES or image.mode == "F":
        return _wide_to_luma(image)
    return image.convert("L")


def to_rgb(image: Image.Image) -> Image.Image:
    """Converts to 8-bit ``RGB``, dropping alpha. Returns ``image`` if already RGB.

    Raises:
        ValueError: If Pillow cannot convert the mode.
    """
    if image.mode == "RGB":
        return image
    if image.mode in _SIXTEEN_BIT_MODES or image.mode == "F":
        return _wide_to_luma(image).convert("RGB")
    if image.mode == "P" and "transparency" in image.info:
        # Going through RGBA avoids Pillow's warning about palette
        # transparency; the alpha is then dropped.
        return image.convert("RGBA").convert("RGB")
    return image.convert("RGB")


def to_rgba(image: Image.Image) -> Image.Image:
    """Converts to 8-bit ``RGBA``. Returns ``image`` if already RGBA.

    Raises:
        ValueError: If Pillow cannot convert the mode.
    """
    if image.mode == "RGBA":
        return image
    if image.mode in _SIXTEEN_BIT_MODES or image.mode == "F":
        return _wide_to_luma(image).convert("RGBA")
    return image.convert("RGBA")
