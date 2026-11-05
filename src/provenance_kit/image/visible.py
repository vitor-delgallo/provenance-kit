"""Visible watermark: a text tiled diagonally across the whole image.

:func:`tile_text` burns a semi-transparent text, repeated on a rotated grid,
into the pixels: for example the identity of the viewer a preview was made
for, so that a leaked screenshot names its source. It is a deterrent, not a
secure mark: anyone can crop, inpaint or paint over it. Pair it with an
invisible watermark (:mod:`provenance_kit.image.watermark`) when the source
must survive such edits.

How it looks:

* The text is drawn in ``color`` at ``opacity`` (default 0.35), rotated by
  ``angle`` degrees (counter-clockwise), repeated in rows with every other
  row shifted by half a repetition (a brick pattern), so every region of the
  image carries a copy.
* ``size`` sets the font height as a fraction of the image's shorter side
  (default 0.04), with a floor of :data:`MIN_FONT_PX` pixels; ``spacing``
  sets the gap between repetitions in multiples of that height.
* ``shadow=True`` (default) adds an outline in ``shadow_color`` shifted
  slightly down and right, so the text stays legible on light and on dark
  areas.

Fonts: pass ``font`` (a path to a TrueType/OpenType file you are licensed to
use). Without it, Pillow's built-in default font is used: when Pillow has
FreeType (the official wheels do), that is a scalable subset of *Aileron
Regular* (released under CC0 by its author, embedded in Pillow), which covers
Basic Latin (ASCII) only, so other scripts render as missing-glyph boxes;
without FreeType it is Pillow's fixed bitmap font, rendered at its native
size and scaled to ``size``, which looks blocky. provenance-kit bundles no
font.

The output has the input's size and mode (see :func:`tile_text` for the
modes supported) and is deterministic: the same input, text and parameters
give identical pixels with the same Pillow and FreeType versions.

Thread safety: :func:`tile_text` is thread-safe.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from provenance_kit._typing import StrPath

__all__ = ["MAX_TEXT_LENGTH", "MIN_FONT_PX", "tile_text"]

#: The smallest font height drawn, in pixels, whatever ``size`` says.
MIN_FONT_PX = 10

#: The longest text accepted, in characters.
MAX_TEXT_LENGTH = 256

_DIRECT_MODES = frozenset({"RGB", "RGBA", "L", "LA"})
_ROUND_TRIP_MODES = frozenset({"RGBX", "CMYK", "YCbCr", "LAB", "HSV"})
_FontType = ImageFont.FreeTypeFont | ImageFont.ImageFont | ImageFont.TransposedFont


def tile_text(
    image: Image.Image,
    text: str,
    *,
    opacity: float = 0.35,
    angle: float = 30.0,
    size: float = 0.04,
    spacing: float = 2.0,
    font: StrPath | None = None,
    color: Sequence[int] = (255, 255, 255),
    shadow: bool = True,
    shadow_color: Sequence[int] = (0, 0, 0),
) -> Image.Image:
    """Returns a copy of ``image`` with ``text`` tiled diagonally across it.

    Args:
        image: The image to mark; it is not modified. Supported modes:
            ``RGB``, ``RGBA``, ``L``, ``LA`` (composited directly; alpha is
            kept unchanged), ``RGBX``, ``CMYK``, ``YCbCr``, ``LAB``, ``HSV``
            (through RGB and back), ``P`` (through RGB and mapped back to the
            original palette, so the text colour snaps to the nearest palette
            entry; transparent pixels stay transparent) and ``1`` (through
            ``L``, then thresholded). Convert other modes (16-bit, float)
            first.
        text: The text, 1 to :data:`MAX_TEXT_LENGTH` characters; a newline
            character starts a new line.
        opacity: Text opacity, greater than 0 and at most 1.
        angle: Rotation of the text grid in degrees, counter-clockwise.
        size: Font height as a fraction of the shorter image side, greater
            than 0 and at most 1 (never below :data:`MIN_FONT_PX` pixels).
        spacing: Gap between repetitions, horizontally and vertically, in
            multiples of the font height (0 to 20).
        font: Path to a TrueType/OpenType font file, or ``None`` for Pillow's
            default font (see the module docstring).
        color: Text colour as ``(red, green, blue)``, 0 to 255.
        shadow: Draw a dark outline behind the text for legibility.
        shadow_color: Outline colour as ``(red, green, blue)``.

    Returns:
        A new image with the same size and mode.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range, the mode is not supported,
            or the font file cannot be loaded.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    _check_text(text)
    opacity = _check_number("opacity", opacity, low=0.0, high=1.0, low_inclusive=False)
    angle = _check_number("angle", angle, low=-1e6, high=1e6)
    size = _check_number("size", size, low=0.0, high=1.0, low_inclusive=False)
    spacing = _check_number("spacing", spacing, low=0.0, high=20.0)
    rgb_color = _check_color("color", color)
    rgb_shadow = _check_color("shadow_color", shadow_color)
    if not isinstance(shadow, bool):
        raise TypeError("shadow must be a bool")
    if image.mode not in _DIRECT_MODES | _ROUND_TRIP_MODES | {"P", "1"}:
        raise ValueError(f"mode {image.mode} is not supported: convert the image to RGB first")
    if image.width < 1 or image.height < 1:
        raise ValueError("cannot mark an empty image")

    font_px = max(MIN_FONT_PX, round(size * min(image.size)))
    text_mask, shadow_mask = _render_masks(
        image.size,
        text,
        font_path=font,
        font_px=font_px,
        angle=angle,
        spacing=spacing,
        shadow=shadow,
    )
    table = [round(level * opacity) for level in range(256)]
    text_alpha = text_mask.point(table)
    shadow_alpha = shadow_mask.point(table) if shadow_mask is not None else None
    return _apply(image, text_alpha, shadow_alpha, rgb_color, rgb_shadow)


# ---------------------------------------------------------------- validation


def _check_text(text: object) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if not text.strip():
        raise ValueError("text must not be empty")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"text must be at most {MAX_TEXT_LENGTH} characters")
    if any(ord(c) < 32 and c != "\n" for c in text):
        raise ValueError("text must not contain control characters other than newlines")


def _check_number(
    name: str,
    value: object,
    *,
    low: float,
    high: float,
    low_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    too_low = number < low or (not low_inclusive and number == low)
    if not math.isfinite(number) or too_low or number > high:
        bound = "greater than" if not low_inclusive else "at least"
        raise ValueError(f"{name} must be {bound} {low:g} and at most {high:g}")
    return number


def _check_color(name: str, value: object) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise TypeError(f"{name} must be a (red, green, blue) sequence")
    channels: list[int] = []
    for channel in value:
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise TypeError(f"{name} channels must be ints")
        if not 0 <= channel <= 255:
            raise ValueError(f"{name} channels must be between 0 and 255")
        channels.append(channel)
    return channels[0], channels[1], channels[2]


# ---------------------------------------------------------------- rendering


def _load_font(font: StrPath | None, font_px: int) -> tuple[_FontType, float]:
    """Returns the font and the factor to scale its rendering by."""
    if font is not None:
        try:
            return ImageFont.truetype(os.fspath(font), size=font_px), 1.0
        except (OSError, ValueError) as exc:
            raise ValueError("the font file could not be loaded") from exc
    try:
        return ImageFont.load_default(size=font_px), 1.0
    except (ImportError, OSError):
        # No FreeType: Pillow's fixed-size bitmap font, scaled afterwards.
        bitmap = ImageFont.load_default()
        _, top, _, bottom = bitmap.getbbox("Ag")
        return bitmap, font_px / max(1, bottom - top)


def _render_masks(
    image_size: tuple[int, int],
    text: str,
    *,
    font_path: StrPath | None,
    font_px: int,
    angle: float,
    spacing: float,
    shadow: bool,
) -> tuple[Image.Image, Image.Image | None]:
    font, scale = _load_font(font_path, font_px)
    native_px = font_px / scale
    stroke = max(1, round(native_px * 0.06)) if shadow else 0
    offset = max(1, round(native_px * 0.04)) if shadow else 0
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    bbox = probe.textbbox((0, 0), text, font=font, stroke_width=stroke)
    left, top = math.floor(bbox[0]), math.floor(bbox[1])
    right, bottom = math.ceil(bbox[2]), math.ceil(bbox[3])
    pad = stroke + offset + 1
    tile_size = (right - left + 2 * pad, bottom - top + 2 * pad)
    origin = (pad - left, pad - top)

    text_tile = Image.new("L", tile_size, 0)
    ImageDraw.Draw(text_tile).text(origin, text, fill=255, font=font)
    shadow_tile = None
    if shadow:
        shadow_tile = Image.new("L", tile_size, 0)
        ImageDraw.Draw(shadow_tile).text(
            (origin[0] + offset, origin[1] + offset),
            text,
            fill=255,
            font=font,
            stroke_width=stroke,
            stroke_fill=255,
        )
    if scale != 1.0:
        scaled = (max(1, round(tile_size[0] * scale)), max(1, round(tile_size[1] * scale)))
        text_tile = text_tile.resize(scaled, Image.Resampling.BILINEAR)
        if shadow_tile is not None:
            shadow_tile = shadow_tile.resize(scaled, Image.Resampling.BILINEAR)

    gap = round(spacing * font_px)
    cell = (text_tile.width + gap, text_tile.height + gap)
    # Rotate the tile once, then stamp it on a lattice rotated by the same
    # angle: the canvas never exceeds the image size.
    rotated_text = text_tile.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    positions = _lattice(image_size, cell, angle, rotated_text.size)
    text_mask = _stamp(rotated_text, image_size, positions)
    shadow_mask = None
    if shadow_tile is not None:
        rotated_shadow = shadow_tile.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        shadow_mask = _stamp(rotated_shadow, image_size, positions)
    return text_mask, shadow_mask


def _lattice(
    image_size: tuple[int, int],
    cell: tuple[int, int],
    angle: float,
    stamp_size: tuple[int, int],
) -> list[tuple[int, int]]:
    """Top-left corners of the stamps that touch the image.

    The lattice has rows ``cell[1]`` apart along the rotated vertical and
    repetitions ``cell[0]`` apart along the rotated baseline, every other row
    shifted by half a cell. Pillow rotates counter-clockwise with the y axis
    pointing down, so the baseline direction is ``(cos, -sin)``.
    """
    width, height = image_size
    cell_w, cell_h = cell
    theta = math.radians(angle)
    cos, sin = math.cos(theta), math.sin(theta)
    half_w, half_h = stamp_size[0] / 2, stamp_size[1] / 2
    reach = math.hypot(width, height) / 2 + max(stamp_size)
    rows = math.ceil(reach / cell_h) + 1
    columns = math.ceil(reach / cell_w) + 1
    positions: list[tuple[int, int]] = []
    for row in range(-rows, rows + 1):
        shift = cell_w / 2 if row % 2 else 0.0
        v = row * cell_h
        for column in range(-columns, columns + 1):
            u = column * cell_w + shift
            x = width / 2 + u * cos + v * sin
            y = height / 2 - u * sin + v * cos
            if -half_w <= x <= width + half_w and -half_h <= y <= height + half_h:
                positions.append((round(x - half_w), round(y - half_h)))
    return positions


def _stamp(
    stamp: Image.Image, image_size: tuple[int, int], positions: list[tuple[int, int]]
) -> Image.Image:
    canvas = Image.new("L", image_size, 0)
    for position in positions:
        # Using the stamp as its own mask keeps neighbouring glyphs that
        # overlap the stamp's empty corners.
        canvas.paste(stamp, position, stamp)
    return canvas


# ---------------------------------------------------------------- compositing


def _luma(color: tuple[int, int, int]) -> int:
    red, green, blue = color
    return (299 * red + 587 * green + 114 * blue + 500) // 1000


def _apply(
    image: Image.Image,
    text_alpha: Image.Image,
    shadow_alpha: Image.Image | None,
    color: tuple[int, int, int],
    shadow_color: tuple[int, int, int],
) -> Image.Image:
    """Composites the colours through the opacity-scaled masks (Pillow, integer math)."""
    mode = image.mode
    if mode in _DIRECT_MODES:
        grey = mode in ("L", "LA")
        base = image.convert("L") if grey else image.convert("RGB")
        fill_mode = "L" if grey else "RGB"
        if shadow_alpha is not None:
            fill = _luma(shadow_color) if grey else shadow_color
            base = Image.composite(Image.new(fill_mode, image.size, fill), base, shadow_alpha)
        fill = _luma(color) if grey else color
        base = Image.composite(Image.new(fill_mode, image.size, fill), base, text_alpha)
        if mode in ("RGBA", "LA"):
            base.putalpha(image.getchannel("A"))
        return base
    if mode in _ROUND_TRIP_MODES:
        marked = _apply(image.convert("RGB"), text_alpha, shadow_alpha, color, shadow_color)
        return marked.convert(mode)
    if mode == "1":
        marked = _apply(image.convert("L"), text_alpha, shadow_alpha, color, shadow_color)
        return marked.convert("1", dither=Image.Dither.NONE)
    return _apply_palette(image, text_alpha, shadow_alpha, color, shadow_color)


def _apply_palette(
    image: Image.Image,
    text_alpha: Image.Image,
    shadow_alpha: Image.Image | None,
    color: tuple[int, int, int],
    shadow_color: tuple[int, int, int],
) -> Image.Image:
    rgb = image.convert("RGBA").convert("RGB") if "transparency" in image.info else image
    marked = _apply(rgb.convert("RGB"), text_alpha, shadow_alpha, color, shadow_color)
    result = marked.quantize(palette=image, dither=Image.Dither.NONE)
    transparency = image.info.get("transparency")
    if isinstance(transparency, int):
        original = np.asarray(image)
        indices = np.asarray(result).copy()
        indices[original == transparency] = transparency
        result = Image.frombytes("P", image.size, indices.tobytes())
        result.putpalette(image.getpalette() or [])
        result.info["transparency"] = transparency
    return result
