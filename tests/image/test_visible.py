"""Visible tiled text watermark."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image, ImageFont

from provenance_kit.image import visible
from provenance_kit.image.visible import MAX_TEXT_LENGTH, tile_text


def _gradient(width=320, height=200):
    x = np.linspace(0, 255, width, dtype=np.float64)[None, :]
    y = np.linspace(0, 255, height, dtype=np.float64)[:, None]
    rgb = np.stack(
        [
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            np.full((height, width), 80.0),
        ],
        axis=2,
    )
    return Image.fromarray(rgb.astype(np.uint8))


def _digest(image):
    return hashlib.sha256(image.tobytes() + image.mode.encode()).hexdigest()


def test_output_keeps_size_and_mode_and_marks_pixels():
    image = _gradient()
    marked = tile_text(image, "viewer 42")
    assert marked.size == image.size
    assert marked.mode == "RGB"
    changed = np.any(np.asarray(marked) != np.asarray(image), axis=2)
    # The tiling covers the whole image: every quarter has marked pixels.
    h, w = changed.shape
    for quarter in (
        changed[: h // 2, : w // 2],
        changed[: h // 2, w // 2 :],
        changed[h // 2 :, : w // 2],
        changed[h // 2 :, w // 2 :],
    ):
        assert quarter.mean() > 0.02
    assert 0.03 < changed.mean() < 0.8


def test_input_is_not_modified():
    image = _gradient()
    before = image.tobytes()
    tile_text(image, "viewer 42")
    assert image.tobytes() == before


def test_deterministic():
    image = _gradient()
    first = tile_text(image, "viewer 42", angle=-20, spacing=1.0)
    second = tile_text(image.copy(), "viewer 42", angle=-20, spacing=1.0)
    assert _digest(first) == _digest(second)


def test_opacity_controls_strength():
    image = Image.new("RGB", (300, 200), (0, 0, 0))
    faint = np.asarray(tile_text(image, "abc", opacity=0.2, shadow=False))
    solid = np.asarray(tile_text(image, "abc", opacity=1.0, shadow=False))
    assert faint.max() == pytest.approx(round(0.2 * 255), abs=1)
    assert solid.max() == 255


def test_shadow_darkens_light_backgrounds():
    white = Image.new("RGB", (300, 200), (255, 255, 255))
    without = np.asarray(tile_text(white, "abc", shadow=False))
    with_shadow = np.asarray(tile_text(white, "abc", shadow=True))
    # White text on white is invisible without the shadow; the outline shows.
    assert without.min() == 255
    assert with_shadow.min() < 200


def test_size_scales_with_the_image():
    small = tile_text(Image.new("L", (300, 300), 0), "W", size=0.1, shadow=False, opacity=1.0)
    large = tile_text(Image.new("L", (900, 900), 0), "W", size=0.1, shadow=False, opacity=1.0)
    # Same relative size and spacing: the inked fraction is about the same.
    small_ink = np.asarray(small).mean() / 255
    large_ink = np.asarray(large).mean() / 255
    assert small_ink == pytest.approx(large_ink, rel=0.25)


def test_minimum_font_size_applies_to_tiny_images():
    marked = tile_text(Image.new("L", (40, 30), 0), "x", size=0.01, opacity=1.0, shadow=False)
    assert np.asarray(marked).max() > 0


@pytest.mark.parametrize(
    "mode", ["RGB", "RGBA", "L", "LA", "RGBX", "CMYK", "YCbCr", "LAB", "HSV", "1"]
)
def test_supported_modes_round_trip(mode):
    source = _gradient()
    image = source.convert("RGBA") if mode == "RGBA" else source.convert(mode)
    marked = tile_text(image, "viewer 42")
    assert marked.mode == mode
    assert marked.size == image.size


def test_alpha_channel_is_kept():
    image = _gradient().convert("RGBA")
    alpha = Image.linear_gradient("L").resize(image.size)
    image.putalpha(alpha)
    marked = tile_text(image, "viewer 42")
    assert np.array_equal(np.asarray(marked.getchannel("A")), np.asarray(alpha))


def test_palette_images_keep_their_palette_and_transparency():
    image = _gradient().quantize(colors=32)
    image.info["transparency"] = 0
    marked = tile_text(image, "viewer 42", color=(255, 255, 255))
    assert marked.mode == "P"
    assert marked.getpalette() == image.getpalette()
    original = np.asarray(image)
    assert np.all(np.asarray(marked)[original == 0] == 0)
    assert marked.info["transparency"] == 0


def test_greyscale_uses_luma_of_the_colour():
    marked = tile_text(
        Image.new("L", (200, 200), 0), "abc", color=(255, 0, 0), opacity=1.0, shadow=False
    )
    assert np.asarray(marked).max() == round(0.299 * 255)


def test_multiline_text():
    marked = tile_text(_gradient(), "viewer 42\n2026-11-05")
    assert marked.size == (320, 200)


def test_custom_font_file(tmp_path):
    # The truetype loader accepts any font path; reuse Pillow's default font bytes.
    font = ImageFont.load_default(size=20)
    data = font.font_bytes if hasattr(font, "font_bytes") else None
    if not data:
        pytest.skip("this Pillow build has no FreeType default font")
    path = tmp_path / "font.ttf"
    path.write_bytes(data)
    via_path = tile_text(_gradient(), "viewer 42", font=path)
    via_default = tile_text(_gradient(), "viewer 42")
    assert _digest(via_path) == _digest(via_default)


def test_bad_font_file_is_a_value_error(tmp_path):
    path = tmp_path / "broken.ttf"
    path.write_bytes(b"not a font")
    with pytest.raises(ValueError, match="font"):
        tile_text(_gradient(), "x", font=path)
    with pytest.raises(ValueError, match="font"):
        tile_text(_gradient(), "x", font=tmp_path / "missing.ttf")


def test_bitmap_font_fallback_scales(monkeypatch):
    real = ImageFont.load_default

    def no_freetype(size=None):
        if size is not None:
            raise ImportError("FreeType is not available")
        return real()

    monkeypatch.setattr(visible.ImageFont, "load_default", no_freetype)
    marked = tile_text(Image.new("L", (400, 400), 0), "abc", size=0.1, shadow=False, opacity=1.0)
    assert np.asarray(marked).max() > 0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"text": ""}, ValueError),
        ({"text": "   "}, ValueError),
        ({"text": "a" * (MAX_TEXT_LENGTH + 1)}, ValueError),
        ({"text": "a\x00b"}, ValueError),
        ({"text": 5}, TypeError),
        ({"opacity": 0}, ValueError),
        ({"opacity": 1.5}, ValueError),
        ({"opacity": float("nan")}, ValueError),
        ({"opacity": "0.3"}, TypeError),
        ({"size": 0}, ValueError),
        ({"size": 2}, ValueError),
        ({"spacing": -1}, ValueError),
        ({"angle": float("inf")}, ValueError),
        ({"color": (255, 255)}, TypeError),
        ({"color": (255, 255, 256)}, ValueError),
        ({"color": "white"}, TypeError),
        ({"shadow_color": (0, 0, 0.5)}, TypeError),
        ({"shadow": 1}, TypeError),
    ],
)
def test_argument_validation(kwargs, error):
    arguments = {"text": "viewer 42", **kwargs}
    text = arguments.pop("text")
    with pytest.raises(error):
        tile_text(_gradient(), text, **arguments)


@pytest.mark.parametrize("mode", ["I;16", "I", "F", "PA"])
def test_unsupported_modes(mode):
    image = Image.new(mode, (20, 20))
    with pytest.raises(ValueError, match="mode"):
        tile_text(image, "x")


def test_rejects_non_images_and_empty_images():
    with pytest.raises(TypeError):
        tile_text(np.zeros((4, 4, 3), np.uint8), "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tile_text(Image.new("RGB", (0, 5)), "x")
