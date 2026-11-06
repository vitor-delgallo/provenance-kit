"""Visible tiled text watermark."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image, ImageFont, features

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


def _palette_image_with_white_transparent_entry():
    # Entry 0 is the transparent one and its RGB is white: white text drawn
    # on opaque grey is nearest to it.
    palette = [255, 255, 255, 0, 0, 0, 128, 128, 128] + [0, 0, 0] * 253
    indices = np.full((200, 300), 2, np.uint8)
    indices[:, :100] = 0  # transparent left third
    image = Image.frombytes("P", (300, 200), indices.tobytes())
    image.putpalette(palette)
    return image, indices


def test_opaque_palette_pixels_never_become_transparent():
    image, indices = _palette_image_with_white_transparent_entry()
    image.info["transparency"] = 0
    marked = tile_text(image, "PREVIEW 2026", opacity=0.9, color=(255, 255, 255))
    result = np.asarray(marked)
    opaque = indices != 0
    assert np.count_nonzero(opaque & (result == 0)) == 0
    assert np.all(result[~opaque] == 0)
    assert np.any(result[opaque] != 2)  # the text is visible (drawn with another entry)
    assert marked.info["transparency"] == 0
    assert marked.getpalette() == image.getpalette()


def test_palette_alpha_tables_are_kept_and_respected():
    # A tRNS table (bytes): entry 0 fully transparent, entry 1 half transparent.
    image, indices = _palette_image_with_white_transparent_entry()
    indices[:20, 100:] = 1
    image = Image.frombytes("P", (300, 200), indices.tobytes())
    image.putpalette([255, 255, 255, 0, 0, 0, 128, 128, 128] + [0, 0, 0] * 253)
    image.info["transparency"] = b"\x00\x80"
    marked = tile_text(image, "PREVIEW 2026", opacity=0.9, color=(255, 255, 255))
    result = np.asarray(marked)
    assert marked.info["transparency"] == b"\x00\x80"
    see_through = np.isin(indices, (0, 1))
    assert np.array_equal(result[see_through], indices[see_through])
    assert not np.any(np.isin(result[~see_through], (0, 1)))


def test_greyscale_uses_luma_of_the_colour():
    marked = tile_text(
        Image.new("L", (200, 200), 0), "abc", color=(255, 0, 0), opacity=1.0, shadow=False
    )
    assert np.asarray(marked).max() == round(0.299 * 255)


def test_multiline_text():
    marked = tile_text(_gradient(), "viewer 42\n2026-11-05")
    assert marked.size == (320, 200)


def _default_font_file(tmp_path):
    """Pillow's embedded default font, written to a file (skips without FreeType)."""
    data = getattr(ImageFont.load_default(size=20), "font_bytes", None)
    if not data:
        pytest.skip("this Pillow build has no FreeType default font")
    path = tmp_path / "font.ttf"
    path.write_bytes(data)
    return path


def test_custom_font_file_is_loaded_and_drawn(tmp_path, monkeypatch):
    # Pillow renders its default font with the basic layout, but a font file
    # loaded with truetype() with Raqm when libraqm is available (Linux
    # wheels), so the same font file can give other pixels than the default
    # font depending on the platform. Pin the basic layout here to compare
    # like with like: then the file's glyphs must be exactly the default's.
    path = _default_font_file(tmp_path)
    via_default = tile_text(_gradient(), "viewer 42")  # before the spy: it uses truetype too
    real_truetype = ImageFont.truetype
    calls = []

    def basic_layout(font, size=10, *args, **kwargs):
        calls.append((font, size, kwargs))
        return real_truetype(font, size, *args, layout_engine=ImageFont.Layout.BASIC)

    monkeypatch.setattr(visible.ImageFont, "truetype", basic_layout)
    via_path = tile_text(_gradient(), "viewer 42", font=path)
    assert calls == [(str(path), visible.MIN_FONT_PX, {})]  # the file, at the computed size
    assert _digest(via_path) == _digest(via_default)
    # The font object loaded from the file is the one drawn: another size
    # from the same file changes the pixels.
    calls.clear()
    larger = tile_text(_gradient(), "viewer 42", font=path, size=0.2)
    assert calls == [(str(path), 40, {})]
    assert _digest(larger) != _digest(via_default)


def test_custom_fonts_keep_pillows_layout_engine(tmp_path):
    # The library must not force the basic layout on custom fonts: Raqm
    # shapes complex scripts (Arabic, Devanagari) correctly where available.
    path = _default_font_file(tmp_path)
    font, scale = visible._load_font(path, 24)
    expected = ImageFont.Layout.RAQM if features.check_feature("raqm") else ImageFont.Layout.BASIC
    assert isinstance(font, ImageFont.FreeTypeFont)
    assert font.layout_engine == expected
    assert scale == 1.0
    marked = tile_text(_gradient(), "viewer 42", font=path)
    assert _digest(marked) == _digest(tile_text(_gradient(), "viewer 42", font=path))


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
        ({"angle": 10**400}, ValueError),  # an int beyond the float range
        ({"spacing": -(10**400)}, ValueError),
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
