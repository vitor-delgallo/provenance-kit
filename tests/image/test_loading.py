"""Safe image loading: formats, modes, orientation, limits and hostile inputs."""

from __future__ import annotations

import dataclasses
import io
import random
import struct
import zlib

import numpy as np
import pytest
from PIL import Image, ImageCms, PngImagePlugin, features

from provenance_kit.errors import (
    MediaDecodeError,
    MediaError,
    MediaLimitExceededError,
    UnsupportedMediaError,
)
from provenance_kit.image.loading import (
    ImageMode,
    iter_frames,
    load_image,
    probe_image,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType
from tests.support import SEED


def _encode(image, fmt, **options):
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return buffer.getvalue()


def _sample(width=64, height=48, mode="RGB"):
    rng = np.random.default_rng(SEED)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    return image if mode == "RGB" else image.convert(mode)


def _png_chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def _png_header_only(width, height):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(b"\x00" * 16))
        + _png_chunk(b"IEND", b"")
    )


def _gif_with_frames(count, width=1, height=1):
    header = b"GIF89a" + struct.pack("<HH", width, height) + b"\x80\x00\x00"
    palette = b"\x00\x00\x00\xff\xff\xff"
    frame = b"\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00"
    return header + palette + frame * count + b"\x3b"


# ---------------------------------------------------------------- formats


@pytest.mark.parametrize(
    ("fmt", "media_type", "options"),
    [
        ("JPEG", MediaType.JPEG, {"quality": 95}),
        ("PNG", MediaType.PNG, {}),
        ("GIF", MediaType.GIF, {}),
        ("WEBP", MediaType.WEBP, {"lossless": True}),
        ("BMP", MediaType.BMP, {}),
        ("TIFF", MediaType.TIFF, {}),
    ],
)
def test_formats_decode_to_rgb(fmt, media_type, options):
    data = _encode(_sample(), fmt, **options)
    allowed = {media_type}
    loaded = load_image(data, allowed=allowed)
    assert loaded.image.mode == "RGB"
    assert loaded.image.size == (64, 48)
    assert loaded.info.media_type is media_type
    assert loaded.info.frame_count == 1
    assert loaded.frame_index == 0


def test_avif_when_pillow_supports_it():
    if not features.check("avif"):
        pytest.skip("this Pillow build has no AVIF support")
    try:
        data = _encode(_sample(), "AVIF", quality=90)
    except (OSError, KeyError, ValueError):
        pytest.skip("this Pillow build cannot encode AVIF")
    loaded = load_image(data)
    assert loaded.info.media_type is MediaType.AVIF
    assert loaded.image.size == (64, 48)


def test_lossless_formats_round_trip_pixels():
    image = _sample()
    for fmt in ("PNG", "BMP"):
        data = _encode(image, fmt)
        loaded = load_image(data, allowed={MediaType.PNG, MediaType.BMP})
        assert loaded.image.tobytes() == image.tobytes()


def test_tiff_and_bmp_are_off_by_default():
    with pytest.raises(UnsupportedMediaError, match="not accepted"):
        load_image(_encode(_sample(), "TIFF"))
    with pytest.raises(UnsupportedMediaError, match="not accepted"):
        load_image(_encode(_sample(), "BMP"))


# ---------------------------------------------------------------- modes


def _translucent(image):
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.linear_gradient("L").resize(image.size))
    return rgba


def test_mode_mix_is_normalized():
    rgb = _sample()
    cases = {
        "CMYK JPEG": (_encode(rgb.convert("CMYK"), "JPEG", quality=95), "RGB", False),
        "L JPEG": (_encode(rgb.convert("L"), "JPEG"), "RGB", False),
        "1-bit PNG": (_encode(rgb.convert("1"), "PNG"), "RGB", False),
        "LA PNG": (_encode(rgb.convert("LA"), "PNG"), "RGBA", True),
        "RGBA WebP": (_encode(_translucent(rgb), "WEBP", lossless=True), "RGBA", True),
        "palette PNG": (_encode(rgb.quantize(16), "PNG"), "RGB", False),
    }
    transparent = rgb.quantize(16)
    transparent.info["transparency"] = 3
    cases["palette PNG + transparency"] = (_encode(transparent, "PNG"), "RGBA", True)
    for name, (data, expected_mode, alpha) in cases.items():
        loaded = load_image(data)
        assert loaded.image.mode == expected_mode, name
        assert loaded.info.has_alpha is alpha, name
        assert loaded.image.size == (64, 48), name


def test_sixteen_bit_png_is_scaled_not_clipped():
    values = np.array([[0, 257, 32768, 65535]], dtype=np.uint16)
    data = _encode(Image.fromarray(values), "PNG")
    loaded = load_image(data)
    assert loaded.info.mode in ("I;16", "I")
    assert loaded.image.mode == "RGB"
    assert np.asarray(loaded.image)[0, :, 0].tolist() == [0, 1, 128, 255]


def test_requested_modes():
    data = _encode(_sample().convert("RGBA"), "PNG")
    assert load_image(data, mode="rgb").image.mode == "RGB"
    assert load_image(data, mode=ImageMode.RGBA).image.mode == "RGBA"
    opaque = load_image(_encode(_sample(), "PNG"), mode="rgba").image
    assert opaque.mode == "RGBA"
    assert np.asarray(opaque.getchannel("A")).min() == 255
    with pytest.raises(ValueError):
        load_image(data, mode="cmyk")


def test_result_is_independent_and_without_exif():
    exif = Image.Exif()
    exif[0x0112] = 1
    data = _encode(_sample(), "JPEG", exif=exif)
    loaded = load_image(data)
    assert "exif" not in loaded.image.info
    loaded.image.putpixel((0, 0), (1, 2, 3))  # writable, detached from the decoder
    assert dataclasses.is_dataclass(loaded)


# ---------------------------------------------------------------- orientation and ICC


def _marked_image():
    image = Image.new("RGB", (40, 20), (0, 0, 0))
    image.paste((255, 0, 0), (0, 0, 10, 5))  # red block at the top-left
    return image


@pytest.mark.parametrize(
    ("orientation", "size", "red_corner"),
    [
        (1, (40, 20), (0, 0)),
        (3, (40, 20), (39, 19)),
        (6, (20, 40), (19, 0)),
        (8, (20, 40), (0, 39)),
    ],
)
def test_exif_orientation_is_applied(orientation, size, red_corner):
    exif = Image.Exif()
    exif[0x0112] = orientation
    data = _encode(_marked_image(), "PNG", exif=exif)
    loaded = load_image(data)
    assert loaded.info.orientation == orientation
    assert loaded.image.size == size
    assert loaded.info.oriented_size == size
    red, green, blue = loaded.image.getpixel(red_corner)
    assert red > 200
    assert max(green, blue) < 50


def test_rotated_size_is_checked_against_the_limits():
    exif = Image.Exif()
    exif[0x0112] = 6
    data = _encode(Image.new("RGB", (100, 10)), "PNG", exif=exif)
    limits = dataclasses.replace(MediaLimits(), max_height=50)
    with pytest.raises(MediaLimitExceededError) as info:
        load_image(data, limits=limits)
    assert info.value.limit == "max_height"


def test_invalid_orientation_values_are_ignored():
    exif = Image.Exif()
    exif[0x0112] = 99
    loaded = load_image(_encode(_marked_image(), "PNG", exif=exif))
    assert loaded.info.orientation == 1
    assert loaded.image.size == (40, 20)


def test_icc_profile_is_kept_and_optionally_applied():
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    data = _encode(_sample(), "PNG", icc_profile=profile)
    loaded = load_image(data)
    assert loaded.info.icc_profile == profile
    assert not loaded.srgb_converted
    converted = load_image(data, to_srgb=True)
    assert converted.srgb_converted
    difference = np.abs(np.asarray(converted.image, np.int16) - np.asarray(loaded.image, np.int16))
    assert difference.max() <= 2


def test_corrupt_icc_profile_falls_back_without_colour_management():
    data = _encode(_sample(), "PNG", icc_profile=b"not a profile")
    loaded = load_image(data, to_srgb=True)
    assert not loaded.srgb_converted
    assert loaded.image.mode == "RGB"


# ---------------------------------------------------------------- animation


def _animated(fmt, colors, **options):
    frames = [Image.new("RGB", (16, 12), color) for color in colors]
    return _encode(
        frames[0],
        fmt,
        save_all=True,
        append_images=frames[1:],
        duration=[40, 80, 120],
        loop=0,
        **options,
    )


@pytest.mark.parametrize(
    ("fmt", "options"), [("GIF", {}), ("PNG", {}), ("WEBP", {"lossless": True})]
)
def test_iter_frames_yields_each_frame_as_displayed(fmt, options):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    data = _animated(fmt, colors, **options)
    first = load_image(data)
    assert first.info.frame_count == 3
    frames = list(iter_frames(data))
    assert [f.frame_index for f in frames] == [0, 1, 2]
    for frame, color in zip(frames, colors, strict=True):
        assert frame.image.size == (16, 12)
        assert frame.image.convert("RGB").getpixel((8, 6)) == color
    assert [f.duration_ms for f in frames] == [40.0, 80.0, 120.0]
    assert first.image.convert("RGB").getpixel((8, 6)) == colors[0]


def test_iter_frames_of_a_still_image():
    frames = list(iter_frames(_encode(_sample(), "PNG")))
    assert len(frames) == 1
    assert frames[0].duration_ms is None


def test_huge_gif_frame_count_is_refused_before_decoding():
    default = MediaLimits().max_image_frames
    assert default == 5_000
    data = _gif_with_frames(default + 1)
    with pytest.raises(MediaLimitExceededError) as info:
        load_image(data)
    assert info.value.limit == "max_image_frames"
    assert info.value.actual == default + 1
    with pytest.raises(MediaLimitExceededError):
        iter_frames(data)
    assert load_image(_gif_with_frames(12), max_frames=12).info.frame_count == 12
    with pytest.raises(MediaLimitExceededError) as info:
        load_image(_gif_with_frames(12), max_frames=11)
    assert info.value.limit == "max_frames"
    limits = dataclasses.replace(MediaLimits(), max_image_frames=11)
    with pytest.raises(MediaLimitExceededError) as info:
        probe_image(_gif_with_frames(12), limits=limits)
    assert info.value.limit == "max_image_frames"
    assert probe_image(_gif_with_frames(12), limits=limits, max_frames=12).frame_count == 12


def test_declared_apng_frame_count_is_refused_cheaply():
    data = bytearray(_animated("PNG", [(1, 1, 1), (2, 2, 2), (3, 3, 3)]))
    index = data.index(b"acTL")
    body = struct.pack(">II", 10_000_000, 0)
    data[index + 4 : index + 12] = body
    data[index + 12 : index + 16] = struct.pack(">I", zlib.crc32(b"acTL" + body))
    with pytest.raises(MediaLimitExceededError) as info:
        load_image(bytes(data))
    assert info.value.limit == "max_image_frames"


def test_video_frame_limit_also_applies():
    limits = dataclasses.replace(MediaLimits(), max_frames=2)
    with pytest.raises(MediaLimitExceededError):
        load_image(_animated("GIF", [(1, 1, 1), (2, 2, 2), (3, 3, 3)]), limits=limits)


# ---------------------------------------------------------------- hostile inputs


@pytest.mark.parametrize(
    ("width", "height", "limit"),
    [(100_000, 100, "max_width"), (100, 100_000, "max_height"), (10_000, 10_000, "max_pixels")],
)
def test_decompression_bomb_header_is_refused_before_decoding(width, height, limit):
    with pytest.raises(MediaLimitExceededError) as info:
        load_image(_png_header_only(width, height))
    assert info.value.limit == limit
    with pytest.raises(MediaLimitExceededError):
        probe_image(_png_header_only(width, height))


def test_gif_logical_screen_bomb_is_refused():
    with pytest.raises(MediaLimitExceededError):
        load_image(_gif_with_frames(1, width=65_535, height=65_535))


def test_pillow_bomb_threshold_is_not_involved():
    # Above Pillow's own warning threshold (89.5 MP): our limit decides, no
    # DecompressionBombWarning (which the test configuration turns into an error).
    limits = MediaLimits(max_pixels=200_000_000, max_width=20_000, max_height=20_000)
    info = probe_image(_png_header_only(10_000, 10_000), limits=limits)
    assert (info.width, info.height) == (10_000, 10_000)


def test_byte_limit():
    data = _encode(_sample(), "PNG")
    with pytest.raises(MediaLimitExceededError) as info:
        load_image(data, limits=dataclasses.replace(MediaLimits(), max_bytes=len(data) - 1))
    assert info.value.limit == "max_bytes"


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"hello, world" * 10,
        b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 64,  # MP4
        b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64,  # WAV
    ],
)
def test_wrong_magic_is_unsupported(data):
    with pytest.raises(UnsupportedMediaError):
        load_image(data)


def test_type_comes_from_bytes_not_from_the_allowed_list():
    with pytest.raises(UnsupportedMediaError, match="image/jpeg"):
        load_image(_encode(_sample(), "JPEG"), allowed={MediaType.PNG})


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "GIF", "WEBP"])
def test_truncated_images_raise_media_decode_error(fmt):
    data = _encode(_sample(256, 256), fmt)
    with pytest.raises(MediaDecodeError):
        load_image(data[: len(data) // 2])
    with pytest.raises(MediaError):
        load_image(data[:24])


def test_probe_does_not_decode_pixels():
    data = _encode(_sample(256, 256), "PNG")
    truncated = data[: len(data) // 2]
    info = probe_image(truncated)
    assert (info.width, info.height, info.media_type) == (256, 256, MediaType.PNG)
    with pytest.raises(MediaDecodeError):
        load_image(truncated)


def test_empty_png_dimensions_are_refused():
    with pytest.raises(MediaError):
        load_image(_png_header_only(0, 10))


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "GIF", "WEBP"])
def test_mutated_inputs_only_raise_media_errors(fmt):
    rng = random.Random(SEED)
    original = bytearray(_encode(_sample(48, 32), fmt))
    for _ in range(120):
        data = bytearray(original)
        for _ in range(rng.randint(1, 8)):
            position = rng.randrange(len(data))
            data[position] = rng.randrange(256)
        if rng.random() < 0.3:
            del data[rng.randrange(len(data)) :]
        try:
            loaded = load_image(bytes(data))
        except MediaError:
            continue
        assert loaded.image.mode in ("RGB", "RGBA")


# ---------------------------------------------------------------- openers (HEIC and others)

_FAKE_HEIC = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 32


def test_heic_needs_an_opener():
    with pytest.raises(UnsupportedMediaError, match="openers"):
        load_image(_FAKE_HEIC, allowed={MediaType.HEIC})
    with pytest.raises(UnsupportedMediaError, match="not accepted"):
        load_image(_FAKE_HEIC)


def test_opener_is_used_and_limits_still_apply():
    png = _encode(_sample(), "PNG")
    seen = []

    def opener(stream):
        seen.append(stream.read())
        return PngImagePlugin.PngImageFile(io.BytesIO(png))

    loaded = load_image(_FAKE_HEIC, openers={MediaType.HEIC: opener})
    assert seen == [_FAKE_HEIC]
    assert loaded.info.media_type is MediaType.HEIC
    assert loaded.image.size == (64, 48)
    small = dataclasses.replace(MediaLimits(), max_width=32)
    with pytest.raises(MediaLimitExceededError):
        load_image(_FAKE_HEIC, openers={MediaType.HEIC: opener}, limits=small)


def test_failing_or_wrong_openers():
    def broken(stream):
        raise RuntimeError("decoder crashed")

    with pytest.raises(MediaDecodeError):
        load_image(_FAKE_HEIC, openers={MediaType.HEIC: broken})
    with pytest.raises(TypeError):
        load_image(_FAKE_HEIC, openers={MediaType.HEIC: lambda stream: b"no"})
    with pytest.raises(TypeError):
        load_image(_FAKE_HEIC, openers={"image/heic": broken})  # type: ignore[dict-item]


# ---------------------------------------------------------------- argument validation


def test_argument_validation():
    data = _encode(_sample(), "PNG")
    with pytest.raises(TypeError):
        load_image("not bytes")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        load_image(data, limits={"max_bytes": 1})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        load_image(data, max_frames=0)
    with pytest.raises(TypeError):
        load_image(data, max_frames=1.5)  # type: ignore[arg-type]
    assert load_image(memoryview(data)).image.size == (64, 48)
    assert load_image(bytearray(data)).image.size == (64, 48)
