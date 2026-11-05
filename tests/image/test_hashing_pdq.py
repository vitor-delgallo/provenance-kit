"""PDQ: the NumPy port against reference vectors, PdqHash and PdqHasher."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from provenance_kit.errors import BackendUnavailableError
from provenance_kit.image import hashing
from provenance_kit.image._pdq_numpy import pdq_hash_rgb
from provenance_kit.image.hashing import PDQ_BITS, PdqHash, PdqHasher
from tests.image.pdq_corpus import CORPUS_SPECS, corpus, make_image
from tests.image.pdq_vectors import REFERENCE_VECTORS


@pytest.mark.parametrize(("name", "array"), corpus(), ids=[spec[0] for spec in CORPUS_SPECS])
def test_numpy_port_reproduces_the_reference_bit_for_bit(name, array):
    expected_hex, expected_quality = REFERENCE_VECTORS[name]
    value, quality = pdq_hash_rgb(array)
    assert value.hex() == expected_hex
    assert quality == expected_quality


def test_every_corpus_image_has_a_vector():
    assert {spec[0] for spec in CORPUS_SPECS} == set(REFERENCE_VECTORS)


def test_corpus_is_deterministic():
    first = make_image(50, 40, "smooth", 7)
    again = make_image(50, 40, "smooth", 7)
    assert first.dtype == np.uint8
    assert first.shape == (40, 50, 3)
    assert np.array_equal(first, again)


# ---------------------------------------------------------------- PdqHash

HEX = "9c151c3af838278e3ef57c180c7d031c07aefd12f2ccc1e18f2a1e1c7d0ff163"  # ThreatExchange sample


def test_pdq_hash_hex_round_trip_and_bit_order():
    h = PdqHash.from_hex(HEX, quality=80)
    assert h.value == bytes.fromhex(HEX)
    assert h.hex() == HEX
    assert h.quality == 80
    assert h.bits == PDQ_BITS
    assert h.to_int() == int(HEX, 16)
    # The reference Hash256 sample has 128 bits set.
    assert h.to_int().bit_count() == 128
    # Bit k of the reference hash is bit k % 8 of byte 31 - k // 8.
    k = 3
    assert (h.value[31 - k // 8] >> (k % 8)) & 1 == (h.to_int() >> k) & 1


def test_pdq_hash_from_hex_accepts_upper_case_and_defaults_quality():
    h = PdqHash.from_hex(HEX.upper())
    assert h.hex() == HEX
    assert h.quality == 100


def test_pdq_hash_distance():
    a = PdqHash.from_hex("0" * 64, quality=90)
    b = PdqHash.from_hex("f" * 64, quality=90)
    c = PdqHash.from_hex(HEX, quality=90)
    assert a.distance(b) == 256
    assert a.distance(c) == 128
    assert c.distance(c) == 0
    with pytest.raises(TypeError):
        a.distance(HEX)  # type: ignore[arg-type]


def test_pdq_hash_reliability_threshold():
    assert PdqHash.from_hex(HEX, quality=50).is_reliable()
    assert not PdqHash.from_hex(HEX, quality=49).is_reliable()
    assert PdqHash.from_hex(HEX, quality=49).is_reliable(min_quality=40)


@pytest.mark.parametrize(
    ("value", "quality", "error"),
    [
        (b"\x00" * 31, 50, ValueError),
        (b"\x00" * 33, 50, ValueError),
        ("00" * 32, 50, TypeError),
        (bytearray(32), 50, TypeError),
        (b"\x00" * 32, 101, ValueError),
        (b"\x00" * 32, -1, ValueError),
        (b"\x00" * 32, True, TypeError),
        (b"\x00" * 32, 50.0, TypeError),
    ],
)
def test_pdq_hash_validation(value, quality, error):
    with pytest.raises(error):
        PdqHash(value, quality)


@pytest.mark.parametrize("text", ["abc", HEX[:-1], HEX + "0", HEX[:-1] + "g", " " + HEX[1:]])
def test_pdq_hash_from_hex_rejects_malformed_text(text):
    with pytest.raises(ValueError):
        PdqHash.from_hex(text)


def test_pdq_hash_from_hex_rejects_non_strings():
    with pytest.raises(TypeError):
        PdqHash.from_hex(b"00" * 32)  # type: ignore[arg-type]


def test_pdq_hash_is_frozen_and_hashable():
    h = PdqHash.from_hex(HEX)
    assert {h: 1}[PdqHash.from_hex(HEX)] == 1
    with pytest.raises(AttributeError):
        h.quality = 3  # type: ignore[misc]


# ---------------------------------------------------------------- PdqHasher


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.shapes = []

    def hash_rgb(self, rgb):
        self.shapes.append(rgb.shape)
        assert rgb.dtype == np.uint8
        return bytes([rgb.shape[0] % 256]) + b"\x00" * 31, 77


def test_hasher_uses_a_backend_object_and_returns_pdq_hash():
    fake = FakeBackend()
    hasher = PdqHasher(fake)
    result = hasher.hash_image(Image.new("RGB", (40, 30), (1, 2, 3)))
    assert result == PdqHash(bytes([30]) + b"\x00" * 31, 77)
    assert hasher.backend == "fake"
    assert fake.shapes == [(30, 40, 3)]


def test_hasher_shrinks_large_images_to_max_side_keeping_aspect():
    fake = FakeBackend()
    PdqHasher(fake).hash_image(Image.new("RGB", (2000, 1000)))
    PdqHasher(fake, max_side=256).hash_image(Image.new("RGB", (300, 900)))
    PdqHasher(fake, max_side=None).hash_image(Image.new("RGB", (1500, 700)))
    PdqHasher(fake).hash_image(Image.new("RGB", (512, 100)))
    assert fake.shapes == [(256, 512, 3), (256, 85, 3), (700, 1500, 3), (100, 512, 3)]


@pytest.mark.parametrize(
    ("size", "shape"),
    [
        # Meta's rule would give 512 x 51: the short side is kept at 64 instead.
        ((1000, 100), (64, 640, 3)),
        ((100, 1000), (640, 64, 3)),
        ((1000, 123), (64, 521, 3)),
        # Already under 64 px: never shrunk further (Meta's rule: 512 x 3).
        ((3000, 20), (20, 3000, 3)),
        # The longer side is capped at 32 * max_side (16,384 px).
        ((20000, 100), (64, 12800, 3)),
        ((40000, 100), (41, 16384, 3)),
    ],
)
def test_pre_scaling_never_pushes_the_short_side_below_64_pixels(size, shape):
    fake = FakeBackend()
    PdqHasher(fake).hash_image(Image.new("RGB", size))
    assert fake.shapes == [shape]


def test_pre_scaling_bounds_follow_max_side():
    fake = FakeBackend()
    PdqHasher(fake, max_side=128).hash_image(Image.new("RGB", (4000, 100)))
    PdqHasher(fake, max_side=128).hash_image(Image.new("RGB", (8000, 100)))  # 32 * 128 cap
    PdqHasher(fake, max_side=None).hash_image(Image.new("RGB", (8000, 100)))
    assert fake.shapes == [(64, 2560, 3), (51, 4096, 3), (100, 8000, 3)]


class ContiguityBackend:
    name = "contiguity"

    def __init__(self):
        self.contiguous = []

    def hash_rgb(self, rgb):
        self.contiguous.append(bool(rgb.flags["C_CONTIGUOUS"]))
        return pdq_hash_rgb(np.ascontiguousarray(rgb))


def _strided_views(array):
    return {
        "rot90": np.rot90(array),
        "transpose": array.transpose(1, 0, 2),
        "fortran": np.asfortranarray(array),
        "mirrored": array[:, ::-1],
        "rgba-slice": np.dstack([array, array[..., :1]])[..., :3],
    }


@pytest.mark.parametrize("view", ["rot90", "transpose", "fortran", "mirrored", "rgba-slice"])
def test_backends_receive_c_contiguous_pixels(view):
    # pdqhash reads the buffer as row-major whatever the strides: a rotated or
    # Fortran-ordered view gave a hash about 126 bits away from the right one.
    array = _strided_views(make_image(200, 120, "shapes", 8))[view]
    assert not array.flags["C_CONTIGUOUS"]
    backend = ContiguityBackend()
    result = PdqHasher(backend).hash_rgb(array)
    assert backend.contiguous == [True]
    assert result == PdqHasher("numpy").hash_rgb(np.ascontiguousarray(array))


@pytest.mark.parametrize("view", ["rot90", "transpose", "fortran"])
def test_numpy_backend_hashes_strided_views_like_their_copies(view):
    array = _strided_views(make_image(300, 200, "smooth", 9))[view]
    hasher = PdqHasher("numpy")
    assert hasher.hash_rgb(array) == hasher.hash_rgb(np.ascontiguousarray(array))


@pytest.mark.parametrize("shape", [(4, 300, 3), (300, 4, 3), (1, 1, 3), (4, 4, 3)])
def test_sides_under_five_pixels_give_the_zero_hash_like_meta(shape):
    # Meta's C++ hasher (MIN_HASHABLE_DIM = 5) returns the all-zero hash with
    # quality 0 for such images; the port must not call them reliable.
    rng = np.random.default_rng(10)
    array = rng.integers(0, 256, shape, dtype=np.uint8)
    assert pdq_hash_rgb(array) == (bytes(32), 0)
    result = PdqHasher("numpy").hash_rgb(array)
    assert result == PdqHash(bytes(32), 0)
    assert not result.is_reliable()


def test_five_pixel_sides_are_hashed():
    value, quality = pdq_hash_rgb(make_image(300, 5, "noise", 11))
    assert value != bytes(32)
    assert quality > 0


def test_hash_rgb_matches_hash_image_including_shrinking():
    hasher = PdqHasher("numpy")
    for array in (make_image(300, 200, "smooth", 1), make_image(900, 700, "shapes", 2)):
        assert hasher.hash_rgb(array) == hasher.hash_image(Image.fromarray(array))


def test_hasher_does_not_modify_the_input_image():
    image = Image.fromarray(make_image(800, 600, "smooth", 3))
    before = image.tobytes()
    PdqHasher("numpy").hash_image(image)
    assert image.size == (800, 600)
    assert image.tobytes() == before


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK", "LA", "1", "I;16", "I", "F"])
def test_hasher_accepts_every_common_mode(mode):
    base = Image.fromarray(make_image(96, 80, "shapes", 4))
    if mode in ("I;16", "I"):
        image = Image.fromarray(np.asarray(base.convert("L"), dtype=np.uint16) * 257)
        if mode == "I":
            image = image.convert("I")
    elif mode == "F":
        image = base.convert("L").convert("F")
    else:
        image = base.convert(mode)
    result = PdqHasher("numpy").hash_image(image)
    assert len(result.value) == 32
    if mode in ("L", "LA", "I;16", "I", "F"):
        # Greyscale inputs, 16-bit ones scaled down, hash like their 8-bit L form.
        assert result == PdqHasher("numpy").hash_image(base.convert("L"))


def test_alpha_is_ignored_like_meta_tools():
    rgb = Image.fromarray(make_image(120, 90, "smooth", 5))
    rgba = rgb.copy()
    rgba.putalpha(Image.new("L", rgb.size, 0))
    hasher = PdqHasher("numpy")
    assert hasher.hash_image(rgba) == hasher.hash_image(rgb)


def test_hasher_rejects_bad_inputs():
    hasher = PdqHasher("numpy")
    with pytest.raises(TypeError):
        hasher.hash_image(np.zeros((8, 8, 3), np.uint8))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        hasher.hash_image(Image.new("RGB", (0, 0)))
    with pytest.raises(TypeError):
        hasher.hash_rgb(np.zeros((8, 8, 3), np.float32))
    with pytest.raises(TypeError):
        hasher.hash_rgb([[0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        hasher.hash_rgb(np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError):
        hasher.hash_rgb(np.zeros((0, 8, 3), np.uint8))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"backend": "cpp"}, ValueError),
        ({"backend": 3}, TypeError),
        ({"max_side": 63}, ValueError),
        ({"max_side": 512.0}, TypeError),
        ({"max_side": True}, TypeError),
    ],
)
def test_hasher_rejects_bad_configuration(kwargs, error):
    with pytest.raises(error):
        PdqHasher(**kwargs)


def test_backend_names_are_case_insensitive():
    assert PdqHasher(" NumPy ").backend == "numpy"


def test_auto_falls_back_to_numpy_when_pdqhash_is_missing(monkeypatch):
    def missing(module, *, extra, feature):
        raise BackendUnavailableError.missing_extra(feature, extra)

    monkeypatch.setattr(hashing, "import_optional", missing)
    assert PdqHasher().backend == "numpy"
    with pytest.raises(BackendUnavailableError, match="pdq-native") as info:
        PdqHasher("pdqhash")
    assert info.value.extra == "pdq-native"


def test_low_information_images_get_low_quality():
    flat = np.full((100, 100, 3), 128, np.uint8)
    assert PdqHasher("numpy").hash_rgb(flat).quality == 0
    assert not PdqHasher("numpy").hash_rgb(flat).is_reliable()


def test_pdq_survives_mild_edits():
    array = make_image(400, 300, "smooth", 6)
    hasher = PdqHasher("numpy")
    original = hasher.hash_rgb(array)
    brighter = hasher.hash_rgb(np.clip(array.astype(np.int16) + 12, 0, 255).astype(np.uint8))
    smaller = hasher.hash_image(Image.fromarray(array).resize((200, 150), Image.Resampling.LANCZOS))
    unrelated = hasher.hash_rgb(make_image(400, 300, "smooth", 99))
    assert original.distance(brighter) <= hashing.PDQ_MATCH_DISTANCE
    assert original.distance(smaller) <= hashing.PDQ_MATCH_DISTANCE
    assert original.distance(unrelated) > hashing.PDQ_MATCH_DISTANCE
