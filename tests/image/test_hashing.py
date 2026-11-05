"""Binary hashes, fingerprints, thresholds and known-hash lists (unit tests)."""

from __future__ import annotations

import types

import numpy as np
import pytest
from PIL import Image

from provenance_kit.image import hashing
from provenance_kit.image.hashing import (
    BinaryHash,
    FingerprintComparison,
    HashAlgorithm,
    HashList,
    ImageFingerprint,
    MatchThresholds,
    PdqHash,
    UndecidedReason,
    compare_fingerprints,
    fingerprint_image,
)
from provenance_kit.matching import HammingMatch, hamming_matches
from tests.support import assert_imports_lightly


def test_image_modules_import_lightly():
    assert_imports_lightly(
        [
            "provenance_kit.image.hashing",
            "provenance_kit.image.loading",
            "provenance_kit.image.visible",
            "provenance_kit.image.watermark",
        ]
    )


def test_thresholds_follow_meta_guidance():
    assert hashing.PDQ_MATCH_DISTANCE == 31
    assert hashing.PDQ_MIN_QUALITY == 50
    defaults = MatchThresholds()
    assert (defaults.pdq_max_distance, defaults.pdq_min_quality) == (31, 50)
    assert (defaults.phash_max_distance, defaults.dhash_max_distance) == (10, 10)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"pdq_max_distance": -1}, ValueError),
        ({"pdq_min_quality": 101}, ValueError),
        ({"phash_max_distance": 1.5}, TypeError),
        ({"dhash_max_distance": True}, TypeError),
        ({"min_bit_balance": -0.1}, ValueError),
        ({"min_bit_balance": 0.6}, ValueError),
        ({"min_bit_balance": float("nan")}, ValueError),
        ({"min_bit_balance": "0.1"}, TypeError),
        ({"min_bit_balance": True}, TypeError),
    ],
)
def test_thresholds_validation(kwargs, error):
    with pytest.raises(error):
        MatchThresholds(**kwargs)


# ---------------------------------------------------------------- BinaryHash


def test_binary_hash_hex_round_trip():
    h = BinaryHash(0x0F, 64)
    assert h.hex() == "000000000000000f"
    assert BinaryHash.from_hex("000000000000000F") == h
    assert BinaryHash.from_hex("0f", bits=8) == BinaryHash(15, 8)
    assert h.to_bytes() == b"\x00" * 7 + b"\x0f"
    assert BinaryHash(1, 12).to_bytes() == b"\x00\x01"


def test_binary_hash_distance():
    assert BinaryHash(0b1011, 64).distance(BinaryHash(0b0001, 64)) == 2
    with pytest.raises(ValueError):
        BinaryHash(1, 64).distance(BinaryHash(1, 32))
    with pytest.raises(TypeError):
        BinaryHash(1, 64).distance(1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "bits", "error"),
    [(-1, 64, ValueError), (1 << 64, 64, ValueError), (64, 6, ValueError), (1, 0, ValueError)],
)
def test_binary_hash_validation(value, bits, error):
    with pytest.raises(error):
        BinaryHash(value, bits)


@pytest.mark.parametrize(("text", "bits"), [("xyz0", None), ("0f", 16), ("", None)])
def test_binary_hash_from_hex_rejects_malformed(text, bits):
    with pytest.raises(ValueError):
        BinaryHash.from_hex(text, bits=bits)


# ---------------------------------------------------------------- pHash / dHash with a fake


class _FakeHash:
    def __init__(self, bits):
        self.hash = np.asarray(bits, dtype=bool)


@pytest.fixture
def fake_imagehash(monkeypatch):
    calls = []

    def phash(image, hash_size):
        calls.append(("phash", image.mode, hash_size))
        bits = np.zeros((hash_size, hash_size), bool)
        bits[0, 0] = True  # most significant bit
        return _FakeHash(bits)

    def dhash(image, hash_size):
        calls.append(("dhash", image.mode, hash_size))
        bits = np.zeros((hash_size, hash_size), bool)
        bits[-1, -1] = True  # least significant bit
        return _FakeHash(bits)

    module = types.SimpleNamespace(phash=phash, dhash=dhash)
    monkeypatch.setattr(hashing, "_imagehash", lambda: module)
    return calls


def test_phash_and_dhash_bit_order_and_luma_input(fake_imagehash):
    image = Image.new("RGBA", (40, 30), (10, 20, 30, 0))
    assert hashing.phash(image) == BinaryHash(1 << 63, 64)
    assert hashing.dhash(image) == BinaryHash(1, 64)
    assert hashing.phash(image, hash_size=3) == BinaryHash(1 << 8, 9)
    assert fake_imagehash[:2] == [("phash", "L", 8), ("dhash", "L", 8)]


def test_odd_widths_pad_the_hex_form_like_imagehash(fake_imagehash):
    # 3 x 3 = 9 bits: imagehash prints ceil(9 / 4) = 3 hex digits.
    h = hashing.phash(Image.new("L", (8, 8)), hash_size=3)
    assert h.hex() == "100"
    assert BinaryHash.from_hex("100", bits=9) == h
    with pytest.raises(ValueError):
        BinaryHash.from_hex("200", bits=9)


@pytest.mark.parametrize("hash_size", [1, 65, 8.0])
def test_hash_size_validation(fake_imagehash, hash_size):
    with pytest.raises((TypeError, ValueError)):
        hashing.phash(Image.new("L", (8, 8)), hash_size=hash_size)


def test_fingerprint_image_selects_algorithms(fake_imagehash):
    image = Image.new("RGB", (64, 64), (100, 50, 0))
    full = fingerprint_image(image, pdq_hasher=hashing.PdqHasher("numpy"))
    assert full.pdq is not None
    assert full.phash == BinaryHash(1 << 63, 64)
    assert full.dhash == BinaryHash(1, 64)
    only_pdq = fingerprint_image(image, algorithms=["pdq"], pdq_hasher=hashing.PdqHasher("numpy"))
    assert only_pdq.phash is None
    assert only_pdq.dhash is None
    with pytest.raises(ValueError):
        fingerprint_image(image, algorithms=["md5"])


def test_fingerprint_type_validation():
    with pytest.raises(TypeError):
        ImageFingerprint(pdq="00")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ImageFingerprint(phash=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------- comparison

GOOD_A = PdqHash.from_hex("0" * 64, quality=90)
GOOD_B = PdqHash(bytes([0xFF, 0xFF]) + b"\x00" * 30, 90)  # 16 bits away
POOR_B = PdqHash(bytes([0xFF, 0xFF]) + b"\x00" * 30, 20)


def test_pdq_decides_when_reliable():
    result = compare_fingerprints(
        ImageFingerprint(pdq=GOOD_A, phash=BinaryHash(0, 64)),
        ImageFingerprint(pdq=GOOD_B, phash=BinaryHash((1 << 40) - 1, 64)),
    )
    assert result == FingerprintComparison(
        pdq_distance=16,
        pdq_reliable=True,
        phash_distance=40,
        dhash_distance=None,
        decided_by=HashAlgorithm.PDQ,
        matched=True,
    )
    strict = compare_fingerprints(
        ImageFingerprint(pdq=GOOD_A),
        ImageFingerprint(pdq=GOOD_B),
        thresholds=MatchThresholds(pdq_max_distance=15),
    )
    assert not strict.matched


#: 64-bit hashes with 32 bits set, as pHash gives for any image without ties.
BALANCED = 0xF0F0F0F0F0F0F0F0
BALANCED_NEAR = BALANCED ^ 0b111  # 3 bits away, still balanced


def test_unreliable_pdq_decides_nothing_even_when_phash_and_dhash_agree():
    # A white and a black canvas: PDQ says "featureless" (quality 0), and the
    # pHash / dHash of both are all zeros. Falling back to them would call
    # any two flat images the same picture.
    a = ImageFingerprint(pdq=GOOD_A, phash=BinaryHash(BALANCED, 64), dhash=BinaryHash(0, 64))
    b = ImageFingerprint(pdq=POOR_B, phash=BinaryHash(BALANCED, 64), dhash=BinaryHash(0, 64))
    result = compare_fingerprints(a, b)
    assert result.pdq_reliable is False
    assert result.phash_distance == 0
    assert result.dhash_distance == 0
    assert result.decided_by is None
    assert result.undecided_reason is UndecidedReason.LOW_QUALITY
    assert not result.matched


def test_phash_then_dhash_decide_only_without_pdq():
    a = ImageFingerprint(phash=BinaryHash(BALANCED, 64), dhash=BinaryHash(BALANCED, 64))
    b = ImageFingerprint(pdq=GOOD_B, phash=BinaryHash(BALANCED_NEAR, 64))
    result = compare_fingerprints(a, b)
    assert result.pdq_distance is None
    assert result.decided_by is HashAlgorithm.PHASH
    assert result.undecided_reason is None
    assert result.matched
    only_dhash = compare_fingerprints(
        ImageFingerprint(dhash=BinaryHash(BALANCED, 64)),
        ImageFingerprint(dhash=BinaryHash(BALANCED ^ 0xFFF, 64)),
    )
    assert only_dhash.decided_by is HashAlgorithm.DHASH
    assert not only_dhash.matched  # 12 bits apart


@pytest.mark.parametrize(
    ("value_a", "value_b"),
    [
        (0, 1),  # flat images: pHash / dHash all zeros (one tie-breaking bit)
        (0, 0),
        ((1 << 64) - 1, (1 << 64) - 1),  # dHash of a left-to-right ramp: all ones
        (0b111, 0),  # 3 of 64 bits set: below the 8-bit minimum
    ],
)
def test_low_information_phash_and_dhash_do_not_decide(value_a, value_b):
    a = ImageFingerprint(phash=BinaryHash(value_a, 64), dhash=BinaryHash(value_a, 64))
    b = ImageFingerprint(phash=BinaryHash(value_b, 64), dhash=BinaryHash(value_b, 64))
    result = compare_fingerprints(a, b)
    assert result.phash_distance is not None
    assert result.phash_distance <= 10
    assert result.decided_by is None
    assert result.undecided_reason is UndecidedReason.LOW_INFORMATION
    assert not result.matched


def test_an_informative_dhash_decides_when_phash_is_low_information():
    a = ImageFingerprint(phash=BinaryHash(0, 64), dhash=BinaryHash(BALANCED, 64))
    b = ImageFingerprint(phash=BinaryHash(1, 64), dhash=BinaryHash(BALANCED_NEAR, 64))
    result = compare_fingerprints(a, b)
    assert result.decided_by is HashAlgorithm.DHASH
    assert result.matched


def test_the_information_gate_is_configurable():
    a = ImageFingerprint(phash=BinaryHash(0b111, 64))
    b = ImageFingerprint(phash=BinaryHash(0b110, 64))
    assert compare_fingerprints(a, b).decided_by is None
    lenient = compare_fingerprints(a, b, thresholds=MatchThresholds(min_bit_balance=0.0))
    assert lenient.decided_by is HashAlgorithm.PHASH
    assert lenient.matched
    # The gate is a fraction of the width: 9-bit hashes need 2 minority bits (ceil 1.125).
    small_a = ImageFingerprint(phash=BinaryHash(0b000000011, 9))
    small_b = ImageFingerprint(phash=BinaryHash(0b000000111, 9))
    assert compare_fingerprints(small_a, small_b).decided_by is HashAlgorithm.PHASH


def test_nothing_to_compare():
    result = compare_fingerprints(ImageFingerprint(pdq=GOOD_A), ImageFingerprint())
    assert result.decided_by is None
    assert result.undecided_reason is UndecidedReason.NO_COMMON_HASH
    assert not result.matched
    with pytest.raises(TypeError):
        compare_fingerprints(GOOD_A, ImageFingerprint())  # type: ignore[arg-type]


# ---------------------------------------------------------------- HashList


def _random_pdq_bytes(rng, count):
    return rng.integers(0, 256, size=(count, 32), dtype=np.uint8)


def test_hash_list_matches_the_core_linear_scan(rng):
    packed = _random_pdq_bytes(rng, 3000)
    query = bytearray(packed[1234].tobytes())
    query[0] ^= 0b1010_0001  # 3 bits away from entry 1234
    keys = [f"id-{i}" for i in range(3000)]
    known = HashList.from_packed(packed, keys=keys)
    expected = hamming_matches(
        int.from_bytes(query, "big"),
        ((k, int.from_bytes(row.tobytes(), "big")) for k, row in zip(keys, packed, strict=True)),
        bits=256,
        max_distance=110,
    )
    assert known.match(bytes(query), max_distance=110) == expected
    assert known.match(bytes(query))[0] == HammingMatch(key="id-1234", distance=3, bits=256)


def test_hash_list_ordering_ties_and_limit():
    zero = "0" * 64
    one_bit = "0" * 63 + "1"
    entries = [("far", "f" * 64), ("b", one_bit), ("a", zero), ("c", one_bit), ("d", zero)]
    known = HashList(entries)
    matches = known.match(PdqHash.from_hex(zero))
    assert [(m.key, m.distance) for m in matches] == [("a", 0), ("d", 0), ("b", 1), ("c", 1)]
    assert [m.key for m in known.match(zero, limit=3)] == ["a", "d", "b"]
    assert [m.key for m in known.nearest(zero, count=1)] == ["a"]
    assert known.nearest("f" * 64)[0].key == "far"
    assert known.match(zero, max_distance=0, limit=1)[0].similarity == 1.0


def test_hash_list_accepts_every_hash_representation():
    value = PdqHash.from_hex("ab" * 32)
    entries = [
        (1, value),
        (2, value.value),
        (3, value.to_int()),
        (4, value.hex()),
        (5, BinaryHash(value.to_int(), 256)),
    ]
    known = HashList(entries)
    assert len(known) == 5
    assert [m.distance for m in known.match(value)] == [0] * 5
    assert list(known) == [1, 2, 3, 4, 5]


def test_hash_list_of_64_bit_hashes_requires_an_explicit_threshold():
    known = HashList([("x", BinaryHash(0b1111, 64)), ("y", BinaryHash(0, 64))], bits=64)
    assert known.nbytes == 16
    assert [m.key for m in known.match(BinaryHash(0b0111, 64), max_distance=1)] == ["x"]
    with pytest.raises(ValueError, match="required"):
        known.match(BinaryHash(0, 64))
    with pytest.raises(ValueError):
        known.match(PdqHash.from_hex("0" * 64), max_distance=3)


def test_hash_list_with_odd_widths_pads_internally():
    known = HashList([("x", (1 << 71) | 1), ("y", 2)], bits=72)
    assert known.nbytes == 32  # two 64-bit words per entry
    assert [(m.key, m.distance) for m in known.match(1, max_distance=1)] == [("x", 1)]
    with pytest.raises(ValueError):
        HashList([("z", 1 << 72)], bits=72)
    with pytest.raises(ValueError):
        HashList([("z", b"\x80" + b"\x00" * 8)], bits=71)


def test_hash_list_from_hex_and_errors():
    lines = ["0" * 64 + "\n", " " + "f" * 64]
    known = HashList.from_hex(lines, keys=range(2))
    assert [m.key for m in known.match("f" * 64)] == [1]
    with pytest.raises(ValueError, match="position 1"):
        HashList.from_hex(["0" * 64, "zz"], keys=range(2))
    with pytest.raises(ValueError, match="2 hashes for 3 keys"):
        HashList.from_hex(["0" * 64, "1" * 64], keys=range(3))
    with pytest.raises(ValueError):
        HashList.from_packed(b"\x00" * 33, keys=[1])


def test_from_packed_accepts_one_row_per_hash_or_a_flat_array(rng):
    packed = _random_pdq_bytes(rng, 64)
    query = bytes(packed[5].tobytes())
    for data in (packed, packed.ravel(), packed.tobytes()):
        known = HashList.from_packed(data, keys=range(64))
        assert [(m.key, m.distance) for m in known.match(query, max_distance=0)] == [(5, 0)]


@pytest.mark.parametrize(
    "shape",
    [(32, 64), (64, 32, 1), (2048, 1), (1, 2048)],  # right total size, wrong layout
)
def test_from_packed_rejects_mis_shaped_arrays(rng, shape):
    packed = np.ascontiguousarray(_random_pdq_bytes(rng, 64).reshape(shape))
    with pytest.raises(ValueError, match="shape"):
        HashList.from_packed(packed, keys=range(64))


def test_from_packed_rejects_other_dtypes(rng):
    packed = _random_pdq_bytes(rng, 4).astype(np.int64)
    with pytest.raises(TypeError, match="uint8"):
        HashList.from_packed(packed, keys=range(4))


def test_empty_hash_list():
    known: HashList[str] = HashList([])
    assert len(known) == 0
    assert known.match("0" * 64) == []


@pytest.mark.parametrize(
    ("query", "kwargs", "error"),
    [
        (1.5, {}, TypeError),
        (True, {}, TypeError),
        (-1, {}, ValueError),
        ("0" * 63, {}, ValueError),
        (b"\x00" * 31, {}, ValueError),
        ("0" * 64, {"max_distance": 257}, ValueError),
        ("0" * 64, {"max_distance": "3"}, TypeError),
        ("0" * 64, {"limit": 0}, ValueError),
        ("0" * 64, {"limit": 1.0}, TypeError),
    ],
)
def test_hash_list_query_validation(query, kwargs, error):
    known = HashList([("a", "0" * 64)])
    with pytest.raises(error):
        known.match(query, **kwargs)


def test_hash_list_is_read_only():
    known = HashList([("a", "0" * 64)])
    with pytest.raises(ValueError):
        known._words[0, 0] = 1


def test_hash_list_scales_to_a_million_pdq_hashes(rng):
    count = 1_000_000
    packed = _random_pdq_bytes(rng, count)
    target = 987_654
    query = bytes(packed[target].tobytes())
    known = HashList.from_packed(packed, keys=range(count))
    assert known.nbytes == 32 * count
    matches = known.match(query)
    # Random 256-bit hashes are about 128 bits apart: only the target is within 31.
    assert [(m.key, m.distance) for m in matches] == [(target, 0)]


def test_popcount_fallback_matches_numpy_bitwise_count(monkeypatch, rng):
    words = rng.integers(0, 2**63, size=(50, 4), dtype=np.uint64)
    fast = hashing._popcount_rows(words)
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    slow = hashing._popcount_rows(words)
    assert np.array_equal(fast, slow)
