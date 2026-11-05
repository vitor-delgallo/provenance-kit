"""Perceptual hashes with the real backends: pdqhash, imagehash, published vectors.

* The two PDQ backends are cross-checked on a generated corpus.
* ThreatExchange's published PDQ test vectors are checked against both
  backends. The images are downloaded once (``pytest --run-network``) at a
  pinned commit into the git-ignored ``.cache/threatexchange-pdq`` directory
  and verified by SHA-256; later runs use the cache offline.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request

import numpy as np
import pytest
from PIL import Image

from provenance_kit.image import hashing
from provenance_kit.image.hashing import (
    PDQ_BACKEND_MAX_DISTANCE,
    PDQ_DEFAULT_MAX_SIDE,
    PDQ_MATCH_DISTANCE,
    PDQ_MIN_QUALITY,
    BinaryHash,
    HashAlgorithm,
    PdqHash,
    PdqHasher,
    UndecidedReason,
    compare_fingerprints,
    dhash,
    fingerprint_image,
    phash,
)
from provenance_kit.image.loading import load_image
from tests.image.pdq_corpus import corpus, make_image
from tests.image.pdq_vectors import (
    REFERENCE_VECTORS,
    THREATEXCHANGE_COMMIT,
    THREATEXCHANGE_DATA_URL,
    THREATEXCHANGE_VECTORS,
)
from tests.support import REPO_ROOT, SEED, require_module

pytestmark = pytest.mark.integration

CACHE = REPO_ROOT / ".cache" / "threatexchange-pdq" / THREATEXCHANGE_COMMIT


@pytest.fixture(scope="module")
def cpp_hasher():
    require_module("pdqhash", extra="pdq-native")
    return PdqHasher("pdqhash")


def _cross_check_corpus(count=180):
    rng = np.random.default_rng(SEED)
    kinds = ("noise", "blocks", "smooth", "shapes", "jpeg")
    for i in range(count):
        kind = kinds[i % len(kinds)]
        width, height = int(rng.integers(16, 700)), int(rng.integers(16, 700))
        if kind == "jpeg":
            buffer = io.BytesIO()
            Image.fromarray(make_image(width, height, "smooth", 2000 + i)).save(
                buffer, "JPEG", quality=int(rng.integers(30, 95))
            )
            yield np.asarray(Image.open(buffer).convert("RGB"))
        else:
            yield make_image(width, height, kind, 3000 + i)


def test_pdqhash_matches_the_reference_vectors(cpp_hasher):
    # pdqhash (float32 C++) on the same pixels as the float64 reference (no pre-scaling).
    full_resolution = PdqHasher("pdqhash", max_side=None)
    for name, array in corpus():
        expected_hex, expected_quality = REFERENCE_VECTORS[name]
        result = full_resolution.hash_rgb(array)
        assert result.distance(PdqHash.from_hex(expected_hex)) <= PDQ_BACKEND_MAX_DISTANCE
        assert result.quality == expected_quality


def test_backends_agree_on_a_generated_corpus(cpp_hasher):
    """The bound: PDQ_BACKEND_MAX_DISTANCE (10) for quality >= 50, both sides >= 64 px.

    Measured on 3,000 images (2026-11-05, pdqhash 0.2.8): of the 2,307
    reliable pairs whose shorter side was at least 64 px as hashed, 2,302
    identical and the others within 2 bits. Thinner images (5 to 63 px) are
    outside the bound: up to 21 bits on 426 pairs, and 132 bits on one
    854 x 6 image. Qualities differed by 1 on 3 of the 3,000 images; on this
    deterministic corpus they are equal.
    """
    port = PdqHasher("numpy")
    distances = []
    for array in _cross_check_corpus():
        a, b = cpp_hasher.hash_rgb(array), port.hash_rgb(array)
        assert abs(a.quality - b.quality) <= 1
        if a.is_reliable() and b.is_reliable() and min(array.shape[:2]) >= 64:
            distances.append(a.distance(b))
    assert len(distances) > 100
    assert max(distances) <= PDQ_BACKEND_MAX_DISTANCE
    assert np.mean(np.asarray(distances) == 0) > 0.95


def test_auto_prefers_pdqhash(cpp_hasher):
    assert PdqHasher().backend == "pdqhash"


@pytest.mark.parametrize("view", ["rot90", "transpose", "fortran"])
def test_pdqhash_hashes_strided_views_like_their_copies(cpp_hasher, view):
    array = make_image(400, 300, "smooth", 31)
    strided = {
        "rot90": np.rot90(array),
        "transpose": array.transpose(1, 0, 2),
        "fortran": np.asfortranarray(array),
    }[view]
    assert not strided.flags["C_CONTIGUOUS"]
    expected = cpp_hasher.hash_rgb(np.ascontiguousarray(strided))
    assert cpp_hasher.hash_rgb(strided) == expected
    assert PdqHasher("numpy").hash_rgb(strided).distance(expected) <= PDQ_BACKEND_MAX_DISTANCE


@pytest.mark.parametrize("size", [(300, 4), (4, 300), (3, 3)])
def test_both_backends_return_the_zero_hash_under_five_pixels(cpp_hasher, size):
    array = make_image(*size, "noise", 32)
    expected = PdqHash(bytes(32), 0)
    assert cpp_hasher.hash_rgb(array) == expected
    assert PdqHasher("numpy").hash_rgb(array) == expected


@pytest.mark.parametrize(
    "size", [(1000, 100), (90, 4000), (2500, 70), (1726, 70), (4000, 128), (16000, 200)]
)
def test_backends_agree_on_elongated_images(cpp_hasher, size):
    # With Meta's 512 px thumbnail these became 512 x 5..63 (or less), where
    # the backends drift apart (up to 21 bits measured, 132 on one image); the
    # pre-scaling now keeps their shorter side at 64 px, inside the bound.
    image = Image.fromarray(make_image(*size, "smooth", 33))
    box = hashing._thumbnail_box(*size, PDQ_DEFAULT_MAX_SIDE)
    scaled = image.copy()
    if box is not None:
        scaled.thumbnail((box, box))
    assert min(scaled.size) >= 64
    port = PdqHasher("numpy").hash_image(image)
    native = cpp_hasher.hash_image(image)
    assert port.is_reliable()
    assert native.is_reliable()
    assert port.distance(native) <= PDQ_BACKEND_MAX_DISTANCE


def test_a_strip_that_meta_shrinks_under_five_pixels_is_now_hashed_by_both(cpp_hasher):
    # 3000 x 20: Meta's 512 px thumbnail gives 512 x 3, where pdqhash returns
    # the zero hash (quality 0) and the port used to return quality 100.
    image = Image.fromarray(make_image(3000, 20, "smooth", 34))
    port = PdqHasher("numpy").hash_image(image)
    native = cpp_hasher.hash_image(image)
    assert native.value != bytes(32)
    assert abs(port.quality - native.quality) <= 1


# ---------------------------------------------------------------- published vectors


@pytest.fixture(scope="module")
def threatexchange_images(request):
    images = {}
    missing = []
    for path, sha256, size, _ in THREATEXCHANGE_VECTORS:
        target = CACHE / path
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == sha256:
            images[path] = target.read_bytes()
        else:
            missing.append((path, sha256, size))
    if missing and not request.config.getoption("--run-network"):
        pytest.skip(
            "ThreatExchange PDQ test images are not cached: run pytest --run-network once "
            "to download them into .cache/threatexchange-pdq"
        )
    for path, sha256, size in missing:
        url = THREATEXCHANGE_DATA_URL + path
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - pinned https URL
            data = response.read(size + 1)
        assert len(data) == size, path
        assert hashlib.sha256(data).hexdigest() == sha256, path
        target = CACHE / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        images[path] = data
    return images


@pytest.mark.parametrize("backend", ["numpy", "pdqhash"])
def test_threatexchange_published_vectors(threatexchange_images, backend):
    if backend == "pdqhash":
        require_module("pdqhash", extra="pdq-native")
    hasher = PdqHasher(backend)
    for path, _, _, expected_hex in THREATEXCHANGE_VECTORS:
        loaded = load_image(threatexchange_images[path])
        result = hasher.hash_image(loaded.image)
        distance = result.distance(PdqHash.from_hex(expected_hex))
        if result.quality >= PDQ_MIN_QUALITY:
            # Measured: 0 for every image with quality >= 50 (pre-scaled to 512).
            assert distance == 0, path
        else:
            # ThreatExchange's own test tolerance.
            assert distance <= 16, path


def test_full_resolution_stays_within_the_match_threshold(threatexchange_images):
    hasher = PdqHasher("numpy", max_side=None)
    for path, _, _, expected_hex in THREATEXCHANGE_VECTORS:
        result = hasher.hash_image(load_image(threatexchange_images[path]).image)
        if result.quality >= PDQ_MIN_QUALITY:
            # Measured: at most 12 bits.
            assert result.distance(PdqHash.from_hex(expected_hex)) <= 16, path


def test_dihedral_copies_are_distinct_hashes(threatexchange_images):
    # PDQ is not rotation invariant: the eight bridge variants hash far apart.
    hasher = PdqHasher("numpy")
    bridges = [p for p, *_ in THREATEXCHANGE_VECTORS if "/dih/" in p]
    hashes = [hasher.hash_image(load_image(threatexchange_images[p]).image) for p in bridges]
    original = hashes[0]
    assert all(original.distance(h) > PDQ_MATCH_DISTANCE for h in hashes[1:])


# ---------------------------------------------------------------- imagehash


def test_phash_and_dhash_match_imagehash():
    imagehash = require_module("imagehash", extra="image-hash")
    image = Image.fromarray(make_image(320, 240, "smooth", 21))
    assert phash(image).hex() == str(imagehash.phash(image.convert("L")))
    assert dhash(image).hex() == str(imagehash.dhash(image.convert("L")))
    assert phash(image, hash_size=16).bits == 256
    assert BinaryHash.from_hex(str(imagehash.phash(image.convert("L")))) == phash(image)


def test_phash_and_dhash_survive_re_encoding():
    require_module("imagehash", extra="image-hash")
    original = Image.fromarray(make_image(640, 480, "smooth", 22))
    buffer = io.BytesIO()
    original.resize((320, 240), Image.Resampling.LANCZOS).save(buffer, "JPEG", quality=70)
    copy = load_image(buffer.getvalue()).image
    other = Image.fromarray(make_image(640, 480, "smooth", 23))
    assert phash(original).distance(phash(copy)) <= 10
    assert dhash(original).distance(dhash(copy)) <= 10
    assert phash(original).distance(phash(other)) > 10


def _ramp(width, height, *, vertical):
    yy, xx = np.mgrid[0:height, 0:width]
    level = (255 - yy * 255 // height) if vertical else (xx * 255 // width)
    return Image.fromarray(np.dstack([level, level, level]).astype(np.uint8))


FEATURELESS_PAIRS = {
    "white-black": (Image.new("RGB", (400, 300), "white"), Image.new("RGB", (400, 300), "black")),
    "red-blue": (
        Image.new("RGB", (400, 300), (200, 30, 30)),
        Image.new("RGB", (400, 300), (30, 30, 200)),
    ),
    "orthogonal-ramps": (
        _ramp(400, 300, vertical=False),
        _ramp(400, 300, vertical=True),
    ),
}


@pytest.mark.parametrize("name", sorted(FEATURELESS_PAIRS))
@pytest.mark.parametrize("algorithms", [("pdq", "phash", "dhash"), ("phash", "dhash")])
def test_different_featureless_images_never_match(name, algorithms):
    require_module("imagehash", extra="image-hash")
    a, b = FEATURELESS_PAIRS[name]
    hasher = PdqHasher("numpy")
    result = compare_fingerprints(
        fingerprint_image(a, algorithms=algorithms, pdq_hasher=hasher),
        fingerprint_image(b, algorithms=algorithms, pdq_hasher=hasher),
    )
    assert not result.matched
    assert result.decided_by is None
    expected = (
        UndecidedReason.LOW_QUALITY if "pdq" in algorithms else UndecidedReason.LOW_INFORMATION
    )
    assert result.undecided_reason is expected


def test_phash_still_matches_a_re_encoded_copy_without_pdq():
    require_module("imagehash", extra="image-hash")
    original = Image.fromarray(make_image(640, 480, "smooth", 22))
    buffer = io.BytesIO()
    original.resize((320, 240), Image.Resampling.LANCZOS).save(buffer, "JPEG", quality=70)
    copy = load_image(buffer.getvalue()).image
    without_pdq = ("phash", "dhash")
    result = compare_fingerprints(
        fingerprint_image(original, algorithms=without_pdq),
        fingerprint_image(copy, algorithms=without_pdq),
    )
    assert result.decided_by is HashAlgorithm.PHASH
    assert result.matched
