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

from provenance_kit.image.hashing import (
    PDQ_BACKEND_MAX_DISTANCE,
    PDQ_MATCH_DISTANCE,
    PDQ_MIN_QUALITY,
    BinaryHash,
    PdqHash,
    PdqHasher,
    dhash,
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
    require_module("pdqhash", extra="image-hash")
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
    """The bound: PDQ_BACKEND_MAX_DISTANCE (10) for quality >= 50.

    Measured on 1,200 images of this kind: 958 of 963 identical, the others
    2, 2, 2, 7 and 8 bits apart; qualities always equal.
    """
    port = PdqHasher("numpy")
    distances = []
    for array in _cross_check_corpus():
        a, b = cpp_hasher.hash_rgb(array), port.hash_rgb(array)
        assert a.quality == b.quality
        if a.quality >= PDQ_MIN_QUALITY:
            distances.append(a.distance(b))
    assert len(distances) > 100
    assert max(distances) <= PDQ_BACKEND_MAX_DISTANCE
    assert np.mean(np.asarray(distances) == 0) > 0.95


def test_auto_prefers_pdqhash(cpp_hasher):
    assert PdqHasher().backend == "pdqhash"


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
        require_module("pdqhash", extra="image-hash")
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
