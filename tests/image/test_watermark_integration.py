"""TrustMark with the real model: round trips, encoders, quietness, model checks.

The model files live in the git-ignored ``.cache/models/trustmark``
directory. ``pytest --run-network`` downloads the Q variant there once (65 MB,
verified by SHA-256); later runs are offline. Tests for the P and C variants
run when their files are present (``download_trustmark_models("P", ...)``).
"""

from __future__ import annotations

import io
import shutil

import numpy as np
import pytest
from PIL import Image

from provenance_kit.errors import ModelLoadError
from provenance_kit.image import _trustmark
from provenance_kit.image.watermark import (
    DEFAULT_STRENGTHS,
    MODEL_FILES,
    ImageWatermarker,
    TrustMarkBackend,
    TrustMarkDetection,
    TrustMarkEncoding,
    TrustMarkVariant,
    download_trustmark_models,
    psnr,
)
from tests.image.pdq_corpus import make_image
from tests.support import REPO_ROOT, require_module

pytestmark = pytest.mark.integration

MODEL_DIR = REPO_ROOT / ".cache" / "models" / "trustmark"


def _has_files(variant):
    return all(_trustmark._is_valid(MODEL_DIR, spec) for spec in MODEL_FILES[variant])


@pytest.fixture(scope="module")
def model_dir(request):
    require_module("torch", extra="image-watermark")
    require_module("trustmark", extra="image-watermark")
    if not _has_files(TrustMarkVariant.Q):
        if not request.config.getoption("--run-network"):
            pytest.skip(
                "TrustMark Q model files are not in .cache/models/trustmark: run "
                "pytest --run-network once to download them"
            )
        download_trustmark_models("Q", model_dir=MODEL_DIR)
    return MODEL_DIR


@pytest.fixture(scope="module")
def backend(model_dir):
    return TrustMarkBackend(model_dir=model_dir)


@pytest.fixture(scope="module")
def marker(backend):
    return ImageWatermarker(backend)


def _photo_like(width=800, height=600, seed=1):
    return Image.fromarray(make_image(width, height, "smooth", seed))


def _encoder(fmt, quality):
    def encode(image):
        buffer = io.BytesIO()
        image.save(buffer, fmt, quality=quality)
        return buffer.getvalue()

    return encode


def test_capacities_agree_with_the_trustmark_source(model_dir):
    datalayer = require_module("trustmark.datalayer", extra="image-watermark")
    for encoding in TrustMarkEncoding:
        layer = datalayer.DataLayer(100, verbose=False, encoding_mode=encoding.schema)
        assert layer.schemaCapacity(encoding.schema) == encoding.payload_bits
        assert layer.bch_encoder.get_ecc_bits() == 7 * encoding.correctable_bits
        assert layer.bch_encoder.ECCstate.t == encoding.correctable_bits


def test_round_trip_in_memory(backend, marker):
    image = _photo_like()
    marked = marker.embed(image, 987_654_321, version=3)
    assert marked.size == image.size
    assert marked.mode == "RGB"
    assert psnr(image, marked) > 38
    detection = marker.detect(marked)
    assert isinstance(detection, TrustMarkDetection)
    assert detection.found
    assert detection.encoding is TrustMarkEncoding.BCH_5
    assert detection.corrected_bits == 0
    assert detection.payload is not None
    assert detection.payload.valid
    assert (detection.payload.version, detection.payload.identifier) == (3, 987_654_321)
    assert backend.payload_bits == 61
    assert backend.device == "cpu"


def test_unmarked_images_give_no_valid_payload(marker):
    false_payloads = 0
    for seed in range(12):
        kind = ("smooth", "shapes", "noise", "blocks")[seed % 4]
        detection = marker.detect(Image.fromarray(make_image(320 + seed, 240, kind, 100 + seed)))
        if detection.found and detection.payload is not None and detection.payload.valid:
            false_payloads += 1
    assert false_payloads == 0


@pytest.mark.parametrize(
    ("fmt", "quality"), [("WEBP", 80), ("WEBP", 60), ("JPEG", 85), ("JPEG", 70)]
)
def test_verified_embedding_through_real_encoders(marker, fmt, quality):
    image = _photo_like(seed=5)
    result = marker.embed_verified(image, 31_337, encode=_encoder(fmt, quality))
    assert result.verified, [(a.strength, a.recovered) for a in result.attempts]
    assert result.strength in DEFAULT_STRENGTHS
    assert result.attempts[-1].psnr > 30
    assert marker.detect_encoded(result.encoded).payload.identifier == 31_337


def test_tiny_images_report_failure_instead_of_raising(marker):
    result = marker.embed_verified(_photo_like(34, 42), 5, encode=_encoder("WEBP", 80))
    assert not result.verified
    assert result.strength is None
    assert len(result.attempts) == len(DEFAULT_STRENGTHS)


def test_alpha_and_wide_images(marker):
    rgba = _photo_like(640, 480).convert("RGBA")
    alpha = Image.linear_gradient("L").resize(rgba.size)
    rgba.putalpha(alpha)
    marked = marker.embed(rgba, 11)
    assert marked.mode == "RGBA"
    assert np.array_equal(np.asarray(marked.getchannel("A")), np.asarray(alpha))
    assert marker.detect(marked).payload.identifier == 11
    wide = _photo_like(1500, 400, seed=9)  # aspect 3.75: only the centre square is marked
    marked_wide = marker.embed(wide, 12)
    assert marked_wide.size == wide.size
    assert np.array_equal(np.asarray(marked_wide)[:, :100], np.asarray(wide)[:, :100])
    assert marker.detect(marked_wide).payload.identifier == 12


def test_construction_and_use_print_nothing(model_dir, capfd):
    capfd.readouterr()
    quiet = TrustMarkBackend(model_dir=model_dir, encoding="bch_super", device="auto")
    marked = quiet.embed(_photo_like(320, 240), "1" * 40, strength=1.0)
    detection = quiet.extract(marked)
    assert detection.found
    assert detection.bits == "1" * 40
    assert capfd.readouterr() == ("", "")


def test_other_encodings_reject_marks_of_this_one(model_dir, marker):
    marked = marker.embed(_photo_like(), 77)
    other = TrustMarkBackend(model_dir=model_dir, encoding="bch_3")
    assert not other.extract(marked).found


def test_tampered_model_file_is_refused(model_dir, tmp_path):
    for spec in MODEL_FILES[TrustMarkVariant.Q]:
        shutil.copyfile(model_dir / spec.name, tmp_path / spec.name)
    target = tmp_path / "decoder_Q.ckpt"
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    target.write_bytes(bytes(data))
    with pytest.raises(ModelLoadError, match="SHA-256"):
        TrustMarkBackend(model_dir=tmp_path)
    (tmp_path / "encoder_Q.ckpt").unlink()
    with pytest.raises(ModelLoadError, match="encoder_Q"):
        TrustMarkBackend(model_dir=tmp_path)


@pytest.mark.parametrize("variant", [TrustMarkVariant.P, TrustMarkVariant.C])
def test_other_variants_when_provisioned(model_dir, variant):
    if not _has_files(variant):
        pytest.skip(f"TrustMark {variant.value} model files are not in .cache/models/trustmark")
    marker = ImageWatermarker(TrustMarkBackend(model_dir=model_dir, variant=variant))
    # A textured image: variant P cannot hold a mark on very smooth content
    # (it fails even before encoding on the smooth corpus images).
    image = Image.fromarray(make_image(800, 600, "shapes", 3))
    marked = marker.embed(image, 4242)
    assert marker.detect(marked).payload.identifier == 4242
    assert psnr(image, marked) > (45 if variant is TrustMarkVariant.P else 40)


@pytest.mark.network
def test_download_is_idempotent(model_dir):
    assert download_trustmark_models("Q", model_dir=model_dir) == model_dir.resolve()
