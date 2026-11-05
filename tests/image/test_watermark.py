"""Image watermarking with fake backends, and the TrustMark adapter's plumbing."""

from __future__ import annotations

import hashlib
import io
import math
import sysconfig

import numpy as np
import pytest
from PIL import Image

from provenance_kit import modelfiles
from provenance_kit.detection import WatermarkDetection
from provenance_kit.errors import (
    BackendUnavailableError,
    MediaLimitExceededError,
    ModelLoadError,
    PayloadError,
    UnsupportedMediaError,
)
from provenance_kit.image import _trustmark, watermark
from provenance_kit.image.watermark import (
    DEFAULT_STRENGTHS,
    MODEL_FILES,
    ImageWatermarker,
    ModelFile,
    TrustMarkBackend,
    TrustMarkEncoding,
    TrustMarkVariant,
    WatermarkBackendError,
    default_model_dir,
    download_trustmark_models,
    psnr,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.payload import PayloadCodec

BITS = 61


class RowBackend:
    """A fake model: the bits live in row 0, with an amplitude set by the strength."""

    payload_bits = BITS

    def __init__(self):
        self.embedded = []

    def embed(self, image, bits, *, strength):
        self.embedded.append((bits, strength))
        array = np.array(image.convert("RGB"), dtype=np.int16)
        amplitude = round(20 * strength)
        for i, bit in enumerate(bits):
            array[0, i] = 128 + (amplitude if bit == "1" else -amplitude)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

    def extract(self, image):
        row = np.asarray(image.convert("RGB"), dtype=np.int16)[0, :BITS, 0] - 128
        if np.any(np.abs(row) < 10):
            return WatermarkDetection.not_found()
        return WatermarkDetection(found=True, bits="".join("1" if v > 0 else "0" for v in row))


def lossy_png(pull):
    """An encoder that pulls row 0 towards grey by ``pull`` levels, then writes PNG."""

    def encode(image):
        array = np.asarray(image.convert("RGB"), dtype=np.int16).copy()
        row = array[0] - 128
        array[0] = 128 + np.sign(row) * np.maximum(0, np.abs(row) - pull)
        buffer = io.BytesIO()
        Image.fromarray(array.astype(np.uint8)).save(buffer, "PNG")
        return buffer.getvalue()

    return encode


def sample(width=96, height=64):
    rng = np.random.default_rng(7)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


# ---------------------------------------------------------------- ImageWatermarker


def test_default_codec_uses_the_backend_capacity():
    marker = ImageWatermarker(RowBackend())
    assert marker.payload_bits == BITS
    assert marker.codec == PayloadCodec(BITS, version_bits=4, check_bits=8)


def test_embed_and_detect_round_trip():
    marker = ImageWatermarker(RowBackend())
    marked = marker.embed(sample(), 123_456, version=2)
    detection = marker.detect(marked)
    assert detection.found
    assert detection.payload is not None
    assert detection.payload.valid
    assert (detection.payload.version, detection.payload.identifier) == (2, 123_456)


def test_detect_without_mark_has_no_payload():
    detection = ImageWatermarker(RowBackend()).detect(sample())
    assert not detection.found
    assert detection.payload is None


def test_codec_must_match_the_backend():
    with pytest.raises(ValueError, match="61"):
        ImageWatermarker(RowBackend(), codec=PayloadCodec(40))
    with pytest.raises(TypeError):
        ImageWatermarker(RowBackend(), codec="codec")  # type: ignore[arg-type]


def test_identifier_errors_come_from_the_codec():
    marker = ImageWatermarker(RowBackend())
    with pytest.raises(PayloadError):
        marker.embed(sample(), 1 << 49)
    with pytest.raises(ValueError):
        marker.embed(sample(), -1)


def test_embed_bits_validates_and_uses_default_strength():
    backend = RowBackend()
    marker = ImageWatermarker(backend, strength=1.5)
    marker.embed_bits(sample(), [1, 0] * 30 + [1])
    assert backend.embedded[-1] == ("10" * 30 + "1", 1.5)
    with pytest.raises(ValueError):
        marker.embed_bits(sample(), "101")
    with pytest.raises(ValueError):
        marker.embed_bits(sample(), "1" * BITS, strength=0)
    with pytest.raises(TypeError):
        ImageWatermarker(RowBackend(), strength="1")  # type: ignore[arg-type]


def test_limits_are_checked_before_the_model_runs():
    backend = RowBackend()
    marker = ImageWatermarker(backend, limits=MediaLimits(max_pixels=1000))
    with pytest.raises(MediaLimitExceededError):
        marker.embed(sample(), 1)
    with pytest.raises(MediaLimitExceededError):
        marker.detect(sample())
    assert backend.embedded == []


def test_wrong_length_bits_from_a_backend_are_an_error():
    class Broken(RowBackend):
        def extract(self, image):
            return WatermarkDetection(found=True, bits="101")

    with pytest.raises(WatermarkBackendError):
        ImageWatermarker(Broken()).detect(sample())


def test_embed_verified_escalates_until_the_mark_survives():
    backend = RowBackend()
    marker = ImageWatermarker(backend)
    # Pull 25: amplitude 20*s must exceed 35, so 1.0 and 1.5 fail and 2.0 works.
    result = marker.embed_verified(
        sample(), 42, encode=lossy_png(25), strengths=(1.0, 1.5, 2.0, 3.0)
    )
    assert result.verified
    assert result.strength == 2.0
    assert [a.strength for a in result.attempts] == [1.0, 1.5, 2.0]
    assert [a.recovered for a in result.attempts] == [False, False, True]
    assert all(math.isfinite(a.psnr) for a in result.attempts)
    assert result.attempts[-1].encoded_size == len(result.encoded)
    assert result.attempts[-1].detection.payload.identifier == 42
    # The published bytes carry the mark.
    assert marker.detect_encoded(result.encoded).payload.identifier == 42
    assert np.array_equal(
        np.asarray(Image.open(io.BytesIO(result.encoded)).convert("RGB"))[1:],
        np.asarray(result.image)[1:],
    )


def test_embed_verified_reports_failure_without_raising():
    marker = ImageWatermarker(RowBackend())
    result = marker.embed_verified(sample(), 42, encode=lossy_png(80), strengths=(0.5, 1.0))
    assert not result.verified
    assert result.strength is None
    assert [a.recovered for a in result.attempts] == [False, False]
    assert result.encoded  # the last attempt can still be published
    assert result.image.size == (96, 64)


def test_embed_verified_with_a_lossless_encoder_needs_one_attempt():
    result = ImageWatermarker(RowBackend()).embed_verified(sample(), 7, encode=lossy_png(0))
    assert result.verified
    assert result.strength == DEFAULT_STRENGTHS[0]
    assert len(result.attempts) == 1


def test_verified_embedding_repr_leaves_out_the_media():
    # Like the video and audio results: a log line must not carry the encoded
    # bytes (139 KB for a 256 x 256 JPEG) or the image.
    result = ImageWatermarker(RowBackend()).embed_verified(sample(256, 256), 7, encode=lossy_png(0))
    text = repr(result)
    assert len(text) < 2000
    assert "encoded=" not in text
    assert "image=" not in text
    assert "verified=True" in text


def test_embed_verified_argument_and_encoder_errors():
    marker = ImageWatermarker(RowBackend())
    with pytest.raises(TypeError):
        marker.embed_verified(sample(), 1, encode="webp")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bytes"):
        marker.embed_verified(sample(), 1, encode=lambda image: image)
    with pytest.raises(UnsupportedMediaError):
        marker.embed_verified(sample(), 1, encode=lambda image: b"not an image")
    with pytest.raises(ValueError):
        marker.embed_verified(sample(), 1, encode=lossy_png(0), strengths=())
    with pytest.raises(ValueError):
        marker.embed_verified(sample(), 1, encode=lossy_png(0), strengths=(1.0, -1.0))
    with pytest.raises(TypeError):
        marker.embed_verified(sample(), 1, encode=lossy_png(0), strengths="1")  # type: ignore[arg-type]

    def failing(image):
        raise RuntimeError("encoder bug")

    with pytest.raises(RuntimeError, match="encoder bug"):
        marker.embed_verified(sample(), 1, encode=failing)


def test_embed_verified_decodes_encoder_output_with_the_limits():
    marker = ImageWatermarker(RowBackend(), limits=MediaLimits(max_bytes=100))
    with pytest.raises(MediaLimitExceededError):
        marker.embed_verified(sample(), 1, encode=lossy_png(0))


def test_psnr():
    image = sample()
    assert psnr(image, image) == math.inf
    noisy = Image.fromarray(np.clip(np.asarray(image, np.int16) + 1, 0, 255).astype(np.uint8))
    assert 45 < psnr(image, noisy) < 50
    with pytest.raises(ValueError):
        psnr(image, image.resize((10, 10)))


# ---------------------------------------------------------------- TrustMark metadata


def test_encoding_capacities_match_trustmark_and_the_payload_table():
    capacities = {e: (e.schema, e.payload_bits, e.correctable_bits) for e in TrustMarkEncoding}
    assert capacities == {
        TrustMarkEncoding.BCH_SUPER: (0, 40, 8),
        TrustMarkEncoding.BCH_5: (1, 61, 5),
        TrustMarkEncoding.BCH_4: (2, 68, 4),
        TrustMarkEncoding.BCH_3: (3, 75, 3),
    }
    # payload + 7 parity bits per correctable error + 4 schema bits = 100 model bits.
    for encoding in TrustMarkEncoding:
        assert encoding.payload_bits + 7 * encoding.correctable_bits + 4 == 100


def test_model_files_are_pinned_for_every_variant():
    assert set(MODEL_FILES) == set(TrustMarkVariant)
    for variant, files in MODEL_FILES.items():
        names = [f.name for f in files]
        assert names == [
            f"trustmark_{variant.value}.yaml",
            f"encoder_{variant.value}.ckpt",
            f"decoder_{variant.value}.ckpt",
        ]
        for spec in files:
            assert len(spec.sha256) == 64
            assert int(spec.sha256, 16) >= 0
            assert spec.size > 0
    assert watermark.MODEL_BASE_URL.startswith("https://")


def test_default_model_dir_honours_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(watermark.MODEL_DIR_ENV, str(tmp_path))
    assert default_model_dir() == tmp_path / "trustmark"
    monkeypatch.delenv(watermark.MODEL_DIR_ENV)
    default = default_model_dir()
    assert default.parts[-3:] == ("provenance-kit", "models", "trustmark")


def test_model_dir_inside_site_packages_is_refused(tmp_path):
    inside = sysconfig.get_paths()["purelib"] + "/trustmark/models"
    with pytest.raises(ValueError, match="site-packages"):
        download_trustmark_models(model_dir=inside)
    with pytest.raises(ValueError, match="site-packages"):
        TrustMarkBackend(model_dir=inside)


def test_model_files_come_from_the_cdn():
    for files in MODEL_FILES.values():
        for spec in files:
            assert isinstance(spec, ModelFile)
            assert spec.url == watermark.MODEL_BASE_URL + spec.name


class FakeCdn:
    """Stands in for the model CDN: serves bytes by URL and records requests."""

    def __init__(self, content: dict[str, bytes]) -> None:
        self.content = content
        self.fetched: list[str] = []

    def open(self, request, timeout):
        name = request.full_url.rsplit("/", 1)[1]
        self.fetched.append(name)
        return io.BytesIO(self.content[name])


@pytest.fixture
def fake_files(monkeypatch):
    content = {"cfg.yaml": b"config", "enc.ckpt": b"encoder", "dec.ckpt": b"decoder"}
    specs = tuple(
        ModelFile(
            name=name,
            url=watermark.MODEL_BASE_URL + name,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        )
        for name, data in content.items()
    )
    monkeypatch.setattr(_trustmark, "MODEL_FILES", {TrustMarkVariant.Q: specs})
    cdn = FakeCdn(content)
    monkeypatch.setattr(modelfiles.urllib.request, "build_opener", lambda *handlers: cdn)
    return content, cdn


def test_download_writes_verified_files_once(tmp_path, fake_files):
    content, cdn = fake_files
    directory = download_trustmark_models(model_dir=tmp_path / "models")
    assert directory == (tmp_path / "models").resolve()
    assert sorted(p.name for p in directory.iterdir()) == sorted(content)
    assert (directory / "enc.ckpt").read_bytes() == b"encoder"
    download_trustmark_models(model_dir=tmp_path / "models")
    assert sorted(cdn.fetched) == sorted(content)  # nothing fetched the second time


def test_download_refuses_bad_content(tmp_path, fake_files):
    _, cdn = fake_files
    cdn.content = {name: b"x" * len(data) for name, data in cdn.content.items()}
    with pytest.raises(ModelLoadError, match="failed its SHA-256 check"):
        download_trustmark_models(model_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_argument_validation(tmp_path):
    with pytest.raises(ValueError):
        download_trustmark_models("Z", model_dir=tmp_path)
    with pytest.raises(ValueError):
        download_trustmark_models(model_dir=tmp_path, timeout_seconds=0)
    with pytest.raises(TypeError):
        download_trustmark_models(model_dir=tmp_path, timeout_seconds="5")  # type: ignore[arg-type]


def test_print_is_silenced_in_trustmark_modules_only(monkeypatch):
    import builtins
    import sys
    import types

    original = builtins.print
    fake = {name: types.ModuleType(name) for name in ("trustmark", "trustmark.model")}
    unrelated = types.ModuleType("trustmarkish")
    for name, module in {**fake, "trustmarkish": unrelated}.items():
        monkeypatch.setitem(sys.modules, name, module)
    _trustmark._silence_trustmark()
    assert all(module.print is _trustmark._quiet_print for module in fake.values())
    assert not hasattr(unrelated, "print")
    assert builtins.print is original


def test_configuration_targets_are_restricted():
    _trustmark._check_targets({"target": "trustmark.model.Identity", "params": {"a": [1, {}]}})
    with pytest.raises(ModelLoadError):
        _trustmark._check_targets({"params": {"x": {"target": "os.system"}}})
    with pytest.raises(ModelLoadError):
        _trustmark._check_targets([{"target": "subprocess.Popen"}])


def test_missing_extra_is_reported(monkeypatch, tmp_path):
    def missing(module, *, extra, feature):
        raise BackendUnavailableError.missing_extra(feature, extra)

    monkeypatch.setattr(_trustmark, "import_optional", missing)
    with pytest.raises(BackendUnavailableError, match="image-watermark"):
        TrustMarkBackend(model_dir=tmp_path)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"variant": "Z"}, ValueError),
        ({"encoding": "bch_9"}, ValueError),
        ({"device": "tpu"}, ValueError),
        ({"device": 0}, TypeError),
    ],
)
def test_backend_argument_validation_happens_before_imports(kwargs, error, tmp_path):
    with pytest.raises(error):
        TrustMarkBackend(model_dir=tmp_path, **kwargs)


def test_strength_and_bit_checks():
    assert _trustmark._check_strength(2) == 2.0
    for bad in (0, -1, 11, float("nan")):
        with pytest.raises(ValueError):
            _trustmark._check_strength(bad)
    with pytest.raises(TypeError):
        _trustmark._check_strength(True)
    with pytest.raises(ValueError):
        _trustmark._check_bits("012", 3)
    with pytest.raises(TypeError):
        _trustmark._check_bits([0, 1], 2)
