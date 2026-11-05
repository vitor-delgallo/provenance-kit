"""Unit tests of the video watermark API with a fake backend (no model needed)."""

from __future__ import annotations

import hashlib
import io
import types
import urllib.error

import numpy as np
import pytest

from provenance_kit import PayloadCodec
from provenance_kit.errors import BackendUnavailableError, ModelLoadError
from provenance_kit.video.frames import VideoFrame
from provenance_kit.video.watermark import (
    RepetitionCode,
    VideoSealBackend,
    VideoWatermarkDetection,
    _api,
    _videoseal,
    _weights,
    detect_watermark,
    embed_verified,
    embed_watermark,
)
from provenance_kit.video.watermark._weights import (
    ModelFile,
    download_model_file,
    resolve_model_file,
)

WIDTH = 256


class FakeBackend:
    """Writes the channel bits into the first pixel row; reads them back.

    ``flips`` channel positions are inverted on read, to simulate a lossy codec.
    """

    payload_bits = 256

    def __init__(self, flips: int = 0, seed: int = 0, within: int = 256) -> None:
        self.flips = np.random.default_rng(seed).choice(within, size=flips, replace=False)

    def embed_frames(self, frames, bits):
        for frame in frames:
            out = np.array(frame, copy=True)
            out[0, :WIDTH, 0] = [200 if c == "1" else 50 for c in bits]
            yield out

    def score_frames(self, frames):
        for frame in frames:
            scores = (frame[0, :WIDTH, 0].astype(np.float32) - 125.0) / 75.0
            scores[self.flips] *= -1
            yield scores


def blank_frames(count: int = 4) -> list[np.ndarray]:
    return [np.full((4, WIDTH, 3), 128, dtype=np.uint8) for _ in range(count)]


def random_bits(rng, n: int) -> str:
    return "".join(rng.choice(["0", "1"], size=n))


def test_embed_and_detect_raw_channel_bits(rng):
    backend = FakeBackend()
    bits = random_bits(rng, 256)
    marked = list(embed_watermark(blank_frames(), bits, backend=backend))
    detection = detect_watermark(marked, backend=backend, min_mean_abs_score=0.5)
    assert isinstance(detection, VideoWatermarkDetection)
    assert detection.found
    assert detection.bits == bits
    assert detection.frames == 4
    assert detection.p_value is None
    assert detection.confidence is None
    assert detection.mean_abs_score == pytest.approx(1.0)


def test_repetition_code_gives_a_calibrated_decision(rng):
    backend = FakeBackend(flips=40, within=64)  # one wrong copy for 40 of the 64 bits
    code = RepetitionCode(copies=4)
    codec = PayloadCodec(code.data_bits, version_bits=4, check_bits=12)
    bits = codec.encode(123_456_789, version=3)
    marked = list(embed_watermark(blank_frames(), bits, backend=backend, repetition=code))
    detection = detect_watermark(marked, backend=backend, codec=codec, repetition=code)
    assert detection.found
    assert detection.p_value is not None
    assert detection.p_value < 1e-10
    assert detection.confidence == pytest.approx(1.0 - detection.p_value)
    assert detection.bits == bits
    assert detection.payload is not None
    assert detection.payload.valid
    assert detection.payload.identifier == 123_456_789


def test_unmarked_frames_are_not_found(rng):
    backend = FakeBackend()
    code = RepetitionCode(copies=4)
    frames = [rng.integers(0, 256, (4, WIDTH, 3), dtype=np.uint8) for _ in range(3)]
    detection = detect_watermark(frames, backend=backend, repetition=code)
    assert not detection.found
    assert detection.bits is None
    assert detection.payload is None
    assert detection.decoded_bits is not None
    assert len(detection.decoded_bits) == 64
    assert detection.p_value is not None
    assert detection.p_value > 1e-4


def test_check_bits_decide_without_a_repetition_code(rng):
    backend = FakeBackend()
    codec = PayloadCodec(256, version_bits=4, check_bits=16)
    marked = list(embed_watermark(blank_frames(2), codec.encode(7), backend=backend))
    assert detect_watermark(marked, backend=backend, codec=codec).found
    noise = [rng.integers(0, 256, (4, WIDTH, 3), dtype=np.uint8)]
    assert not detect_watermark(noise, backend=backend, codec=codec).found


def test_no_frames_means_not_found():
    detection = detect_watermark([], backend=FakeBackend())
    assert not detection.found
    assert detection.frames == 0


def test_argument_validation(rng):
    backend = FakeBackend()
    code = RepetitionCode(copies=4)
    with pytest.raises(ValueError):
        list(embed_watermark(blank_frames(1), "01", backend=backend))
    with pytest.raises(ValueError):
        embed_watermark(blank_frames(1), "0" * 256, backend=backend, repetition=code)
    with pytest.raises(ValueError):
        embed_watermark(
            blank_frames(1), "0" * 128, backend=backend, repetition=RepetitionCode(total_bits=128)
        )
    with pytest.raises(ValueError):
        detect_watermark(blank_frames(1), backend=backend, codec=PayloadCodec(61))
    with pytest.raises(ValueError):
        detect_watermark(blank_frames(1), backend=backend, max_p_value=0.0)
    with pytest.raises(TypeError):
        detect_watermark(blank_frames(1), backend=backend, repetition=4)  # type: ignore[arg-type]


def test_malformed_backend_scores_are_refused():
    class Broken(FakeBackend):
        def score_frames(self, frames):
            for _ in frames:
                yield np.full(256, np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="malformed"):
        detect_watermark(blank_frames(1), backend=Broken())


def fake_codec_round_trip(monkeypatch, *, damage: int = 0):
    """Patches frame decoding so that the 'encoded' bytes are raw frames."""

    def encoder(frames):
        stacked = np.stack(list(frames))
        if damage:
            stacked[:, 0, :damage, 0] = 255 - stacked[:, 0, :damage, 0]
        return stacked.tobytes()

    def fake_iter_frames(data, **kwargs):
        array = np.frombuffer(data, dtype=np.uint8).reshape(-1, 4, WIDTH, 3)
        for index, image in enumerate(array):
            yield VideoFrame(index=index, timestamp=index / 25, image=image)

    monkeypatch.setattr(_api, "iter_frames", fake_iter_frames)
    return encoder


def test_embed_verified_round_trip(monkeypatch, rng):
    encoder = fake_codec_round_trip(monkeypatch)
    code = RepetitionCode(copies=4)
    codec = PayloadCodec(64, version_bits=4, check_bits=8)
    bits = codec.encode(99)
    result = embed_verified(
        iter(blank_frames(5)),
        bits,
        encoder=encoder,
        backend=FakeBackend(),
        repetition=code,
        codec=codec,
    )
    assert result.verified
    assert result.bit_errors == 0
    assert result.frames_embedded == 5
    assert result.detection.payload is not None
    assert result.detection.payload.identifier == 99
    assert isinstance(result.encoded, bytes)


def test_embed_verified_reports_failures(monkeypatch, rng):
    encoder = fake_codec_round_trip(monkeypatch, damage=256)  # inverts every channel bit
    bits = random_bits(rng, 256)
    result = embed_verified(blank_frames(2), bits, encoder=encoder, backend=FakeBackend())
    assert not result.verified
    assert result.bit_errors == 256


def test_embed_verified_requires_bytes(monkeypatch):
    with pytest.raises(TypeError):
        embed_verified(
            blank_frames(1), "0" * 256, encoder=lambda frames: "text", backend=FakeBackend()
        )
    with pytest.raises(TypeError):
        embed_verified(blank_frames(1), "0" * 256, encoder=None, backend=FakeBackend())  # type: ignore[arg-type]


# ------------------------------------------------------------ model files


SPEC_DATA = b"model weights" * 100
SPEC = ModelFile(
    name="model.bin",
    url="https://example.invalid/model.bin",
    sha256=hashlib.sha256(SPEC_DATA).hexdigest(),
    size=len(SPEC_DATA),
)


def resolve(**kwargs):
    options = {
        "model_path": None,
        "model_dir": None,
        "default_subdir": "test",
        "sha256": None,
        "download": False,
    }
    options.update(kwargs)
    return resolve_model_file(SPEC, **options)


def test_resolve_verifies_pre_provisioned_files(tmp_path):
    (tmp_path / "model.bin").write_bytes(SPEC_DATA)
    assert resolve(model_dir=tmp_path) == tmp_path / "model.bin"
    assert resolve(model_path=tmp_path / "model.bin") == tmp_path / "model.bin"


def test_resolve_refuses_tampered_files(tmp_path):
    (tmp_path / "model.bin").write_bytes(SPEC_DATA + b"!")
    with pytest.raises(ModelLoadError, match="SHA-256"):
        resolve(model_dir=tmp_path)


def test_resolve_never_downloads_unless_asked(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network used")

    monkeypatch.setattr(_weights.urllib.request, "build_opener", no_network)
    with pytest.raises(ModelLoadError, match="download=True"):
        resolve(model_dir=tmp_path)
    with pytest.raises(ModelLoadError):
        resolve(model_path=tmp_path / "missing.bin")


def test_resolve_validates_arguments(tmp_path):
    with pytest.raises(ValueError):
        resolve(model_dir=tmp_path, sha256="xyz")
    with pytest.raises(ValueError):
        resolve(model_path=tmp_path / "model.bin", download=True)


def test_default_model_dir_honors_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(_weights.MODEL_DIR_ENV, str(tmp_path))
    assert _weights.default_model_dir("videoseal") == tmp_path / "videoseal"
    monkeypatch.delenv(_weights.MODEL_DIR_ENV)
    assert _weights.default_model_dir("videoseal").parts[-2:] == ("provenance-kit", "videoseal")


class FakeOpener:
    def __init__(self, payload: bytes | Exception) -> None:
        self.payload = payload
        self.requests: list[str] = []

    def open(self, request, timeout):
        self.requests.append(request.full_url)
        if isinstance(self.payload, Exception):
            raise self.payload
        return io.BytesIO(self.payload)


def patch_opener(monkeypatch, payload) -> FakeOpener:
    opener = FakeOpener(payload)
    monkeypatch.setattr(_weights.urllib.request, "build_opener", lambda *handlers: opener)
    return opener


def test_download_writes_only_verified_files(tmp_path, monkeypatch):
    opener = patch_opener(monkeypatch, SPEC_DATA)
    path = resolve(model_dir=tmp_path / "new", download=True)
    assert path.read_bytes() == SPEC_DATA
    assert opener.requests == [SPEC.url]
    assert sorted(p.name for p in path.parent.iterdir()) == ["model.bin"]


@pytest.mark.parametrize(
    "payload",
    [
        SPEC_DATA + b"x",
        SPEC_DATA[:-1],
        b"y" * len(SPEC_DATA),
        urllib.error.URLError("down"),
        OSError(),
    ],
)
def test_failed_downloads_leave_nothing_behind(tmp_path, monkeypatch, payload):
    patch_opener(monkeypatch, payload)
    with pytest.raises(ModelLoadError):
        download_model_file(SPEC, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_redirects_must_stay_on_https():
    handler = _weights._HttpsOnlyRedirects()
    request = _weights.urllib.request.Request("https://example.invalid/a")
    with pytest.raises(urllib.error.URLError):
        handler.redirect_request(
            request, io.BytesIO(), 302, "Found", {}, "http://example.invalid/b"
        )


def test_model_file_validation():
    with pytest.raises(ValueError):
        ModelFile(name="a", url="http://example.invalid/a", sha256="0" * 64, size=1)
    with pytest.raises(ValueError):
        ModelFile(name="../a", url="https://example.invalid/a", sha256="0" * 64, size=1)
    with pytest.raises(ValueError):
        ModelFile(name="a", url="https://example.invalid/a", sha256="0" * 63, size=1)


# ------------------------------------------------------------ backend construction


def fake_torch() -> types.SimpleNamespace:
    return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))


def test_videoseal_backend_requires_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(_videoseal, "import_optional", lambda *a, **k: fake_torch())
    with pytest.raises(ModelLoadError, match="download=True"):
        VideoSealBackend(model_dir=tmp_path)


def test_videoseal_backend_reports_missing_torch(monkeypatch):
    def missing(*args, **kwargs):
        raise BackendUnavailableError.missing_extra("VideoSeal", "video-watermark")

    monkeypatch.setattr(_videoseal, "import_optional", missing)
    with pytest.raises(BackendUnavailableError) as info:
        VideoSealBackend()
    assert info.value.extra == "video-watermark"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strength": 0.0},
        {"strength": 99.0},
        {"step_size": 0},
        {"batch_frames": 0},
        {"step_size": 4, "batch_frames": 6},
        {"device": "tpu"},
    ],
)
def test_videoseal_backend_validates_arguments_before_loading(kwargs, monkeypatch):
    monkeypatch.setattr(_videoseal, "import_optional", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(ValueError):
        VideoSealBackend(**kwargs)


def test_videoseal_constants():
    assert (
        _videoseal.VIDEOSEAL_MODEL.url == "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit"
    )
    assert _videoseal.VIDEOSEAL_PAYLOAD_BITS == 256
