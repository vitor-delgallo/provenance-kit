"""Unit tests of the video watermark API with a fake backend (no model needed)."""

from __future__ import annotations

import hashlib
import sysconfig
import types

import numpy as np
import pytest

from provenance_kit import PayloadCodec, modelfiles
from provenance_kit.errors import BackendUnavailableError, ModelLoadError, WatermarkBackendError
from provenance_kit.video.frames import VideoFrame
from provenance_kit.video.watermark import (
    VIDEOSEAL_MODEL,
    RepetitionCode,
    VideoSealBackend,
    VideoWatermarkDetection,
    _api,
    _videoseal,
    detect_watermark,
    download_videoseal_models,
    embed_verified,
    embed_watermark,
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


def test_backend_errors_are_the_shared_type():
    from provenance_kit import errors
    from provenance_kit.video import watermark

    assert watermark.WatermarkBackendError is errors.WatermarkBackendError is WatermarkBackendError


@pytest.mark.parametrize(
    "scores",
    [np.full(256, np.nan, dtype=np.float32), np.zeros(10, np.float32), "not scores"],
)
def test_malformed_backend_scores_are_refused(scores):
    class Broken(FakeBackend):
        def score_frames(self, frames):
            for _ in frames:
                yield scores

    with pytest.raises(WatermarkBackendError, match="malformed"):
        detect_watermark(blank_frames(1), backend=Broken())


@pytest.mark.parametrize(
    "frame",
    [np.zeros((4, WIDTH), np.uint8), np.zeros((4, WIDTH, 3), np.float32), "frame"],
)
def test_malformed_backend_frames_are_refused(frame, monkeypatch):
    class Broken(FakeBackend):
        def embed_frames(self, frames, bits):
            for _ in frames:
                yield frame

    marked = embed_watermark(blank_frames(1), "0" * 256, backend=Broken())
    with pytest.raises(WatermarkBackendError, match="frame"):
        list(marked)
    encoder = fake_codec_round_trip(monkeypatch)
    with pytest.raises(WatermarkBackendError, match="frame"):
        embed_verified(blank_frames(1), "0" * 256, encoder=encoder, backend=Broken())


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
        {"download_timeout_seconds": 0},
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


def test_download_videoseal_models_fetches_the_pinned_file(tmp_path, monkeypatch):
    calls = []

    def ensure(specs, directory, *, timeout_seconds):
        calls.append((tuple(specs), directory, timeout_seconds))
        return tuple(directory / spec.name for spec in specs)

    monkeypatch.setattr(_videoseal, "ensure_model_files", ensure)
    directory = download_videoseal_models(tmp_path, timeout_seconds=9)
    assert directory == tmp_path.resolve()  # the directory, like the other backends
    assert calls == [((VIDEOSEAL_MODEL,), tmp_path.resolve(), 9)]
    download_videoseal_models(tmp_path)
    assert calls[-1][2] == modelfiles.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS


def test_backend_passes_the_download_timeout(tmp_path, monkeypatch):
    seen = {}

    def find(spec, **kwargs):
        seen.update(kwargs)
        raise ModelLoadError("stop here")

    monkeypatch.setattr(_videoseal, "import_optional", lambda *a, **k: fake_torch())
    monkeypatch.setattr(_videoseal, "find_model_file", find)
    with pytest.raises(ModelLoadError, match="stop here"):
        VideoSealBackend(model_dir=tmp_path, download=True, download_timeout_seconds=7)
    assert seen["download"] is True
    assert seen["timeout_seconds"] == 7


class _FakeScriptModel:
    def __init__(self) -> None:
        self.blender = types.SimpleNamespace(scaling_w=0.2)
        self.step_size = 4

    def eval(self) -> _FakeScriptModel:
        return self


def _torch_loading(record: list) -> types.SimpleNamespace:
    def jit_load(source, map_location=None):
        # TorchScript must come from the verified bytes, never from the path.
        assert not isinstance(source, (str, bytes))
        assert hasattr(source, "read")
        record.append((source.read(), map_location))
        return _FakeScriptModel()

    torch = fake_torch()
    torch.jit = types.SimpleNamespace(load=jit_load)
    return torch


def _custom_model(tmp_path) -> tuple[object, str]:
    content = b"pretend TorchScript archive"
    path = tmp_path / "converted.jit"
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def test_videoseal_loads_the_bytes_it_verified(tmp_path, monkeypatch):
    path, digest = _custom_model(tmp_path)
    loaded: list = []
    monkeypatch.setattr(_videoseal, "import_optional", lambda *a, **k: _torch_loading(loaded))
    backend = VideoSealBackend(model_path=path, sha256=digest, strength=0.5)
    assert loaded == [(b"pretend TorchScript archive", "cpu")]
    assert (backend.strength, backend.step_size) == (0.5, 4)


def test_a_model_file_replaced_after_verification_is_never_loaded(tmp_path, monkeypatch):
    # The file is hashed, then swapped before it is read again: the bytes
    # loaded must be the bytes checked (time-of-check to time-of-use).
    path, digest = _custom_model(tmp_path)
    real_find = _videoseal.find_model_file

    def find_then_swap(spec, **kwargs):
        verified = real_find(spec, **kwargs)
        verified.write_bytes(b"something else entirely")
        return verified

    loaded: list = []
    monkeypatch.setattr(_videoseal, "find_model_file", find_then_swap)
    monkeypatch.setattr(_videoseal, "import_optional", lambda *a, **k: _torch_loading(loaded))
    with pytest.raises(ModelLoadError, match="SHA-256"):
        VideoSealBackend(model_path=path, sha256=digest)
    assert loaded == []


def test_model_dir_inside_site_packages_is_refused(monkeypatch):
    inside = sysconfig.get_paths()["purelib"] + "/videoseal"
    with pytest.raises(ValueError, match="site-packages"):
        download_videoseal_models(inside)
    monkeypatch.setattr(_videoseal, "import_optional", lambda *a, **k: fake_torch())
    with pytest.raises(ValueError, match="site-packages"):
        VideoSealBackend(model_dir=inside)
