"""Unit tests of the audio watermark API with a fake backend (no model needed)."""

from __future__ import annotations

import types

import numpy as np
import pytest

from provenance_kit import PayloadCodec
from provenance_kit.audio.pcm import PcmAudio
from provenance_kit.audio.watermark import (
    AUDIOSEAL_DETECTOR,
    AUDIOSEAL_GENERATOR,
    AudioScores,
    AudioSealBackend,
    AudioWatermarkDetection,
    _api,
    _audioseal,
    detect_watermark,
    embed_verified,
    embed_watermark,
)
from provenance_kit.errors import BackendUnavailableError, ModelLoadError


class FakeBackend:
    """Hides the 16 bits in the sign of the first 16 samples' offset.

    The detection probability is the share of samples whose value ends in the
    marker pattern; good enough to drive the domain logic.
    """

    payload_bits = 16

    def __init__(self, probability: float = 1.0) -> None:
        self.probability = probability
        self.calls: list[int] = []

    def embed(self, samples, sample_rate, bits):
        self.calls.append(sample_rate)
        out = np.array(samples, dtype=np.float32, copy=True)
        for index, bit in enumerate(bits):
            out[:, index] = 0.5 if bit == "1" else -0.5
        return out

    def score(self, samples, sample_rate):
        probabilities = tuple(0.9 if samples[0, i] > 0 else 0.1 for i in range(16))
        return AudioScores(detection_probability=self.probability, bit_probabilities=probabilities)


def silence(channels: int = 2, frames: int = 1000) -> np.ndarray:
    return np.zeros((channels, frames), dtype=np.float32)


def test_embed_and_detect():
    backend = FakeBackend()
    bits = "1011001110001101"
    marked = embed_watermark(silence(), 44100, bits, backend=backend)
    assert marked.shape == (2, 1000)
    assert backend.calls == [44100]
    detection = detect_watermark(marked, 44100, backend=backend)
    assert isinstance(detection, AudioWatermarkDetection)
    assert detection.found
    assert detection.confidence == 1.0
    assert detection.bits == bits
    assert detection.decoded_bits == bits
    assert detection.bit_confidence == pytest.approx(0.8)


def test_low_probability_is_not_found():
    backend = FakeBackend(probability=0.2)
    detection = detect_watermark(silence(), 16000, backend=backend)
    assert not detection.found
    assert detection.bits is None
    assert detection.payload is None
    assert detection.confidence == 0.2
    assert detect_watermark(silence(), 16000, backend=backend, threshold=0.1).found


def test_codec_check_bits_flag_invalid_payloads():
    backend = FakeBackend()
    codec = PayloadCodec(16, version_bits=0, check_bits=4)
    marked = embed_watermark(silence(), 16000, codec.encode(1234), backend=backend)
    good = detect_watermark(marked, 16000, backend=backend, codec=codec)
    assert good.payload is not None
    assert good.payload.valid
    assert good.payload.identifier == 1234
    tampered = np.array(marked)
    tampered[:, 0] *= -1
    bad = detect_watermark(tampered, 16000, backend=backend, codec=codec)
    assert bad.found
    assert bad.payload is not None
    assert not bad.payload.valid


def test_argument_validation():
    backend = FakeBackend()
    with pytest.raises(ValueError):
        embed_watermark(silence(), 16000, "0101", backend=backend)
    with pytest.raises(ValueError):
        embed_watermark(silence(), 0, "0" * 16, backend=backend)
    with pytest.raises(ValueError):
        detect_watermark(silence(), 16000, backend=backend, codec=PayloadCodec(61))
    with pytest.raises(ValueError):
        detect_watermark(silence(), 16000, backend=backend, threshold=1.5)
    with pytest.raises(TypeError):
        detect_watermark(silence(), 16000, backend=backend, codec="codec")  # type: ignore[arg-type]


def test_audio_scores_validation():
    with pytest.raises(ValueError):
        AudioScores(detection_probability=1.5, bit_probabilities=())
    with pytest.raises(ValueError):
        AudioScores(detection_probability=0.5, bit_probabilities=(float("nan"),))


def patch_decoder(monkeypatch, *, invert: bool = False):
    """Makes the 'encoded' bytes raw float32 frames at 16 kHz."""

    def encoder(samples, sample_rate):
        assert sample_rate == 22050
        data = -samples if invert else samples
        return np.ascontiguousarray(data, dtype="<f4").tobytes()

    def fake_decode_pcm(data, **kwargs):
        samples = np.frombuffer(data, dtype="<f4").reshape(2, -1).copy()
        return PcmAudio(samples=samples, sample_rate=22050)

    monkeypatch.setattr(_api, "decode_pcm", fake_decode_pcm)
    return encoder


def test_embed_verified_round_trip(monkeypatch):
    encoder = patch_decoder(monkeypatch)
    codec = PayloadCodec(16, version_bits=0, check_bits=4)
    bits = codec.encode(77)
    result = embed_verified(
        silence(), 22050, bits, encoder=encoder, backend=FakeBackend(), codec=codec
    )
    assert result.verified
    assert result.bit_errors == 0
    assert result.detection.payload is not None
    assert result.detection.payload.identifier == 77


def test_embed_verified_reports_bit_errors(monkeypatch):
    encoder = patch_decoder(monkeypatch, invert=True)
    result = embed_verified(silence(), 22050, "1" * 16, encoder=encoder, backend=FakeBackend())
    assert not result.verified
    assert result.bit_errors == 16


def test_embed_verified_requires_bytes():
    with pytest.raises(TypeError):
        embed_verified(silence(), 16000, "0" * 16, encoder=lambda s, r: [1], backend=FakeBackend())


# ------------------------------------------------------------ backend construction


def fake_import(name, **kwargs):
    if name == "torch":
        return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    return types.SimpleNamespace()


def test_audioseal_backend_requires_the_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(_audioseal, "import_optional", fake_import)
    with pytest.raises(ModelLoadError, match="download=True"):
        AudioSealBackend(model_dir=tmp_path)


def test_audioseal_backend_reports_missing_packages(monkeypatch):
    def missing(*args, **kwargs):
        raise BackendUnavailableError.missing_extra("AudioSeal", "audio-watermark")

    monkeypatch.setattr(_audioseal, "import_optional", missing)
    with pytest.raises(BackendUnavailableError) as info:
        AudioSealBackend()
    assert info.value.extra == "audio-watermark"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strength": 0.0},
        {"strength": 11.0},
        {"chunk_seconds": 0.5},
        {"frame_threshold": 2.0},
        {"device": "gpu"},
    ],
)
def test_audioseal_backend_validates_arguments_before_loading(kwargs, monkeypatch):
    monkeypatch.setattr(_audioseal, "import_optional", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(ValueError):
        AudioSealBackend(**kwargs)


def test_pinned_checkpoints_come_from_the_current_mit_revision():
    for spec in (AUDIOSEAL_GENERATOR, AUDIOSEAL_DETECTOR):
        assert spec.url.startswith(
            "https://huggingface.co/facebook/audioseal/resolve/"
            "3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba/"
        )
    assert AUDIOSEAL_GENERATOR.sha256.startswith("7a845b5f")
    assert AUDIOSEAL_DETECTOR.sha256.startswith("8a78e8a8")


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros(10, np.float32),
        np.zeros((1, 0), np.float32),
        np.zeros((1, 10), np.int16),
        np.full((1, 4), np.inf, np.float32),
        [[0.0, 0.1]],
    ],
)
def test_check_samples_rejects_bad_input(samples):
    with pytest.raises((TypeError, ValueError)):
        _audioseal.check_samples(samples)


def test_check_samples_converts_to_float32():
    out = _audioseal.check_samples(np.zeros((2, 5), np.float64))
    assert out.dtype == np.float32
    assert out.shape == (2, 5)


def test_chunks_fold_short_tails(monkeypatch):
    backend = object.__new__(AudioSealBackend)
    backend._chunk = 16000 * 10
    assert list(backend._chunks(0)) == []
    assert list(backend._chunks(5)) == [(0, 5)]
    assert list(backend._chunks(16000 * 20 + 8000)) == [(0, 160000), (160000, 328000)]
    assert list(backend._chunks(16000 * 12)) == [(0, 160000), (160000, 192000)]
