"""Unit tests of the in-memory windowed-sinc resampler."""

from __future__ import annotations

import numpy as np
import pytest

from provenance_kit.audio import _resample
from provenance_kit.audio._resample import resample, resampled_length


def tone(rate: int, seconds: float, frequency: float) -> np.ndarray:
    t = np.arange(round(rate * seconds)) / rate
    return np.sin(2 * np.pi * frequency * t)


@pytest.mark.parametrize(
    ("old", "new"),
    [(44100, 16000), (48000, 16000), (16000, 44100), (16000, 48000), (22050, 16000), (8000, 16000)],
)
def test_tones_survive_resampling(old, new):
    out = resample(tone(old, 1.0, 1000.0), old, new)
    assert out.dtype == np.float32
    assert out.size == resampled_length(round(old * 1.0), old, new)
    expected = tone(new, 1.0, 1000.0)[: out.size]
    middle = slice(new // 10, -new // 10)  # ignore the zero-padded edges
    assert np.max(np.abs(out[middle] - expected[middle])) < 2e-3


def test_frequencies_above_the_new_nyquist_are_removed():
    out = resample(tone(48000, 1.0, 12000.0), 48000, 16000)  # 12 kHz > 8 kHz Nyquist
    assert np.sqrt(np.mean(out[1600:-1600] ** 2)) < 0.01


def test_round_trip_keeps_the_signal(rng):
    t = np.arange(16000) / 16000
    frequencies = rng.uniform(50, 6000, size=20)  # below the 7.5 kHz pass band edge
    phases = rng.uniform(0, 2 * np.pi, size=20)
    signal = np.sum(np.sin(2 * np.pi * frequencies[:, None] * t + phases[:, None]), axis=0) / 20
    back = resample(resample(signal, 16000, 44100), 44100, 16000)
    assert back.size == signal.size
    error = back[800:-800] - signal[800:-800]
    assert np.sqrt(np.mean(error**2)) < 0.01 * np.sqrt(np.mean(signal**2)) + 1e-3


def test_direct_and_polyphase_paths_agree(rng, monkeypatch):
    signal = np.convolve(rng.normal(size=4000), np.ones(4) / 4, mode="same")
    polyphase = resample(signal, 44100, 16000)
    monkeypatch.setattr(_resample, "_MAX_TABLE", 1)
    direct = resample(signal, 44100, 16000)
    np.testing.assert_allclose(polyphase, direct, atol=1e-5)


def test_awkward_ratios_use_the_direct_path():
    out = resample(tone(44101, 0.2, 500.0), 44101, 16000)
    expected = tone(16000, 0.2, 500.0)[: out.size]
    assert np.max(np.abs(out[400:-400] - expected[400:-400])) < 2e-3


def test_identity_and_empty():
    signal = np.arange(10, dtype=np.float64)
    np.testing.assert_array_equal(resample(signal, 16000, 16000), signal.astype(np.float32))
    assert resample(np.zeros(0), 44100, 16000).size == 0


def test_invalid_input():
    with pytest.raises(ValueError):
        resample(np.zeros((2, 10)), 44100, 16000)
    with pytest.raises(ValueError):
        resample(np.zeros(10), 0, 16000)
