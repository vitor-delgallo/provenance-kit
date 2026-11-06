"""Unit tests of the audio watermark API with a fake backend (no model needed)."""

from __future__ import annotations

import hashlib
import json
import sysconfig
import types

import numpy as np
import pytest

from provenance_kit import PayloadCodec, modelfiles
from provenance_kit.audio.pcm import PcmAudio
from provenance_kit.audio.watermark import (
    AUDIOSEAL_DETECTOR,
    AUDIOSEAL_GENERATOR,
    MAX_ALIGNMENT_CANDIDATES,
    AlignmentSearch,
    AudioScores,
    AudioSealBackend,
    AudioWatermarkDetection,
    _api,
    _audioseal,
    detect_watermark,
    download_audioseal_models,
    embed_verified,
    embed_watermark,
)
from provenance_kit.errors import (
    BackendUnavailableError,
    MediaLimitExceededError,
    ModelLoadError,
    WatermarkBackendError,
)
from provenance_kit.limits import MediaLimits
from tests.audio.fakes import WAV, FakeTools, probe_json


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


def test_backend_errors_are_the_shared_type():
    from provenance_kit import errors
    from provenance_kit.audio import watermark

    assert watermark.WatermarkBackendError is errors.WatermarkBackendError is WatermarkBackendError


@pytest.mark.parametrize(
    "scores",
    [
        AudioScores(detection_probability=0.9, bit_probabilities=(0.5,) * 10),
        (0.9, (0.5,) * 16),
    ],
)
def test_malformed_backend_scores_are_refused(scores):
    class Broken(FakeBackend):
        def score(self, samples, sample_rate):
            return scores

    with pytest.raises(WatermarkBackendError):
        detect_watermark(silence(), 16000, backend=Broken())
    with pytest.raises(WatermarkBackendError):
        detect_watermark(silence(), 16000, backend=Broken(), alignment=AlignmentSearch())


@pytest.mark.parametrize(
    "returned",
    [np.zeros((2, 999), np.float32), np.zeros((2, 1000), np.float64), [[0.0]]],
)
def test_malformed_backend_output_is_refused(returned):
    class Broken(FakeBackend):
        def embed(self, samples, sample_rate, bits):
            return returned

    with pytest.raises(WatermarkBackendError, match="shape"):
        embed_watermark(silence(), 16000, "0" * 16, backend=Broken())


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


def test_verified_needs_check_bits_that_match():
    """``found`` is the detector's opinion; ``verified`` adds the payload's check bits."""
    backend = FakeBackend()
    codec = PayloadCodec(16, version_bits=0, check_bits=4)
    marked = embed_watermark(silence(), 16000, codec.encode(1234), backend=backend)
    assert detect_watermark(marked, 16000, backend=backend, codec=codec).verified
    tampered = np.array(marked)
    tampered[:, 0] *= -1
    bad = detect_watermark(tampered, 16000, backend=backend, codec=codec)
    assert bad.found
    assert not bad.verified
    # Without check bits nothing corroborates the detector: never verified.
    assert not detect_watermark(marked, 16000, backend=backend).verified
    no_check = PayloadCodec(16, version_bits=0)
    assert not detect_watermark(marked, 16000, backend=backend, codec=no_check).verified
    unfound = detect_watermark(marked, 16000, backend=FakeBackend(probability=0.1), codec=codec)
    assert not unfound.verified


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


# ------------------------------------------------------------ alignment search


class DelaySensitiveBackend(FakeBackend):
    """Confident only when the excerpt starts exactly on the marked samples.

    Mimics AudioSeal's sensitivity to sample offsets: one sample off and the
    bit probabilities are coin flips.
    """

    def __init__(self) -> None:
        super().__init__()
        self.windows: list[int] = []

    def score(self, samples, sample_rate):
        self.windows.append(samples.shape[1])
        head = samples[0, :16]
        if head.size == 16 and np.all(np.abs(head) == 0.5):
            probabilities = tuple(0.9 if v > 0 else 0.1 for v in head)
        else:
            probabilities = (0.5,) * 16
        return AudioScores(detection_probability=1.0, bit_probabilities=probabilities)


def delayed(samples: np.ndarray, delay: int) -> np.ndarray:
    return np.concatenate([np.zeros((samples.shape[0], delay), np.float32), samples], axis=1)


BITS = "1011001110001101"


def test_alignment_search_recovers_a_delayed_mark():
    backend = DelaySensitiveBackend()
    marked = embed_watermark(silence(frames=4000), 16000, BITS, backend=backend)
    shifted = delayed(marked, 37)
    plain = detect_watermark(shifted, 16000, backend=backend)
    assert plain.decoded_bits != BITS
    assert plain.bit_confidence == 0.0
    assert plain.offset_samples is None
    search = AlignmentSearch(max_offset_samples=100, step_samples=1, window_seconds=0.1)
    found = detect_watermark(shifted, 16000, backend=backend, alignment=search)
    assert found.offset_samples == 37
    assert found.bits == BITS
    assert found.bit_confidence == pytest.approx(0.8)


def test_alignment_search_cost_is_bounded():
    backend = DelaySensitiveBackend()
    marked = embed_watermark(silence(frames=4000), 16000, BITS, backend=backend)
    search = AlignmentSearch(max_offset_samples=40, step_samples=4, window_seconds=0.1)
    assert search.candidates == 11
    assert list(search.offsets()) == list(range(0, 41, 4))
    backend.windows.clear()
    detection = detect_watermark(delayed(marked, 24), 16000, backend=backend, alignment=search)
    assert detection.offset_samples == 24
    # One call per candidate on a 0.1 s excerpt, then the whole signal
    # unshifted and shifted (the shift must be corroborated against it).
    assert backend.windows == [1600] * 11 + [4024, 4000]
    # An offset off the step grid is not found (the step bounds the precision).
    off_grid = detect_watermark(delayed(marked, 25), 16000, backend=backend, alignment=search)
    assert off_grid.bit_confidence == 0.0


def test_alignment_search_keeps_the_signal_when_no_offset_stands_out():
    class Lukewarm(FakeBackend):
        """Weakly confident bits everywhere, a little more at offset 6."""

        def score(self, samples, sample_rate):
            level = 0.58 if samples.shape[1] == 994 else 0.55
            return AudioScores(detection_probability=0.3, bit_probabilities=(level,) * 16)

    samples = silence(frames=1000)
    floor = AlignmentSearch(max_offset_samples=10, step_samples=1, window_seconds=1.0)
    assert floor.min_bit_confidence == 0.2
    assert detect_watermark(samples, 1000, backend=Lukewarm(), alignment=floor).offset_samples == 0
    eager = AlignmentSearch(
        max_offset_samples=10, step_samples=1, window_seconds=1.0, min_bit_confidence=0.15
    )
    # The unshifted signal is found (0.3 >= 0.25), so the shift may refine it.
    assert (
        detect_watermark(
            samples, 1000, backend=Lukewarm(), alignment=eager, threshold=0.25
        ).offset_samples
        == 6
    )


class FooledAtOffset(FakeBackend):
    """Unmarked audio that the detector takes for marked once 6 samples are dropped.

    Mimics an unmarked click train (measured with AudioSeal: detection 0.14
    unshifted, 0.95 after dropping 380 samples, excerpt bit confidence 0.205,
    payload check bits wrong). The samples are a ramp, so the first value
    tells which offset the backend is looking at.
    """

    def __init__(self, bits: str) -> None:
        super().__init__()
        self.bits = bits

    def score(self, samples, sample_rate):
        if int(samples[0, 0]) == 6:
            probabilities = tuple(0.9 if b == "1" else 0.1 for b in self.bits)
            return AudioScores(detection_probability=0.95, bit_probabilities=probabilities)
        return AudioScores(detection_probability=0.1, bit_probabilities=(0.5,) * 16)


def ramp(frames: int = 1000) -> np.ndarray:
    return np.arange(frames, dtype=np.float32)[None, :]


def test_alignment_search_does_not_turn_unmarked_audio_into_a_detection():
    search = AlignmentSearch(max_offset_samples=10, step_samples=1, window_seconds=0.1)
    codec = PayloadCodec(16, version_bits=0, check_bits=4)
    valid = codec.encode(99)
    invalid = valid[:-1] + ("1" if valid[-1] == "0" else "0")
    # Without check bits nothing corroborates the shift: the signal, not
    # found unshifted, stays unshifted and not found.
    alone = detect_watermark(ramp(), 1000, backend=FooledAtOffset(valid), alignment=search)
    assert alone.offset_samples == 0
    assert not alone.found
    # With a codec, the shifted payload must pass the check bits.
    wrong = detect_watermark(
        ramp(), 1000, backend=FooledAtOffset(invalid), codec=codec, alignment=search
    )
    assert wrong.offset_samples == 0
    assert not wrong.found
    assert not wrong.verified
    right = detect_watermark(
        ramp(), 1000, backend=FooledAtOffset(valid), codec=codec, alignment=search
    )
    assert right.offset_samples == 6
    assert right.found
    assert right.verified
    assert right.payload is not None
    assert right.payload.identifier == 99


class WeakAfterShift(FakeBackend):
    """Found unshifted (0.9); 6 samples dropped, the bits stand out but the detection is 0.2.

    The samples are a ramp, so the first value tells which offset is scored.
    """

    def __init__(self, bits: str) -> None:
        super().__init__()
        self.bits = bits

    def score(self, samples, sample_rate):
        sure = tuple(0.9 if b == "1" else 0.1 for b in self.bits)
        if int(samples[0, 0]) == 6:
            return AudioScores(detection_probability=0.2, bit_probabilities=sure)
        return AudioScores(detection_probability=0.9, bit_probabilities=(0.6,) * 16)


@pytest.mark.parametrize(
    "codec",
    [None, PayloadCodec(16, version_bits=4, check_bits=0), PayloadCodec(16, check_bits=4)],
)
def test_alignment_search_never_loses_a_found_signal(codec):
    """R2-M07: a shift is used only if the shifted detection is found.

    With no check bits the shift was accepted whenever the unshifted signal
    was found, and the shifted detection, not found, was returned. With check
    bits, payload bits that pass them were enough even when the shifted
    signal was not found.
    """
    search = AlignmentSearch(max_offset_samples=10, step_samples=1, window_seconds=0.1)
    bits = (codec or PayloadCodec(16, version_bits=0, check_bits=4)).encode(5)
    plain = detect_watermark(ramp(), 1000, backend=WeakAfterShift(bits), codec=codec)
    searched = detect_watermark(
        ramp(), 1000, backend=WeakAfterShift(bits), codec=codec, alignment=search
    )
    assert plain.found
    assert searched.found
    assert searched.offset_samples == 0
    assert searched.confidence == plain.confidence


def test_alignment_search_stops_at_the_end_of_the_signal():
    backend = DelaySensitiveBackend()
    search = AlignmentSearch(max_offset_samples=2048, step_samples=1, window_seconds=0.1)
    detection = detect_watermark(silence(frames=10), 16000, backend=backend, alignment=search)
    assert detection.offset_samples == 0  # ties go to the smallest offset
    assert len(backend.windows) == 10 + 1


def test_alignment_search_validation():
    defaults = AlignmentSearch()
    assert (defaults.max_offset_samples, defaults.step_samples) == (2048, 2)
    assert defaults.candidates == 1025
    for kwargs in (
        {"step_samples": 0},
        {"max_offset_samples": -1},
        {"max_offset_samples": 200_000, "step_samples": 100},
        {"max_offset_samples": MAX_ALIGNMENT_CANDIDATES, "step_samples": 1},
        {"window_seconds": 0.05},
        {"window_seconds": float("nan")},
        {"min_bit_confidence": 1.5},
        {"min_bit_confidence": -0.1},
    ):
        with pytest.raises(ValueError):
            AlignmentSearch(**kwargs)
    with pytest.raises(TypeError):
        AlignmentSearch(step_samples=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        detect_watermark(silence(), 16000, backend=FakeBackend(), alignment=100)  # type: ignore[arg-type]


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


def test_embed_verified_bounds_the_decoded_read_back():
    # The encoder's output is untrusted media too: its declared decoded size
    # is checked against max_decoded_audio_bytes before it is decoded.
    report = json.loads(probe_json(channels=8, sample_rate="192000", duration="20"))
    report["format"]["duration"] = "20"
    tools = FakeTools(ffprobe=json.dumps(report).encode())
    limits = MediaLimits(max_decoded_audio_bytes=1_000_000)
    with pytest.raises(MediaLimitExceededError) as info:
        embed_verified(
            silence(),
            16000,
            "0" * 16,
            encoder=lambda s, r: WAV,
            backend=FakeBackend(),
            tools=tools,  # type: ignore[arg-type]
            limits=limits,
        )
    assert info.value.limit == "max_decoded_audio_bytes"
    assert tools.args_of("ffmpeg") == []


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
        {"download_timeout_seconds": -1.0},
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


def test_download_audioseal_models_fetches_both_pinned_files(tmp_path, monkeypatch):
    calls = []

    def ensure(specs, directory, *, timeout_seconds):
        calls.append((tuple(specs), directory, timeout_seconds))
        return tuple(directory / spec.name for spec in specs)

    monkeypatch.setattr(_audioseal, "ensure_model_files", ensure)
    directory = download_audioseal_models(tmp_path, timeout_seconds=9)
    assert directory == tmp_path.resolve()  # the directory, like the other backends
    assert calls == [((AUDIOSEAL_GENERATOR, AUDIOSEAL_DETECTOR), tmp_path.resolve(), 9)]
    download_audioseal_models(tmp_path)
    assert calls[-1][2] == modelfiles.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS


def test_backend_passes_the_download_timeout(tmp_path, monkeypatch):
    seen = []

    def find(spec, **kwargs):
        seen.append(kwargs)
        return tmp_path / spec.name

    monkeypatch.setattr(_audioseal, "import_optional", fake_import)
    monkeypatch.setattr(_audioseal, "find_model_file", find)
    with pytest.raises(ModelLoadError):  # stops at building the fake models
        AudioSealBackend(model_dir=tmp_path, download=True, download_timeout_seconds=7)
    assert [call["timeout_seconds"] for call in seen] == [7, 7]
    assert [call["download"] for call in seen] == [True, True]


def test_the_models_are_built_from_the_bytes_that_were_verified(tmp_path, monkeypatch):
    """A checkpoint replaced after its check is refused, not unpickled.

    find_model_file verifies the file on disk; the backend then reads the
    bytes once more, checks them, and builds the model from them, so the
    file cannot be swapped between the check and torch.load.
    """
    swapped = []

    def find(spec, **kwargs):
        path = tmp_path / spec.name
        path.write_bytes(b"replaced after the check")  # what an attacker swapped in
        swapped.append(spec.name)
        return path

    def no_build(*args, **kwargs):
        pytest.fail("a model was built from unverified bytes")

    monkeypatch.setattr(_audioseal, "import_optional", fake_import)
    monkeypatch.setattr(_audioseal, "find_model_file", find)
    monkeypatch.setattr(_audioseal, "_build", no_build)
    with pytest.raises(ModelLoadError, match="SHA-256"):
        AudioSealBackend(model_dir=tmp_path)
    assert swapped


def test_read_verified_checks_the_bytes_it_returns(tmp_path):
    content = b"checkpoint bytes"
    spec = modelfiles.ModelFile(
        name="tiny.pth",
        url="https://example.invalid/tiny.pth",
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
    )
    path = tmp_path / spec.name
    path.write_bytes(content)
    assert _audioseal._read_verified(path, spec, None) == content
    path.write_bytes(content + b"!")  # one byte more than pinned
    with pytest.raises(ModelLoadError, match="SHA-256"):
        _audioseal._read_verified(path, spec, None)
    other = b"weights you converted yourself"
    path.write_bytes(other)
    custom = hashlib.sha256(other).hexdigest().upper()  # find_model_file accepts any case
    assert _audioseal._read_verified(path, spec, f" {custom} ") == other
    with pytest.raises(ModelLoadError, match="SHA-256"):
        _audioseal._read_verified(path, spec, "0" * 64)
    path.unlink()
    with pytest.raises(ModelLoadError, match="could not be read"):
        _audioseal._read_verified(path, spec, None)


def test_model_dir_inside_site_packages_is_refused(monkeypatch):
    inside = sysconfig.get_paths()["purelib"] + "/audioseal"
    with pytest.raises(ValueError, match="site-packages"):
        download_audioseal_models(inside)
    monkeypatch.setattr(_audioseal, "import_optional", fake_import)
    with pytest.raises(ValueError, match="site-packages"):
        AudioSealBackend(model_dir=inside)
