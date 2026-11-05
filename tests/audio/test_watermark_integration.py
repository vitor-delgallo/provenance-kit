"""AudioSeal with the real checkpoints and real codecs.

The checkpoints (59 MB and 35 MB) are read from ``model-cache/audioseal/``
(git-ignored). If they are missing the tests skip, unless ``--run-network`` is
given, in which case they are downloaded there from the pinned Hugging Face
revision and verified against the pinned SHA-256. Run with ``-s`` to print
the measured numbers; they are recorded in ``docs/audio.md``.
"""

from __future__ import annotations

import importlib.util
import os
import pickle

import numpy as np
import pytest

from provenance_kit import PayloadCodec
from provenance_kit.audio.pcm import decode_pcm
from provenance_kit.audio.watermark import (
    AUDIOSEAL_DETECTOR,
    AUDIOSEAL_GENERATOR,
    AudioSealBackend,
    _audioseal,
    detect_watermark,
    download_audioseal_models,
    embed_verified,
    embed_watermark,
)
from provenance_kit.errors import ModelLoadError
from provenance_kit.tools import ExternalTools, Tool
from tests.audio.media import BROADBAND, TONES, encode_pcm, ffmpeg, lavfi
from tests.support import REPO_ROOT, find_tools, require_module, require_tool

pytestmark = pytest.mark.integration

MODEL_DIR = REPO_ROOT / "model-cache" / "audioseal"
CODEC = PayloadCodec(16, version_bits=0, check_bits=4)
PAYLOADS = [CODEC.encode(identifier) for identifier in (1, 777, 2024, 4095)]
# (ffmpeg arguments, output file suffix or None for a pipe, delay signaled?).
# AudioSeal is sensitive to sample-level misalignment, so what matters is
# whether the container records the encoder delay: MP4/M4A (edit list), MP3
# written to a seekable file (LAME/Info tag) and Ogg Opus (pre-skip) do; raw
# ADTS AAC and MP3 written to a pipe do not, and the decoded audio is shifted.
CODECS = {
    "m4a-aac-64k": (["-c:a", "aac", "-b:a", "64k"], ".m4a", True),
    "m4a-aac-128k": (["-c:a", "aac", "-b:a", "128k"], ".m4a", True),
    "m4a-aac-96k-22khz": (["-ar", "22050", "-c:a", "aac", "-b:a", "96k"], ".m4a", True),
    "mp3-file-64k": (["-c:a", "libmp3lame", "-b:a", "64k"], ".mp3", True),
    "mp3-file-128k": (["-c:a", "libmp3lame", "-b:a", "128k"], ".mp3", True),
    "mp3-file-96k-mono": (["-ac", "1", "-c:a", "libmp3lame", "-b:a", "96k"], ".mp3", True),
    "ogg-opus-32k": (["-c:a", "libopus", "-b:a", "32k", "-f", "ogg"], None, True),
    "adts-aac-128k-pipe": (["-c:a", "aac", "-b:a", "128k", "-f", "adts"], None, False),
    "mp3-128k-pipe": (["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"], None, False),
}
SOURCES = {"broadband": BROADBAND, "tones": TONES}


def require_audioseal() -> None:
    # Only check that it is installed: importing audioseal here would trip
    # upstream FutureWarnings (errors in this suite) that the backend
    # silences while it imports the package.
    if importlib.util.find_spec("audioseal") is None:
        pytest.skip('needs audioseal: pip install "provenance-kit[audio-watermark]"')
    require_module("torch", extra="audio-watermark")


def ensure_models(request: pytest.FixtureRequest) -> None:
    require_audioseal()
    present = all(
        (MODEL_DIR / spec.name).is_file() for spec in (AUDIOSEAL_GENERATOR, AUDIOSEAL_DETECTOR)
    )
    if present:
        return
    if not request.config.getoption("--run-network"):
        pytest.skip(
            "the AudioSeal checkpoints are not in model-cache/audioseal/: run with --run-network "
            "to download them"
        )
    download_audioseal_models(MODEL_DIR)


@pytest.fixture(scope="module")
def tools() -> ExternalTools:
    require_tool(Tool.FFMPEG)
    require_tool(Tool.FFPROBE)
    return find_tools()


@pytest.fixture(scope="module")
def backend(request) -> AudioSealBackend:
    ensure_models(request)
    return AudioSealBackend(model_dir=MODEL_DIR)


@pytest.fixture(scope="module")
def sources(tools) -> dict[str, np.ndarray]:
    out = {}
    for name, expression in SOURCES.items():
        data = ffmpeg(
            tools, [*lavfi(expression.format(rate=44100), 8), "-ac", "2", "-f", "wav", "pipe:1"]
        )
        out[name] = decode_pcm(data, tools=tools).samples
    return out


@pytest.mark.network
def test_download_verifies_the_pinned_checkpoints():
    require_audioseal()
    generator, detector = download_audioseal_models(MODEL_DIR)
    assert generator.stat().st_size == AUDIOSEAL_GENERATOR.size
    assert detector.stat().st_size == AUDIOSEAL_DETECTOR.size


def test_unmarked_audio_is_not_found(backend, sources):
    for name, samples in sources.items():
        detection = detect_watermark(samples, 44100, backend=backend)
        print(f"\n{name} unmarked: detection probability {detection.confidence:.3f}")
        assert not detection.found


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_raw_round_trip(backend, sources, source):
    samples = sources[source]
    correct = 0
    for bits in PAYLOADS:
        marked = embed_watermark(samples, 44100, bits, backend=backend)
        assert marked.shape == samples.shape
        assert marked.dtype == np.float32
        detection = detect_watermark(marked, 44100, backend=backend, codec=CODEC)
        assert detection.found
        assert detection.confidence is not None
        assert detection.confidence > 0.9
        correct += sum(a == b for a, b in zip(detection.decoded_bits or "", bits, strict=True))
    noise = marked - samples
    snr = 10 * np.log10(np.sum(samples**2) / np.sum(noise**2))
    accuracy = correct / (16 * len(PAYLOADS))
    print(f"\n{source} raw: SNR {snr:.1f} dB, bit accuracy {accuracy:.3f}")
    assert snr > 20
    if source == "broadband":
        assert accuracy >= 0.95


@pytest.mark.parametrize("codec_name", sorted(CODECS))
@pytest.mark.parametrize("source", sorted(SOURCES))
def test_accuracy_after_real_codecs(  # noqa: PLR0917 - pytest passes fixtures positionally
    backend, sources, tools, tmp_path, source, codec_name
):
    args, suffix, aligned = CODECS[codec_name]

    def encoder(samples: np.ndarray, rate: int) -> bytes:
        if suffix is None:
            return encode_pcm(tools, samples, rate, args)
        target = tmp_path / f"out{suffix}"
        encode_pcm(tools, samples, rate, [*args, "-y", str(target)])
        return target.read_bytes()

    correct = 0
    probabilities = []
    verified = 0
    for bits in PAYLOADS:
        result = embed_verified(
            sources[source],
            44100,
            bits,
            encoder=encoder,
            backend=backend,
            codec=CODEC,
            tools=tools,
        )
        probabilities.append(result.detection.confidence or 0.0)
        correct += 16 - result.bit_errors
        verified += result.verified
    accuracy = correct / (16 * len(PAYLOADS))
    print(
        f"\n{source} {codec_name}: detection probability {min(probabilities):.3f} to "
        f"{max(probabilities):.3f}, bit accuracy {accuracy:.3f}, "
        f"verified {verified}/{len(PAYLOADS)}"
    )
    if aligned:
        assert min(probabilities) >= 0.9  # detected after every delay-signaling codec
        if source == "broadband":
            assert accuracy >= 0.9


def test_mono_downmix_keeps_the_mark(backend, sources):
    marked = embed_watermark(sources["broadband"], 44100, PAYLOADS[1], backend=backend)
    mono = marked.mean(axis=0, keepdims=True)
    detection = detect_watermark(mono, 44100, backend=backend, codec=CODEC)
    assert detection.found
    assert detection.decoded_bits == PAYLOADS[1]


def test_other_sample_rates(backend, sources):
    samples = sources["broadband"]
    for rate in (16000, 48000):
        marked = embed_watermark(samples, rate, PAYLOADS[2], backend=backend)
        detection = detect_watermark(marked, rate, backend=backend)
        assert detection.found
        assert detection.decoded_bits == PAYLOADS[2]


def test_the_checkpoint_loader_refuses_code(tmp_path, request):
    torch = require_module("torch", extra="audio-watermark")
    marker = tmp_path / "pwned"

    class Exploit:
        def __reduce__(self):
            return (os.mkdir, (str(marker),))

    path = tmp_path / "evil.pth"
    torch.save({"model": {}, "extra": Exploit()}, str(path))
    with pytest.raises(ModelLoadError):
        _audioseal._load_state_dict(torch, path)
    assert not marker.exists()
    legacy = tmp_path / "legacy.pth"
    legacy.write_bytes(pickle.dumps({"model": {}}))
    with pytest.raises(ModelLoadError):
        _audioseal._load_state_dict(torch, legacy)


def test_tampered_checkpoints_are_refused(tmp_path, request):
    ensure_models(request)
    (tmp_path / AUDIOSEAL_GENERATOR.name).write_bytes(b"not a checkpoint")
    (tmp_path / AUDIOSEAL_DETECTOR.name).write_bytes(
        (MODEL_DIR / AUDIOSEAL_DETECTOR.name).read_bytes()
    )
    with pytest.raises(ModelLoadError, match="SHA-256"):
        AudioSealBackend(model_dir=tmp_path)
