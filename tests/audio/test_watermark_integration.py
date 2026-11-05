"""AudioSeal with the real checkpoints and real codecs.

The checkpoints (59 MB and 35 MB) are read from ``tests.support.model_dir``
(the git-ignored ``.cache/models/audioseal/``, or
``$PROVENANCE_KIT_MODEL_DIR/audioseal``). If they are missing the tests skip,
unless ``--run-network`` is given, in which case they are downloaded there
from the pinned Hugging Face revision and verified against the pinned
SHA-256. Run with ``-s`` to print the measured numbers; they are recorded in
``docs/audio.md``.
"""

from __future__ import annotations

import importlib.util
import os
import pickle
import time

import numpy as np
import pytest

from provenance_kit import PayloadCodec
from provenance_kit.audio.pcm import decode_pcm
from provenance_kit.audio.watermark import (
    AUDIOSEAL_DETECTOR,
    AUDIOSEAL_GENERATOR,
    AlignmentSearch,
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
from tests.support import find_tools, model_dir, require_module, require_tool

pytestmark = pytest.mark.integration

MODEL_DIR = model_dir("audioseal")
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
            f"the AudioSeal checkpoints are not in {MODEL_DIR}: run with --run-network "
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


@pytest.mark.parametrize(
    ("codec_name", "delay"), [("adts-aac-128k-pipe", 1024), ("mp3-128k-pipe", 1106)]
)
def test_alignment_search_recovers_unsignaled_delays(backend, sources, tools, codec_name, delay):
    """The opt-in search finds the encoder delay that these pipelines do not record.

    ffmpeg's AAC encoder primes 1024 samples; LAME's 576 + 529 samples, plus
    one from the decoder, give 1106. Without the search the bits are lost
    (bit accuracy 0.5 to 0.75); with it they come back. Two payloads per
    codec, to bound the run time (each search takes about 20 s on a CPU).
    """
    args, suffix, aligned = CODECS[codec_name]
    assert suffix is None
    assert not aligned
    search = AlignmentSearch()  # the defaults: offsets 0 to 2048, step 2, 0.5 s excerpts
    for bits in PAYLOADS[:2]:
        marked = embed_watermark(sources["broadband"], 44100, bits, backend=backend)
        decoded = decode_pcm(encode_pcm(tools, marked, 44100, args), tools=tools)
        plain = detect_watermark(decoded.samples, decoded.sample_rate, backend=backend, codec=CODEC)
        started = time.perf_counter()
        found = detect_watermark(
            decoded.samples, decoded.sample_rate, backend=backend, codec=CODEC, alignment=search
        )
        elapsed = time.perf_counter() - started
        print(
            f"\nbroadband {codec_name}: without search p={plain.confidence:.3f} bits "
            f"{_matching(plain.decoded_bits, bits)}/16; with search offset {found.offset_samples}, "
            f"p={found.confidence:.3f} bits {_matching(found.decoded_bits, bits)}/16, "
            f"{search.candidates} candidates in {elapsed:.1f} s"
        )
        # Near the encoder delay: the bit confidence peaks within a few samples
        # of it (measured: 1024 for ADTS, 1106 or 1116 for the MP3 pipe), and
        # all 16 bits come back.
        assert found.offset_samples is not None
        assert abs(found.offset_samples - delay) <= 16
        assert found.found
        assert found.verified
        assert found.bits == bits
        assert found.payload is not None
        assert found.payload.valid
        assert plain.decoded_bits != bits


def click_train(rate: int, hz: int, *, seconds: float, phase: int = 0, level: float = 0.9):
    """Unmarked one-sample clicks every ``rate // hz`` samples, shaped (1, n)."""
    clicks = ((np.arange(int(seconds * rate)) + phase) % (rate // hz)) == 0
    return (clicks * level).astype(np.float32)[None, :]


def test_alignment_search_on_unmarked_audio(backend, sources):
    """The search does not turn unmarked audio into a detection.

    Broadband audio: no offset reaches the bit-confidence floor (best 0.04 to
    0.05). A 100 Hz click train does (0.205 at offset 380), its bits repeat
    in every window, and dropping those 380 samples raised its detection
    probability from 0.14 to 0.95: the shift is not corroborated (no check
    bits to pass, and the signal is not found unshifted), so it is not used.
    """
    search = AlignmentSearch()
    signals = {
        "broadband": (sources["broadband"], 44100),
        "100 Hz clicks": (click_train(44100, 100, seconds=4.0, level=0.2), 44100),
    }
    for name, (samples, rate) in signals.items():
        detection = detect_watermark(samples, rate, backend=backend, alignment=search)
        print(
            f"\n{name} unmarked with search: offset {detection.offset_samples}, "
            f"detection probability {detection.confidence:.3f}"
        )
        assert not detection.found
        assert detection.offset_samples == 0


def test_unmarked_click_trains_fool_the_detector_but_not_the_check_bits(backend):
    """``found`` is not evidence; ``verified`` (check bits) is.

    Measured: 12 of 24 unmarked click trains (50 and 100 Hz; 16, 44.1 and
    48 kHz; four phases) reach the 0.5 threshold, with detection
    probabilities up to 0.97; none passes 4 check bits.
    """
    found = verified = 0
    for rate in (44100, 48000):
        period = rate // 50
        for phase in (0, period // 4, period // 2):
            samples = click_train(rate, 50, seconds=6.0, phase=phase)
            detection = detect_watermark(samples, rate, backend=backend, codec=CODEC)
            print(f"\n50 Hz clicks @ {rate} phase {phase}: p={detection.confidence:.3f}")
            found += detection.found
            verified += detection.verified
    assert found >= 4  # measured 6 of 6: the detector alone is fooled
    assert verified == 0


def _matching(read: str | None, bits: str) -> int:
    return sum(a == b for a, b in zip(read or "", bits, strict=False))


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
