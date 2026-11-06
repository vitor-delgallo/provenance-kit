"""Unit tests of PCM decoding and audio probing with fake tools."""

from __future__ import annotations

import dataclasses
import json
import tracemalloc

import numpy as np
import pytest

from provenance_kit.audio import pcm
from provenance_kit.audio.pcm import PcmAudio, _parse_probe_output, decode_pcm, probe_audio
from provenance_kit.errors import (
    MediaDecodeError,
    MediaLimitExceededError,
    UnsupportedMediaError,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType
from tests.audio.fakes import M4A_MDAT_FIRST, WAV, FakeTools, probe_json
from tests.support import assert_imports_lightly

LIMITS = MediaLimits()


def f32(samples: np.ndarray) -> bytes:
    """Interleaves (channels, n) samples as ffmpeg's f32le output."""
    return np.ascontiguousarray(samples.T, dtype="<f4").tobytes()


def test_audio_modules_import_lightly():
    assert_imports_lightly(
        [
            "provenance_kit.audio.pcm",
            "provenance_kit.audio.fingerprint",
            "provenance_kit.audio.watermark",
        ]
    )


def test_pcm_exposes_only_its_public_api():
    """Helpers start with an underscore (ARCHITECTURE.md, section 8)."""
    defined_here = {
        name
        for name, value in vars(pcm).items()
        if not name.startswith("_") and getattr(value, "__module__", None) == pcm.__name__
    }
    assert defined_here == {"AudioInfo", "PcmAudio", "decode_pcm", "probe_audio"}
    assert set(pcm.__all__) == defined_here | {"MAX_STREAMS", "MIN_SAMPLE_RATE"}
    for helper in (
        "pcm_args",
        "read_pcm",
        "read_samples",
        "ChannelBuffer",
        "sample_cap_bytes",
        "duration_cap_bytes",
        "check_declared_decoded_size",
        "probe_staged",
        "parse_probe_output",
        "validate_probe_report",
        "AUDIO_CONTAINERS",
        "FFMPEG_COMMON",
        "StagedInput",
        "staged_input",
    ):
        assert not hasattr(pcm, helper), helper


def test_decode_pcm_returns_channels_first_float32(rng):
    source = rng.uniform(-1.2, 1.2, size=(2, 480)).astype(np.float32)
    tools = FakeTools(ffprobe=probe_json(), ffmpeg=f32(source))
    audio = decode_pcm(WAV, tools=tools)  # type: ignore[arg-type]
    assert isinstance(audio, PcmAudio)
    assert audio.samples.shape == (2, 480)
    assert audio.samples.dtype == np.float32
    assert audio.samples.flags.c_contiguous
    assert audio.sample_rate == 48000
    assert audio.duration_seconds == pytest.approx(0.01)
    np.testing.assert_allclose(audio.samples, np.clip(source, -1, 1))
    (args,) = tools.args_of("ffmpeg")
    assert args[:4] == ["-nostdin", "-hide_banner", "-loglevel", "error"]
    assert args[args.index("-protocol_whitelist") + 1] == "pipe"
    assert args[args.index("-f") + 1] == "wav"
    assert args[args.index("-ac") + 1] == "2"
    assert args[args.index("-ar") + 1] == "48000"
    assert args[-3:] == ["-c:a", "pcm_f32le", "pipe:1"]


def test_decode_pcm_resamples_and_downmixes_through_ffmpeg():
    tools = FakeTools(ffprobe=probe_json(), ffmpeg=np.zeros(160, "<f4").tobytes())
    audio = decode_pcm(WAV, sample_rate=16000, channels=1, tools=tools)  # type: ignore[arg-type]
    assert audio.samples.shape == (1, 160)
    assert audio.sample_rate == 16000
    (args,) = tools.args_of("ffmpeg")
    assert args[args.index("-ac") + 1] == "1"
    assert args[args.index("-ar") + 1] == "16000"


def test_downmix_matrix_is_normalized():
    # ffmpeg's float output does not normalize its rematrix by default
    # (+3 dB for identical stereo channels); the 16-bit output does.
    for out_channels in (None, 1):
        tools = FakeTools(ffprobe=probe_json(), ffmpeg=b"")
        decode_pcm(WAV, channels=out_channels, tools=tools)  # type: ignore[arg-type]
        (args,) = tools.args_of("ffmpeg")
        assert args[args.index("-rematrix_maxval") + 1] == "1.0"
        assert args.index("-rematrix_maxval") > args.index("-i")  # an output option


def test_nan_samples_are_zeroed():
    tools = FakeTools(
        ffprobe=probe_json(channels=1), ffmpeg=np.array([np.nan, 0.5], "<f4").tobytes()
    )
    audio = decode_pcm(WAV, tools=tools)  # type: ignore[arg-type]
    assert audio.samples.tolist() == [[0.0, 0.5]]


def test_infinities_are_clipped_and_nan_zeroed_across_chunks(monkeypatch):
    monkeypatch.setattr(pcm, "_READ_CHUNK", 20)  # 1 2/3 frames of 3 channels per chunk
    values = np.array([np.nan, np.inf, -np.inf, 0.25, -2.0, 1.5, np.nan] * 9, "<f4")
    tools = FakeTools(ffprobe=probe_json(channels=3), ffmpeg=values.tobytes())
    audio = decode_pcm(WAV, tools=tools)  # type: ignore[arg-type]
    expected = np.nan_to_num(np.clip(values[: 21 * 3], -1, 1), nan=0.0).reshape(-1, 3).T
    np.testing.assert_array_equal(audio.samples, expected)


@pytest.mark.parametrize("channels", [1, 2, 3, 6])
@pytest.mark.parametrize(
    ("declared", "runs"), [("exact", 1), ("short", 3), ("long", 1), ("none", 2)]
)
def test_samples_are_deinterleaved_whatever_the_chunks(rng, monkeypatch, channels, declared, runs):
    # Chunks that cut frames anywhere, a trailing partial frame (dropped),
    # and a declared duration that is right, too short, too long or absent.
    # Too short: the first run stops when the buffer is full, a second one
    # counts the samples and a third decodes them into an exact buffer;
    # absent: count, then decode. Too long: the unused room is released.
    monkeypatch.setattr(pcm, "_READ_CHUNK", 4 * 1021)
    frames = 30_011
    source = rng.uniform(-1.5, 1.5, size=(channels, frames)).astype(np.float32)
    seconds = {"exact": frames / 1000, "short": 3.0, "long": 90.0, "none": None}[declared]
    report = json.loads(probe_json(channels=channels, sample_rate="1000"))
    if seconds is None:
        del report["format"]["duration"]
        del report["streams"][0]["duration"]
    else:
        report["format"]["duration"] = str(seconds)
    output = f32(source) + b"\x01\x02\x03"
    tools = FakeTools(ffprobe=json.dumps(report).encode(), ffmpeg=output)
    audio = decode_pcm(WAV, tools=tools)  # type: ignore[arg-type]
    np.testing.assert_array_equal(audio.samples, np.clip(source, -1, 1))
    assert audio.samples.flags.c_contiguous
    assert audio.samples.flags.writeable
    assert len(tools.args_of("ffmpeg")) == runs
    # At most a quarter of unused room is kept alive (1 % and 1 s here when
    # the duration is right; none after the exact second run, or a release).
    owner = audio.samples if audio.samples.base is None else audio.samples.base
    assert owner.nbytes <= 1.25 * audio.samples.nbytes
    if declared != "exact":
        assert owner.nbytes == audio.samples.nbytes


def test_a_decoder_whose_output_changes_between_runs_is_an_error(rng):
    report = json.loads(probe_json(channels=1))
    del report["format"]["duration"]
    del report["streams"][0]["duration"]
    counted, decoded = np.zeros(4800, "<f4").tobytes(), np.zeros(9600, "<f4").tobytes()
    tools = FakeTools(ffprobe=json.dumps(report).encode(), ffmpeg=[counted, decoded])
    with pytest.raises(MediaDecodeError, match="changed between two runs"):
        decode_pcm(WAV, tools=tools)  # type: ignore[arg-type]


_READ_BUFFERS = 4 * 1024 * 1024  # a read chunk, its copy with a cut frame, masks


def _traced_decode(tools: FakeTools, limits: MediaLimits) -> tuple[PcmAudio, int]:
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        audio = decode_pcm(WAV, tools=tools, limits=limits)  # type: ignore[arg-type]
        return audio, tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize(
    ("declared", "bound"),
    [
        ("exact", 1.05),  # sized from the declared duration (+1 %, +1 s)
        ("short", 1.05),  # the 30 s buffer, then counted and decoded exactly
        ("none", 1.05),  # counted, then decoded exactly
        ("long", (50 * 1.01 + 1) / (2_000_000 / 48000)),  # the declared 50 s
    ],
)
def test_decoding_holds_about_one_copy_of_the_samples(rng, declared, bound):
    """R2-S01: the peak was 3.25x the samples (chunk list, joined bytes, a copy).

    The samples are written channel by channel into one buffer whose size is
    fixed before each run: from the declared duration, or counted by a
    first run when nothing (or too little) is declared. No buffer is ever
    reallocated (NumPy 2.5 traces a reallocation as old plus new size, and
    a growing one can hold both for real).
    """
    frames = 2_000_000  # 16 MB of stereo float32, about 41.7 s at 48 kHz
    source = rng.uniform(-1, 1, size=(2, frames)).astype(np.float32)
    seconds = {"exact": frames / 48000, "short": 30.0, "long": 50.0, "none": None}[declared]
    report = json.loads(probe_json())
    if seconds is None:
        del report["format"]["duration"]
        del report["streams"][0]["duration"]
    else:
        report["format"]["duration"] = str(seconds)
    tools = FakeTools(ffprobe=json.dumps(report).encode(), ffmpeg=f32(source))
    audio, peak = _traced_decode(tools, dataclasses.replace(LIMITS, max_decoded_audio_bytes=10**9))
    assert peak <= bound * source.nbytes + _READ_BUFFERS, peak / source.nbytes
    np.testing.assert_array_equal(audio.samples, source)


def test_decoding_never_holds_more_than_the_limit(rng):
    # Undeclared media just under max_decoded_audio_bytes: counted, then
    # decoded into a buffer of exactly its size.
    source = rng.uniform(-1, 1, size=(2, 2_000_000)).astype(np.float32)
    report = json.loads(probe_json())
    del report["format"]["duration"]
    del report["streams"][0]["duration"]
    tools = FakeTools(ffprobe=json.dumps(report).encode(), ffmpeg=f32(source))
    limit = source.nbytes + 4096
    audio, peak = _traced_decode(tools, dataclasses.replace(LIMITS, max_decoded_audio_bytes=limit))
    assert peak <= limit + _READ_BUFFERS, peak / limit
    assert audio.samples.nbytes == source.nbytes


def test_more_audio_than_the_duration_limit_raises():
    limits = dataclasses.replace(LIMITS, max_duration_seconds=0.01)
    report = probe_json(channels=1, duration=None)
    report_obj = json.loads(report)
    report_obj["format"].pop("duration")
    tools = FakeTools(
        ffprobe=json.dumps(report_obj).encode(), ffmpeg=np.zeros(4800, "<f4").tobytes()
    )
    with pytest.raises(MediaLimitExceededError) as info:
        decode_pcm(WAV, tools=tools, limits=limits)  # type: ignore[arg-type]
    assert info.value.limit == "max_duration_seconds"


def test_caller_truncation_is_not_an_error():
    tools = FakeTools(ffprobe=probe_json(channels=1), ffmpeg=np.zeros(480, "<f4").tobytes())
    audio = decode_pcm(WAV, max_duration_seconds=0.01, tools=tools)  # type: ignore[arg-type]
    assert audio.frames == 480
    (args,) = tools.args_of("ffmpeg")
    assert float(args[args.index("-t") + 1]) == pytest.approx(0.01)


def surround_report(seconds: float) -> bytes:
    """An ffprobe report for 8-channel 192 kHz audio of the given declared duration."""
    report = json.loads(probe_json(channels=8, sample_rate="192000", duration=str(seconds)))
    report["format"]["duration"] = str(seconds)
    return json.dumps(report).encode()


def test_declared_decoded_size_is_refused_before_decoding():
    # 20 s of 8-channel 192 kHz audio is 123 MB of float32: a 16 KB FLAC of
    # silence declares that much. Three hours of it would be 66 GB.
    small = dataclasses.replace(LIMITS, max_decoded_audio_bytes=100_000_000)
    tools = FakeTools(ffprobe=surround_report(20))
    with pytest.raises(MediaLimitExceededError) as info:
        decode_pcm(WAV, tools=tools, limits=small)  # type: ignore[arg-type]
    assert info.value.limit == "max_decoded_audio_bytes"
    assert info.value.actual == 20 * 192_000 * 8 * 4
    assert tools.args_of("ffmpeg") == []  # refused from the probe alone
    tools = FakeTools(ffprobe=surround_report(10_000))
    with pytest.raises(MediaLimitExceededError, match="max_decoded_audio_bytes"):
        decode_pcm(WAV, tools=tools)  # type: ignore[arg-type]  # default limits
    assert tools.args_of("ffmpeg") == []


def test_decoded_size_uses_the_output_layout_and_the_truncation():
    report = surround_report(20)
    small = dataclasses.replace(LIMITS, max_decoded_audio_bytes=100_000_000)
    tools = FakeTools(ffprobe=report, ffmpeg=np.zeros(160, "<f4").tobytes())
    audio = decode_pcm(WAV, sample_rate=16000, channels=1, tools=tools, limits=small)  # type: ignore[arg-type]
    assert audio.samples.shape == (1, 160)
    tools = FakeTools(ffprobe=report, ffmpeg=np.zeros(8 * 192, "<f4").tobytes())
    decode_pcm(WAV, max_duration_seconds=1.0, tools=tools, limits=small)  # type: ignore[arg-type]


def test_undeclared_decoded_size_is_counted_while_decoding():
    report = json.loads(probe_json(channels=2, duration=None))
    report["format"].pop("duration")
    tools = FakeTools(ffprobe=json.dumps(report).encode(), ffmpeg=np.zeros(4096, "<f4").tobytes())
    small = dataclasses.replace(LIMITS, max_decoded_audio_bytes=8000)
    with pytest.raises(MediaLimitExceededError) as info:
        decode_pcm(WAV, tools=tools, limits=small)  # type: ignore[arg-type]
    assert info.value.limit == "max_decoded_audio_bytes"
    assert info.value.maximum == 8000
    # The same output within the bound decodes.
    fits = dataclasses.replace(LIMITS, max_decoded_audio_bytes=4096 * 4)
    tools = FakeTools(ffprobe=json.dumps(report).encode(), ffmpeg=np.zeros(4096, "<f4").tobytes())
    assert decode_pcm(WAV, tools=tools, limits=fits).frames == 2048  # type: ignore[arg-type]


def test_requested_layout_is_checked_against_limits():
    tools = FakeTools(ffprobe=probe_json())
    with pytest.raises(MediaLimitExceededError):
        decode_pcm(WAV, channels=9, tools=tools)  # type: ignore[arg-type]
    with pytest.raises(MediaLimitExceededError):
        decode_pcm(WAV, sample_rate=400_000, tools=tools)  # type: ignore[arg-type]
    assert tools.args_of("ffmpeg") == []


@pytest.mark.parametrize(
    "kwargs",
    [{"sample_rate": 999}, {"channels": 0}, {"max_duration_seconds": 0.0}],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        decode_pcm(WAV, tools=FakeTools(), **kwargs)  # type: ignore[arg-type]


def test_decoder_failure_is_a_decode_error():
    tools = FakeTools(ffprobe=probe_json(), ffmpeg=b"", ffmpeg_returncode=1)
    with pytest.raises(MediaDecodeError):
        decode_pcm(WAV, tools=tools)  # type: ignore[arg-type]


def test_seekable_containers_use_a_private_temporary_file(tmp_path):
    tools = FakeTools(ffprobe=probe_json(), ffmpeg=b"")
    decode_pcm(M4A_MDAT_FIRST, tools=tools, temp_dir=tmp_path)  # type: ignore[arg-type]
    for _, args, kwargs in tools.calls:
        assert args[args.index("-protocol_whitelist") + 1] == "file,pipe"
        assert args[args.index("-f") + 1] == "mov"
        assert kwargs["stdin"] is None
    assert list(tmp_path.iterdir()) == []


def test_non_audio_bytes_are_refused():
    with pytest.raises(UnsupportedMediaError):
        probe_audio(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, tools=FakeTools())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        probe_audio("song.wav")  # type: ignore[arg-type]


# ------------------------------------------------------------ hostile reports


def parse(raw: bytes, limits: MediaLimits = LIMITS):
    return _parse_probe_output(raw, MediaType.WAV, limits)


def test_probe_parses_a_normal_report():
    info = parse(probe_json())
    assert (info.sample_rate, info.channels, info.codec) == (48000, 2, "pcm_s16le")
    assert info.duration_seconds == 1.0
    assert info.audio_streams == 1


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{",
        b"[]",
        b'{"streams": [{"codec_type": "audio", "channels": NaN}]}',
        b"\x80\x81",
        b'{"streams": 5}',
        probe_json(sample_rate="fast"),
        probe_json(channels=0),
        probe_json(channels=True),
        probe_json(index=-1),
    ],
)
def test_unusable_reports_are_decode_errors(raw):
    with pytest.raises(MediaDecodeError):
        parse(raw)


def test_no_audio_stream_is_unsupported():
    raw = json.dumps({"streams": [{"index": 0, "codec_type": "video"}]}).encode()
    with pytest.raises(UnsupportedMediaError):
        parse(raw)
    many = json.dumps({"streams": [{"codec_type": "audio"}] * 100}).encode()
    with pytest.raises(UnsupportedMediaError):
        parse(many)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [({"channels": 64}, "max_audio_channels"), ({"sample_rate": "9999999"}, "max_sample_rate")],
)
def test_declared_layout_is_checked(overrides, field):
    with pytest.raises(MediaLimitExceededError) as info:
        parse(probe_json(**overrides))
    assert info.value.limit == field


def test_declared_duration_is_checked():
    report = json.loads(probe_json())
    report["format"]["duration"] = "1e9"
    with pytest.raises(MediaLimitExceededError):
        parse(json.dumps(report).encode())


def test_bad_durations_are_ignored():
    report = json.loads(probe_json(duration="-5"))
    report["format"]["duration"] = "N/A"
    assert parse(json.dumps(report).encode()).duration_seconds is None


def test_pcm_audio_validation():
    with pytest.raises(TypeError):
        PcmAudio(samples=np.zeros((1, 4), np.float64), sample_rate=8000)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PcmAudio(samples=np.zeros(4, np.float32), sample_rate=8000)
    with pytest.raises(ValueError):
        PcmAudio(samples=np.zeros((1, 4), np.float32), sample_rate=0)
    assert "frames=4" in repr(PcmAudio(samples=np.zeros((1, 4), np.float32), sample_rate=8000))
