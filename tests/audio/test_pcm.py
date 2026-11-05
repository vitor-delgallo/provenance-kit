"""Unit tests of PCM decoding and audio probing with fake tools."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from provenance_kit.audio.pcm import PcmAudio, decode_pcm, parse_probe_output, probe_audio
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


def test_nan_samples_are_zeroed():
    tools = FakeTools(
        ffprobe=probe_json(channels=1), ffmpeg=np.array([np.nan, 0.5], "<f4").tobytes()
    )
    audio = decode_pcm(WAV, tools=tools)  # type: ignore[arg-type]
    assert audio.samples.tolist() == [[0.0, 0.5]]


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
    return parse_probe_output(raw, MediaType.WAV, limits)


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
