"""PCM decoding against the real ffmpeg/ffprobe binaries."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from provenance_kit.audio._ffmpeg import needs_seekable_input
from provenance_kit.audio.pcm import decode_pcm, probe_audio
from provenance_kit.errors import (
    MediaDecodeError,
    MediaLimitExceededError,
    UnsupportedMediaError,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType
from provenance_kit.tools import ExternalTools, Tool
from tests.audio.media import ffmpeg, lavfi, song, to_file
from tests.support import find_tools, require_tool

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tools() -> ExternalTools:
    require_tool(Tool.FFMPEG)
    require_tool(Tool.FFPROBE)
    return find_tools()


@pytest.fixture(scope="module")
def media(tools, tmp_path_factory) -> dict[str, bytes]:
    directory = tmp_path_factory.mktemp("audio")
    source = lavfi(song(0, 44100), 3)
    stereo = [*source, "-ac", "2"]
    out = {
        "wav": ffmpeg(tools, [*stereo, "-f", "wav", "pipe:1"]),
        "flac": ffmpeg(tools, [*stereo, "-c:a", "flac", "-f", "flac", "pipe:1"]),
        "mp3": ffmpeg(
            tools, [*stereo, "-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3", "pipe:1"]
        ),
        "adts": ffmpeg(tools, [*stereo, "-c:a", "aac", "-b:a", "96k", "-f", "adts", "pipe:1"]),
        "ogg": ffmpeg(tools, [*stereo, "-c:a", "libopus", "-b:a", "64k", "-f", "ogg", "pipe:1"]),
        "m4a": to_file(tools, [*stereo, "-c:a", "aac"], directory / "a.m4a"),
    }
    out["mp4_with_video"] = to_file(
        tools,
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=10",
            *source,
            "-ac",
            "2",
            "-shortest",
            "-c:v",
            "libopenh264",
            "-c:a",
            "aac",
        ],
        directory / "v.mp4",
    )
    return out


@pytest.mark.parametrize("name", ["wav", "flac", "mp3", "adts", "ogg", "m4a", "mp4_with_video"])
def test_decodes_every_container(media, tools, name):
    audio = decode_pcm(media[name], tools=tools)
    assert audio.channels == 2
    assert audio.sample_rate in (44100, 48000)  # Opus decodes at 48 kHz
    assert audio.duration_seconds == pytest.approx(3.0, abs=0.06)
    assert audio.samples.dtype == np.float32
    assert np.abs(audio.samples).max() <= 1.0
    assert np.abs(audio.samples).max() > 0.1


def test_m4a_with_trailing_moov_uses_a_temporary_file(media, tools, tmp_path):
    assert needs_seekable_input(media["m4a"], MediaType.M4A)
    decode_pcm(media["m4a"], tools=tools, temp_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_resampling_and_downmixing(media, tools):
    reference = decode_pcm(media["wav"], tools=tools)
    mono = decode_pcm(media["wav"], sample_rate=16000, channels=1, tools=tools)
    assert mono.channels == 1
    assert mono.sample_rate == 16000
    assert mono.frames == pytest.approx(48000, abs=2)
    # ffmpeg's default matrix sums the two channels at -3 dB each, so a
    # stereo file with identical channels comes out sqrt(2) louder in mono.
    average = np.sqrt(np.mean(reference.samples.mean(axis=0) ** 2))
    assert np.sqrt(np.mean(mono.samples**2)) == pytest.approx(average * np.sqrt(2), rel=0.05)


def test_probe_reports_the_audio_stream(media, tools):
    info = probe_audio(media["mp4_with_video"], tools=tools)
    assert (info.codec, info.channels, info.audio_streams, info.video_streams) == ("aac", 2, 1, 1)
    assert info.duration_seconds == pytest.approx(3.0, abs=0.1)


def test_truncation_and_limits(media, tools):
    short = decode_pcm(media["flac"], max_duration_seconds=1.0, tools=tools)
    assert short.duration_seconds == pytest.approx(1.0, abs=0.01)
    limits = dataclasses.replace(MediaLimits(), max_duration_seconds=1.0)
    with pytest.raises(MediaLimitExceededError):  # declared duration
        decode_pcm(media["wav"], tools=tools, limits=limits)
    # FLAC written to a pipe declares no duration: the decoder counts samples.
    assert probe_audio(media["flac"], tools=tools).duration_seconds is None
    with pytest.raises(MediaLimitExceededError):
        decode_pcm(media["flac"], tools=tools, limits=limits)
    with pytest.raises(MediaLimitExceededError):
        decode_pcm(
            media["wav"],
            tools=tools,
            limits=dataclasses.replace(MediaLimits(), max_audio_channels=1),
        )


def test_corrupt_and_foreign_media(media, tools, rng):
    with pytest.raises(MediaDecodeError):
        decode_pcm(media["m4a"][: len(media["m4a"]) // 2], tools=tools)
    with pytest.raises(UnsupportedMediaError):
        decode_pcm(b"#EXTM3U\nhttp://example.invalid/a.aac\n", tools=tools)
    video_only = ffmpeg(
        tools,
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=10",
            "-t",
            "1",
            "-c:v",
            "mpeg4",
            "-f",
            "matroska",
            "pipe:1",
        ],
    )
    with pytest.raises(UnsupportedMediaError):
        decode_pcm(video_only, tools=tools)
