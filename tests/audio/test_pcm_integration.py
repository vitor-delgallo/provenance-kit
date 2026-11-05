"""PCM decoding against the real ffmpeg/ffprobe binaries."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from provenance_kit._ffmpeg import needs_seekable_input
from provenance_kit.audio.pcm import decode_pcm, probe_audio
from provenance_kit.errors import (
    MediaDecodeError,
    MediaLimitExceededError,
    UnsupportedMediaError,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType
from provenance_kit.tools import ExternalTools, Tool
from tests.audio.media import ffmpeg, lavfi, song, to_file, wav
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


def test_flac_behind_an_id3v2_tag_decodes_as_flac(media, tools):
    # The layout of a C2PA-signed FLAC (c2pa-rs puts the manifest in an ID3v2
    # tag in front of the stream); here the tag holds 8 KiB of padding.
    padding = 8192
    size = bytes((padding >> shift) & 0x7F for shift in (21, 14, 7, 0))
    tagged = b"ID3\x04\x00\x00" + size + b"\x00" * padding + media["flac"]
    info = probe_audio(tagged, tools=tools)
    assert info.container is MediaType.FLAC
    assert info.codec == "flac"
    audio = decode_pcm(tagged, tools=tools)
    plain = decode_pcm(media["flac"], tools=tools)
    assert np.array_equal(audio.samples, plain.samples)


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
    # The normalized matrix makes stereo to mono the mean of the channels
    # (ffmpeg's float default would be sqrt(2) louder for identical channels).
    average = np.sqrt(np.mean(reference.samples.mean(axis=0) ** 2))
    assert np.sqrt(np.mean(mono.samples**2)) == pytest.approx(average, rel=0.05)


def test_downmix_of_loud_identical_channels_is_not_clipped(tools):
    rate = 44100
    sine = 0.9 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
    stereo = wav(np.vstack([sine, sine]), rate)
    mono = decode_pcm(stereo, channels=1, tools=tools).samples[0]
    assert np.abs(mono).max() == pytest.approx(0.9, abs=0.002)
    assert np.mean(np.abs(mono) >= 0.9999) == 0.0  # was 42 % clipped at +3 dB
    np.testing.assert_allclose(mono, sine, atol=1e-3)
    # 5.1 to stereo is normalized too (ffmpeg's float default peaked at 2.17).
    six = wav(np.vstack([sine, sine, sine, 0 * sine, sine, sine]), rate)
    front = decode_pcm(six, channels=2, tools=tools).samples
    assert np.abs(front).max() <= 0.9 + 0.002


def test_decoded_size_is_bounded_for_compressed_silence(tools, tmp_path):
    # 4 s of 8-channel 192 kHz silence: a few KB of FLAC, 24.6 MB of float32.
    silence = ["-f", "lavfi", "-i", "anullsrc=r=192000:cl=7.1", "-t", "4", "-c:a", "flac"]
    declared = to_file(tools, silence, tmp_path / "silence.flac")  # STREAMINFO has the length
    piped = ffmpeg(tools, [*silence, "-f", "flac", "pipe:1"])  # no declared duration
    assert len(declared) < 64 * 1024
    assert probe_audio(declared, tools=tools).duration_seconds == pytest.approx(4.0)
    assert probe_audio(piped, tools=tools).duration_seconds is None
    limits = dataclasses.replace(MediaLimits(), max_decoded_audio_bytes=10_000_000)
    for data in (declared, piped):
        with pytest.raises(MediaLimitExceededError) as info:
            decode_pcm(data, tools=tools, limits=limits)
        assert info.value.limit == "max_decoded_audio_bytes"
    # The same file fits once downmixed and resampled (4 s x 16 kHz x 1 x 4 bytes).
    mono = decode_pcm(declared, sample_rate=16000, channels=1, tools=tools, limits=limits)
    assert mono.frames == pytest.approx(4 * 16000, abs=2)


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
