"""Chromaprint fingerprints with the real ffmpeg and fpcalc binaries.

Measured on the reference machine (2026-11-05; 60 s test songs at 44.1 kHz
stereo, fpcalc 1.6.2, default threshold 0.20):

=================================  ============  ==========================
Variant                            Offset found  Bit error rate
=================================  ============  ==========================
MP3 64 kb/s                        0 s           0.039 to 0.040
AAC 32 kb/s, mono, 22.05 kHz       0 s           0.065 to 0.070
Opus 24 kb/s, trim 17.3 s to 37.3  17.33 s       0.069 to 0.079
AAC 96 kb/s, trim 41.5 s to 49.5   41.48 s       0.002 to 0.003
Volume x0.3, low-pass 4 kHz, MP3   0 s           0.042 to 0.044
Unrelated song (similar synthesis) none          0.316 to 0.365 (12 pairs)
Unrelated pink noise               none          0.435
=================================  ============  ==========================

Run with ``-s`` to print the numbers.
"""

from __future__ import annotations

import pytest

from provenance_kit.audio.fingerprint import (
    DEFAULT_MAX_BIT_ERROR_RATE,
    AudioFingerprint,
    compare_fingerprints,
    fingerprint_audio,
    fingerprint_pcm,
)
from provenance_kit.audio.pcm import decode_pcm
from provenance_kit.tools import ExternalTools, Tool
from tests.audio.media import ffmpeg, lavfi, song
from tests.support import find_tools, require_tool

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tools() -> ExternalTools:
    for tool in (Tool.FFMPEG, Tool.FFPROBE, Tool.FPCALC):
        require_tool(tool)
    return find_tools()


@pytest.fixture(scope="module")
def songs(tools) -> dict[int, bytes]:
    return {
        k: ffmpeg(tools, [*lavfi(song(k), 60), "-ac", "2", "-c:a", "flac", "-f", "flac", "pipe:1"])
        for k in range(4)
    }


@pytest.fixture(scope="module")
def fingerprints(songs, tools) -> dict[int, AudioFingerprint]:
    return {k: fingerprint_audio(data, tools=tools) for k, data in songs.items()}


VARIANTS = {
    "mp3-64k": (["-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"], 0.0),
    "aac-32k-mono-22k": (
        ["-ac", "1", "-ar", "22050", "-c:a", "aac", "-b:a", "32k", "-f", "adts"],
        0.0,
    ),
    "opus-24k-trim-17.3": (
        ["-ss", "17.3", "-t", "20", "-c:a", "libopus", "-b:a", "24k", "-f", "ogg"],
        17.3,
    ),
    "aac-trim-41.5-49.5": (
        ["-ss", "41.5", "-t", "8", "-c:a", "aac", "-b:a", "96k", "-f", "adts"],
        41.5,
    ),
    "volume-lowpass-mp3": (
        ["-af", "volume=0.3,lowpass=f=4000", "-c:a", "libmp3lame", "-b:a", "96k", "-f", "mp3"],
        0.0,
    ),
}


def test_duration_comes_from_the_pcm(fingerprints):
    fp = fingerprints[0]
    assert fp.duration_seconds == pytest.approx(60.0, abs=0.01)  # fpcalc says 0.0 on stdin
    assert len(fp) == pytest.approx(60 / 0.1238, abs=25)
    assert AudioFingerprint.from_bytes(fp.to_bytes()) == fp


@pytest.mark.parametrize("name", sorted(VARIANTS))
@pytest.mark.parametrize("k", [0, 1])
def test_derived_audio_matches_at_the_right_offset(name, k, songs, fingerprints, tools):
    args, offset = VARIANTS[name]
    derived = ffmpeg(tools, ["-i", "pipe:0", *args, "pipe:1"], stdin=songs[k])
    match = compare_fingerprints(fingerprint_audio(derived, tools=tools), fingerprints[k])
    print(f"\nsong {k} {name}: {match}")
    assert match.is_match
    assert match.bit_error_rate < 0.1
    assert match.offset_seconds == pytest.approx(offset, abs=0.13)  # one sub-fingerprint


def test_unrelated_audio_does_not_match(songs, fingerprints, tools):
    rates = []
    for a in range(4):
        clip = ffmpeg(
            tools,
            [
                "-i",
                "pipe:0",
                "-ss",
                "10",
                "-t",
                "10",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-f",
                "adts",
                "pipe:1",
            ],
            stdin=songs[a],
        )
        query = fingerprint_audio(clip, tools=tools)
        for b in range(4):
            if a != b:
                match = compare_fingerprints(query, fingerprints[b])
                rates.append(match.bit_error_rate)
                assert not match.is_match
    noise = ffmpeg(
        tools,
        [
            *lavfi("anoisesrc=color=pink:seed=9:sample_rate=44100:amplitude=0.3", 30),
            "-c:a",
            "flac",
            "-f",
            "flac",
            "pipe:1",
        ],
    )
    noise_rate = compare_fingerprints(fingerprint_audio(noise, tools=tools), fingerprints[0])
    print(
        f"\nunrelated songs: {min(rates):.3f} to {max(rates):.3f}; "
        f"noise {noise_rate.bit_error_rate:.3f}"
    )
    assert min(rates) > DEFAULT_MAX_BIT_ERROR_RATE + 0.05
    assert not noise_rate.is_match


def test_decoded_pcm_gives_the_same_fingerprint(songs, fingerprints, tools):
    audio = decode_pcm(songs[0], tools=tools)
    match = compare_fingerprints(fingerprint_pcm(audio, tools=tools), fingerprints[0])
    assert match.offset_items == 0
    assert match.bit_error_rate < 0.01


def test_short_audio_gives_an_empty_fingerprint(tools):
    clip = ffmpeg(tools, [*lavfi(song(0), 1.5), "-f", "wav", "pipe:1"])
    fp = fingerprint_audio(clip, tools=tools)
    assert fp.values == ()
    assert fp.duration_seconds == pytest.approx(1.5, abs=0.01)
