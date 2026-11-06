"""Chromaprint fingerprints with the real ffmpeg and fpcalc binaries.

Measured on the reference machine (2026-11-05; 60 s test songs at 44.1 kHz
stereo, fpcalc 1.6.2, default thresholds: 0.11 up to 5 s of compared span,
0.20 from 20 s; the full study, 13 degradations x 6 lengths, is in
docs/audio.md):

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

import numpy as np
import pytest

from provenance_kit.audio.fingerprint import (
    DEFAULT_MAX_BIT_ERROR_RATE,
    ITEM_SECONDS,
    MIN_COMPARED_ITEMS,
    SILENCE_VALUE,
    AudioFingerprint,
    compare_fingerprints,
    fingerprint_audio,
    fingerprint_pcm,
    overlap_threshold,
)
from provenance_kit.audio.pcm import decode_pcm
from provenance_kit.tools import ExternalTools, Tool
from tests.audio.media import ffmpeg, lavfi, melody, music, song, wav
from tests.support import SEED, find_tools, require_tool

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


# ------------------------------------------------------------ length-aware threshold


@pytest.fixture(scope="module")
def long_songs(tools) -> dict[int, AudioFingerprint]:
    """The suite's songs, 120 s each: after about 20 s they sweep through similar ranges."""
    return {
        k: fingerprint_audio(
            ffmpeg(tools, [*lavfi(song(k), 120), "-ac", "1", "-f", "wav", "pipe:1"]), tools=tools
        )
        for k in range(4)
    }


def test_short_windows_of_songs_that_sound_alike_do_not_match(long_songs):
    """Windows of 5, 8 and 10 s of span from one song against another whole song.

    With a flat threshold of 0.20, 36, 21 and 12 % of such windows matched
    (1,000 windows each, eight songs); with the length-aware threshold, 0.2,
    0.0 and 0.0 %.
    """
    rng = np.random.default_rng(SEED)
    matched = []
    for seconds in (5, 8, 10):
        items = round(seconds / ITEM_SECONDS)
        for _ in range(40):
            a, b = (int(k) for k in rng.choice(4, 2, replace=False))
            values = long_songs[a].values
            start = int(rng.integers(0, len(values) - items))
            window = AudioFingerprint(
                values=values[start : start + items], duration_seconds=items * ITEM_SECONDS
            )
            match = compare_fingerprints(window, long_songs[b])
            if match.is_match:
                matched.append((seconds, round(match.bit_error_rate, 3)))
    print(f"\nwindows of other songs reported as matches: {matched} of 120")
    assert len(matched) <= 1


@pytest.fixture(scope="module")
def piece(tools) -> tuple[bytes, AudioFingerprint]:
    """60 s of bass-heavy generated music (stereo WAV) and its fingerprint."""
    samples = melody(60, 44100, 400)  # one of the measured pieces
    data = wav(np.vstack([samples, samples]), 44100)
    return data, fingerprint_audio(data, tools=tools)


DEGRADED = {
    "mp3-64k": ["-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"],
    "aac-64k": ["-c:a", "aac", "-b:a", "64k", "-f", "adts"],
    "opus-32k": ["-c:a", "libopus", "-b:a", "32k", "-f", "ogg"],
    "22.05 kHz": ["-ar", "22050", "-c:a", "flac", "-f", "flac"],
    "treble +6 dB": ["-af", "treble=g=6", "-c:a", "flac", "-f", "flac"],
    "volume -6 dB": ["-af", "volume=-6dB", "-c:a", "flac", "-f", "flac"],
    "mono": ["-ac", "1", "-c:a", "flac", "-f", "flac"],
}


@pytest.mark.parametrize("name", sorted(DEGRADED))
def test_degraded_clips_still_match(piece, tools, name):
    data, reference = piece
    for start, seconds in ((12.0, 8), (31.5, 15)):
        cut = ["-ss", str(start), "-t", str(seconds)]
        clip = ffmpeg(tools, ["-i", "pipe:0", *cut, *DEGRADED[name], "pipe:1"], stdin=data)
        match = compare_fingerprints(fingerprint_audio(clip, tools=tools), reference)
        print(
            f"\n{name}, {seconds} s clip: BER {match.bit_error_rate:.3f} "
            f"(threshold {match.bit_error_threshold:.3f})"
        )
        assert match.is_match
        assert match.offset_seconds == pytest.approx(start, abs=0.25)


def test_long_overlaps_keep_the_room_that_equalization_needs(piece, tools):
    """A +6 dB bass shelf on bass-heavy music: above the short threshold, below the long one."""
    data, reference = piece
    shelved = ffmpeg(
        tools,
        ["-i", "pipe:0", "-af", "bass=g=6", "-c:a", "flac", "-f", "flac", "pipe:1"],
        stdin=data,
    )
    match = compare_fingerprints(fingerprint_audio(shelved, tools=tools), reference)
    print(f"\nbass +6 dB, whole piece: {match}")
    assert match.is_match
    assert match.bit_error_threshold == pytest.approx(DEFAULT_MAX_BIT_ERROR_RATE)
    assert match.bit_error_rate > overlap_threshold(MIN_COMPARED_ITEMS)


# ------------------------------------------------------------ low information


def test_digital_silence_has_one_code_and_is_never_compared(tools):
    """An 8 s silent intro must not match an unrelated track's 8 s silent outro."""
    rate = 44100
    gap = np.zeros(8 * rate)
    upload = fingerprint_audio(wav(np.concatenate([gap, music(30, rate, 11)]), rate), tools=tools)
    original = fingerprint_audio(wav(np.concatenate([music(45, rate, 22), gap]), rate), tools=tools)
    assert SILENCE_VALUE in upload.values
    assert SILENCE_VALUE in original.values
    match = compare_fingerprints(upload, original)
    print(f"\nsilent gaps: {match}")
    assert not match.is_match  # was a match at bit error rate 0.000
    assert match.bit_error_rate > 0.3


@pytest.mark.parametrize(
    "source",
    [
        "anoisesrc=color=white:seed={seed}:sample_rate=44100:amplitude=0.3",
        "anoisesrc=color=pink:seed={seed}:sample_rate=44100:amplitude=0.3",
        "anoisesrc=color=white:seed={seed}:sample_rate=44100:amplitude=0.001",  # -60 dBFS floor
    ],
)
def test_independent_noise_recordings_do_not_match(tools, source):
    first, second = (
        fingerprint_audio(
            ffmpeg(tools, [*lavfi(source.format(seed=seed), 30), "-f", "wav", "pipe:1"]),
            tools=tools,
        )
        for seed in (1, 2)
    )
    match = compare_fingerprints(first, second)
    ratio = match.bit_error_rate / match.chance_bit_error_rate
    print(f"\n{source}: BER {match.bit_error_rate:.3f}, chance {match.chance_bit_error_rate:.3f}")
    assert not match.is_match
    assert ratio > 0.75  # measured 0.82 to 0.94: agreement at chance level


def _noise(kind: str, seconds: float, rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * rate))
    if kind == "blue":
        x = np.diff(x, prepend=0.0)
    return 0.2 * x / np.max(np.abs(x))


def _drone(kind: str, seconds: float, rate: int, seed: int) -> np.ndarray:
    """A steady drone with a slow tremolo: 50 Hz mains hum or a 110 Hz sawtooth pad."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    tremolo = 1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t + rng.uniform(0, 6))
    if kind == "hum":
        partials = ((50 * k, a) for k, a in ((1, 1.0), (2, 0.5), (3, 0.4), (5, 0.2)))
    else:
        partials = ((110 * k, 1 / k) for k in range(1, 30))
    s = sum(a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f, a in partials)
    s = s / np.max(np.abs(s))
    return 0.4 * tremolo * s + 0.01 * rng.standard_normal(t.size)


@pytest.mark.parametrize(
    ("kind", "make"),
    [("white", _noise), ("blue", _noise), ("hum", _drone), ("saw", _drone)],
)
def test_low_information_audio_followed_by_silence_or_music_does_not_match(tools, kind, make):
    """R2-M01: the chance error rate is taken over the compared pairs only.

    Two independent 20 s recordings, the query followed by 40 s of digital
    silence and the reference by 40 s of unrelated music. The music faces the
    silence and is never compared, but it used to count in the reference's
    bit statistics: the chance rate rose to 0.42 to 0.46 and independent
    white and blue noise matched at BER 0.18 to 0.19, as did hum, sawtooth
    and organ drones (12 of 12 pairs).
    """
    rate = 22050
    first, second = make(kind, 20, rate, 1), make(kind, 20, rate, 2)
    alone = compare_fingerprints(
        fingerprint_audio(wav(first, rate), tools=tools),
        fingerprint_audio(wav(second, rate), tools=tools),
    )
    query = fingerprint_audio(wav(np.concatenate([first, np.zeros(40 * rate)]), rate), tools=tools)
    reference = fingerprint_audio(
        wav(np.concatenate([second, music(40, rate, 70)]), rate), tools=tools
    )
    match = compare_fingerprints(query, reference)
    print(
        f"\n{kind}: alone BER {alone.bit_error_rate:.3f} / chance "
        f"{alone.chance_bit_error_rate:.3f}; followed by silence / music: BER "
        f"{match.bit_error_rate:.3f} / chance {match.chance_bit_error_rate:.3f}"
    )
    assert not alone.is_match
    assert not match.is_match
    if make is _noise:
        # A copy of the noise behind the same asymmetry is still found. (A copy
        # of a drone is not: its codes agree with any drone by chance, which
        # is the point of the chance test.)
        copy = fingerprint_audio(
            wav(np.concatenate([first, music(40, rate, 70)]), rate), tools=tools
        )
        assert compare_fingerprints(query, copy).is_match


def test_short_unrelated_clips_never_match_a_long_reference(tools):
    """Clips of 2.7 to 3.6 s give 1 to 8 sub-fingerprints: below MIN_COMPARED_ITEMS."""
    rate = 11025
    reference = fingerprint_audio(wav(music(300, rate, 1000), rate), tools=tools)
    for seed in range(5):
        for seconds in (2.8, 3.0, 3.5):
            clip = fingerprint_audio(wav(music(seconds, rate, seed), rate), tools=tools)
            assert 0 < len(clip) < MIN_COMPARED_ITEMS
            match = compare_fingerprints(clip, reference)
            assert not match.is_match
            assert match.overlap_items == 0


def test_clips_from_about_5_s_are_found(songs, fingerprints, tools):
    for seconds, found in ((4.0, False), (5.0, True), (6.0, True)):
        cut = ["-ss", "21", "-t", str(seconds)]
        clip = ffmpeg(
            tools, ["-i", "pipe:0", *cut, "-c:a", "aac", "-f", "adts", "pipe:1"], stdin=songs[0]
        )
        match = compare_fingerprints(fingerprint_audio(clip, tools=tools), fingerprints[0])
        print(f"\n{seconds} s clip: {match}")
        assert match.is_match is found
        if found:
            assert match.offset_seconds == pytest.approx(21.0, abs=0.13)


def test_an_upload_that_starts_with_the_end_of_the_original_is_found(tools):
    """The genuine 7 s overlap wins although a long unrelated overlap has a higher z-score."""
    rate = 44100
    original_audio = ffmpeg(tools, [*lavfi(song(1, rate), 90), "-ac", "1", "-f", "wav", "pipe:1"])
    original = decode_pcm(original_audio, tools=tools).samples[0]
    other = decode_pcm(
        ffmpeg(tools, [*lavfi(song(2, rate), 120), "-ac", "1", "-f", "wav", "pipe:1"]),
        tools=tools,
    ).samples[0]
    upload = np.concatenate([original[-10 * rate :], other])
    match = compare_fingerprints(
        fingerprint_audio(wav(upload, rate), tools=tools),
        fingerprint_audio(original_audio, tools=tools),
    )
    print(f"\nupload starting with the original's end: {match}")
    assert match.is_match
    assert match.bit_error_rate < 0.05
    assert match.offset_seconds == pytest.approx(80.0, abs=0.5)
