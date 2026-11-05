"""Unit tests of Chromaprint fingerprints with fake tools and synthetic values."""

from __future__ import annotations

import dataclasses
import json
import struct

import numpy as np
import pytest

from provenance_kit.audio.fingerprint import (
    CHROMAPRINT_SAMPLE_RATE,
    FORMAT_VERSION,
    ITEM_SECONDS,
    AudioFingerprint,
    compare_fingerprints,
    fingerprint_audio,
    fingerprint_pcm,
)
from provenance_kit.audio.pcm import PcmAudio
from provenance_kit.errors import MediaDecodeError, MediaLimitExceededError
from provenance_kit.limits import MediaLimits
from tests.audio.fakes import WAV, FakeTools, probe_json


def random_fp(rng: np.random.Generator, count: int) -> AudioFingerprint:
    values = tuple(int(v) for v in rng.integers(0, 2**32, size=count, dtype=np.uint64))
    return AudioFingerprint(values=values, duration_seconds=count * ITEM_SECONDS)


def noisy_clip(
    fp: AudioFingerprint, start: int, stop: int, rng, flip_rate: float
) -> AudioFingerprint:
    values = []
    for value in fp.values[start:stop]:
        mask = 0
        for bit in range(32):
            if rng.random() < flip_rate:
                mask |= 1 << bit
        values.append(value ^ mask)
    return AudioFingerprint(values=tuple(values), duration_seconds=(stop - start) * ITEM_SECONDS)


# ------------------------------------------------------------ serialization


def test_serialization_round_trip(rng):
    fp = random_fp(rng, 300)
    data = fp.to_bytes()
    assert data[:4] == b"PKAF"
    assert data[4] == FORMAT_VERSION
    assert data[5] == 2
    assert len(data) == 20 + 4 * 300
    assert AudioFingerprint.from_bytes(data) == fp


def header(*, magic=b"PKAF", version=1, algorithm=2, reserved=0, duration=1.0, count=0) -> bytes:
    return struct.pack(">4sBBHdI", magic, version, algorithm, reserved, duration, count)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        header()[:-1],
        header(magic=b"PKVF"),
        header(version=9),
        header(reserved=7),
        header(duration=float("inf")),
        header(duration=-1.0),
        header(count=2),
        header(count=1) + b"\x00" * 5,
        header(count=2**30),
    ],
)
def test_from_bytes_rejects_invalid_data(data):
    with pytest.raises(ValueError):
        AudioFingerprint.from_bytes(data)


def test_fingerprint_validation():
    with pytest.raises(ValueError):
        AudioFingerprint(values=(2**32,), duration_seconds=1.0)
    with pytest.raises(ValueError):
        AudioFingerprint(values=(-1,), duration_seconds=1.0)
    with pytest.raises(ValueError):
        AudioFingerprint(values=(), duration_seconds=float("nan"))
    with pytest.raises(TypeError):
        AudioFingerprint(values=(1.5,), duration_seconds=1.0)  # type: ignore[arg-type]


# ------------------------------------------------------------ comparison


def test_identical_fingerprints_match_exactly(rng):
    fp = random_fp(rng, 200)
    match = compare_fingerprints(fp, fp)
    assert match.is_match
    assert match.bit_error_rate == 0.0
    assert match.offset_items == 0
    assert match.overlap_items == 200
    assert match.query_coverage == 1.0


@pytest.mark.parametrize("start", [0, 13, 250])
def test_a_noisy_clip_is_found_at_its_offset(rng, start):
    reference = random_fp(rng, 400)
    clip = noisy_clip(reference, start, start + 80, rng, flip_rate=0.1)
    match = compare_fingerprints(clip, reference)
    assert match.is_match
    assert match.offset_items == start
    assert match.offset_seconds == pytest.approx(start * ITEM_SECONDS)
    assert 0.05 < match.bit_error_rate < 0.15
    assert match.overlap_items == 80


def test_the_reference_may_be_the_shorter_one(rng):
    long = random_fp(rng, 300)
    short = noisy_clip(long, 100, 150, rng, flip_rate=0.05)
    match = compare_fingerprints(long, short)
    assert match.is_match
    assert match.offset_items == -100
    assert match.overlap_items == 50


def test_unrelated_fingerprints_do_not_match(rng):
    match = compare_fingerprints(random_fp(rng, 80), random_fp(rng, 400))
    assert not match.is_match
    assert match.bit_error_rate > 0.4


def test_long_overlaps_beat_short_lucky_ones(rng):
    # A periodic reference: every offset that is a multiple of the period
    # matches; the full overlap must win over a shorter one.
    period = random_fp(rng, 50).values
    reference = AudioFingerprint(values=period * 4, duration_seconds=200 * ITEM_SECONDS)
    query = AudioFingerprint(values=period * 4, duration_seconds=200 * ITEM_SECONDS)
    match = compare_fingerprints(query, reference)
    assert match.offset_items == 0
    assert match.overlap_items == 200


def test_min_overlap_and_max_offset(rng):
    reference = random_fp(rng, 300)
    clip = noisy_clip(reference, 200, 300, rng, flip_rate=0.0)
    assert compare_fingerprints(clip, reference, max_offset_seconds=10).bit_error_rate > 0.3
    assert compare_fingerprints(clip, reference).offset_items == 200
    tiny = noisy_clip(reference, 0, 3, rng, flip_rate=0.0)
    assert compare_fingerprints(tiny, reference).overlap_items == 3  # whole query


def test_empty_fingerprints_never_match(rng):
    empty = AudioFingerprint(values=(), duration_seconds=1.0)
    match = compare_fingerprints(empty, random_fp(rng, 10))
    assert not match.is_match
    assert match.overlap_items == 0
    assert match.bit_error_rate == 1.0


def test_comparison_validates_arguments(rng):
    fp = random_fp(rng, 10)
    with pytest.raises(ValueError):
        compare_fingerprints(fp, dataclasses.replace(fp, algorithm=1))
    with pytest.raises(ValueError):
        compare_fingerprints(fp, fp, max_bit_error_rate=2.0)
    with pytest.raises(ValueError):
        compare_fingerprints(fp, fp, min_overlap_seconds=-1)
    with pytest.raises(TypeError):
        compare_fingerprints(fp, fp.values)  # type: ignore[arg-type]


# ------------------------------------------------------------ running fpcalc


def fpcalc_json(values: list[object]) -> bytes:
    return json.dumps({"duration": 0.0, "fingerprint": values}).encode()


def test_fingerprint_audio_streams_pcm_into_fpcalc():
    pcm = np.zeros(CHROMAPRINT_SAMPLE_RATE * 3, "<i2").tobytes()
    tools = FakeTools(
        ffprobe=probe_json(), ffmpeg=pcm, fpcalc=lambda data: (0, fpcalc_json([1, 2, 4294967295]))
    )
    fp = fingerprint_audio(WAV, tools=tools)  # type: ignore[arg-type]
    assert fp.values == (1, 2, 4294967295)
    assert fp.duration_seconds == pytest.approx(3.0)  # from the PCM, not from fpcalc
    assert tools.fpcalc_input == pcm
    (ffmpeg,) = tools.args_of("ffmpeg")
    assert ffmpeg[ffmpeg.index("-ac") + 1] == "1"
    assert ffmpeg[ffmpeg.index("-ar") + 1] == str(CHROMAPRINT_SAMPLE_RATE)
    assert ffmpeg[ffmpeg.index("-f", ffmpeg.index("-i")) + 1] == "s16le"
    (fpcalc,) = tools.args_of("fpcalc")
    assert fpcalc[:2] == ["-raw", "-json"]
    assert fpcalc[-1] == "-"
    assert fpcalc[fpcalc.index("-length") + 1] == "0"


def test_short_audio_gives_an_empty_fingerprint():
    tools = FakeTools(
        ffprobe=probe_json(),
        ffmpeg=np.zeros(1000, "<i2").tobytes(),
        fpcalc=lambda data: (2, b""),
    )
    fp = fingerprint_audio(WAV, tools=tools)  # type: ignore[arg-type]
    assert fp.values == ()
    assert fp.duration_seconds == pytest.approx(1000 / CHROMAPRINT_SAMPLE_RATE)


def test_fpcalc_failure_on_long_audio_is_a_decode_error():
    tools = FakeTools(
        ffprobe=probe_json(),
        ffmpeg=np.zeros(CHROMAPRINT_SAMPLE_RATE * 5, "<i2").tobytes(),
        fpcalc=lambda data: (3, b""),
    )
    with pytest.raises(MediaDecodeError):
        fingerprint_audio(WAV, tools=tools)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "output",
    [
        b"not json",
        fpcalc_json([-1]),
        fpcalc_json([2**32]),
        fpcalc_json([1.5]),
        fpcalc_json([True]),
        fpcalc_json(["1"]),
        json.dumps({"fingerprint": "AQAAA"}).encode(),
        b'{"fingerprint": [NaN]}',
    ],
)
def test_hostile_fpcalc_output_is_a_decode_error(output):
    tools = FakeTools(
        ffprobe=probe_json(),
        ffmpeg=np.zeros(CHROMAPRINT_SAMPLE_RATE * 3, "<i2").tobytes(),
        fpcalc=lambda data: (0, output),
    )
    with pytest.raises(MediaDecodeError):
        fingerprint_audio(WAV, tools=tools)  # type: ignore[arg-type]


def test_pcm_beyond_the_duration_limit_raises():
    limits = dataclasses.replace(MediaLimits(), max_duration_seconds=1.0)
    report = json.loads(probe_json(duration=None))
    report["format"].pop("duration")
    tools = FakeTools(
        ffprobe=json.dumps(report).encode(),
        ffmpeg=np.zeros(CHROMAPRINT_SAMPLE_RATE * 3, "<i2").tobytes(),
        fpcalc=lambda data: (0, fpcalc_json([1])),
    )
    with pytest.raises(MediaLimitExceededError):
        fingerprint_audio(WAV, tools=tools, limits=limits)  # type: ignore[arg-type]


def test_decoder_failure_is_a_decode_error():
    tools = FakeTools(
        ffprobe=probe_json(),
        ffmpeg=b"",
        fpcalc=lambda data: (2, b""),
        ffmpeg_returncode=1,
    )
    with pytest.raises(MediaDecodeError):
        fingerprint_audio(WAV, tools=tools)  # type: ignore[arg-type]


def test_fingerprint_pcm_mixes_down_to_16_bit_mono():
    samples = np.stack([np.full(48000, 0.5, np.float32), np.full(48000, -0.25, np.float32)])
    tools = FakeTools(fpcalc=lambda data: (0, fpcalc_json([7])))
    fp = fingerprint_pcm(PcmAudio(samples=samples, sample_rate=48000), tools=tools)  # type: ignore[arg-type]
    assert fp.values == (7,)
    assert fp.duration_seconds == pytest.approx(1.0)
    pcm = np.frombuffer(tools.fpcalc_input, "<i2")
    assert pcm.size == 48000
    assert int(pcm[0]) == round(0.125 * 32767)
    (args,) = tools.args_of("fpcalc")
    assert args[args.index("-rate") + 1] == "48000"
