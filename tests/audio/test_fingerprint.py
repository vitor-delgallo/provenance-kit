"""Unit tests of Chromaprint fingerprints with fake tools and synthetic values."""

from __future__ import annotations

import dataclasses
import json
import math
import struct

import numpy as np
import pytest

from provenance_kit.audio import fingerprint
from provenance_kit.audio.fingerprint import (
    CHROMAPRINT_SAMPLE_RATE,
    DEFAULT_MAX_BIT_ERROR_RATE,
    FORMAT_VERSION,
    ITEM_SECONDS,
    LONG_OVERLAP_SECONDS,
    MAX_CHANCE_ERROR_FRACTION,
    MAX_COMPARISON_SECONDS,
    MAX_FINGERPRINT_ITEMS,
    MIN_COMPARED_ITEMS,
    MIN_DISTINCT_VALUES,
    SHORT_OVERLAP_FACTOR,
    SHORT_OVERLAP_SECONDS,
    SILENCE_VALUE,
    AudioFingerprint,
    compare_fingerprints,
    fingerprint_audio,
    fingerprint_pcm,
    overlap_threshold,
)
from provenance_kit.audio.pcm import PcmAudio
from provenance_kit.errors import MediaDecodeError, MediaLimitExceededError
from provenance_kit.limits import MediaLimits
from tests.audio.fakes import WAV, FakeTools, probe_json


def random_fp(rng: np.random.Generator, count: int) -> AudioFingerprint:
    values = tuple(int(v) for v in rng.integers(0, 2**32, size=count, dtype=np.uint64))
    return AudioFingerprint(values=values, duration_seconds=count * ITEM_SECONDS)


def fp(values: tuple[int, ...]) -> AudioFingerprint:
    return AudioFingerprint(values=tuple(values), duration_seconds=len(values) * ITEM_SECONDS)


# The same per-bit bias for every "recording", as Chromaprint's filters give
# stationary noise: most bits are almost always 1 or almost always 0.
_BIAS_SIGNS = np.array([1.0 if bit % 3 else -1.0 for bit in range(32)])


def biased_fp(rng: np.random.Generator, count: int, strength: float = 0.8) -> AudioFingerprint:
    """Independent random sub-fingerprints whose bits are biased (|2p - 1| = strength)."""
    ones = rng.random((count, 32)) < 0.5 + 0.5 * strength * _BIAS_SIGNS
    weights = np.left_shift(np.uint64(1), np.arange(32, dtype=np.uint64))
    return fp(tuple(int(v) for v in (ones * weights).sum(axis=1, dtype=np.uint64)))


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
    assert match.chance_bit_error_rate == pytest.approx(0.5, abs=0.02)  # random bits


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
    short = noisy_clip(reference, 0, 30, rng, flip_rate=0.0)
    assert compare_fingerprints(short, reference).overlap_items == 30  # whole query
    # min_overlap_seconds=0 still needs MIN_COMPARED_ITEMS: the edge offsets
    # with 1 to 19 items of overlap are never eligible.
    edge = compare_fingerprints(short, reference, min_overlap_seconds=0, max_bit_error_rate=1.0)
    assert edge.overlap_items >= MIN_COMPARED_ITEMS


# ------------------------------------------------------------ low information (P01, P02)


def test_silence_does_not_match_silence(rng):
    gap = (SILENCE_VALUE,) * 64  # about 8 s of digital silence
    upload = fp(gap + random_fp(rng, 300).values)  # silent intro, then other music
    original = fp(random_fp(rng, 480).values + gap)  # silent outro
    match = compare_fingerprints(upload, original)
    assert not match.is_match
    assert match.bit_error_rate > 0.4  # the silence pairs are not counted
    silence_only = compare_fingerprints(fp(gap), fp(gap * 3))
    assert not silence_only.is_match
    assert silence_only.overlap_items == 0  # nothing informative to compare


def test_silence_is_left_out_of_the_comparison(rng):
    values = list(random_fp(rng, 400).values)
    values[150:190] = [SILENCE_VALUE] * 40
    reference = fp(tuple(values))
    clip = noisy_clip(reference, 100, 260, rng, flip_rate=0.05)
    match = compare_fingerprints(clip, reference)
    assert match.is_match
    assert match.offset_items == 100
    assert match.overlap_items == 160 - 40  # compared pairs only
    assert 0.02 < match.bit_error_rate < 0.08


def test_constant_codes_carry_no_information(rng):
    hum = (0x12345678,) * 120  # a steady tone gives one value, like silence
    upload = fp(hum + random_fp(rng, 200).values)
    original = fp(random_fp(rng, 400).values + hum)
    assert not compare_fingerprints(upload, original).is_match
    alone = compare_fingerprints(fp(hum), fp(hum * 3))
    assert not alone.is_match
    assert alone.bit_error_rate == 0.0
    assert alone.chance_bit_error_rate == 0.0


def test_independent_stationary_noise_does_not_match(rng):
    # Two noise recordings agree on the biased bits by chance: BER about 0.18,
    # below the 0.20 threshold, but no better than chance.
    match = compare_fingerprints(biased_fp(rng, 300), biased_fp(rng, 500))
    assert match.bit_error_rate < DEFAULT_MAX_BIT_ERROR_RATE
    assert match.chance_bit_error_rate == pytest.approx(0.18, abs=0.02)
    assert not match.is_match


def test_a_copy_of_noise_still_matches(rng):
    noise = biased_fp(rng, 400)
    clip = noisy_clip(noise, 50, 250, rng, flip_rate=0.01)
    match = compare_fingerprints(clip, noise)
    assert match.is_match
    assert match.offset_items == 50
    assert match.bit_error_rate < MAX_CHANCE_ERROR_FRACTION * match.chance_bit_error_rate


def _few_values(rng: np.random.Generator, count: int) -> tuple[int, ...]:
    """Independent draws from 8 values that differ in 4 bits (a drone or hum gives such codes)."""
    base = 0x5A3C_96E1
    alphabet = [base ^ sum(1 << (3 + 8 * j) for j in range(4) if (k >> j) & 1) for k in range(8)]
    return tuple(alphabet[i] for i in rng.integers(0, 8, size=count))


@pytest.mark.parametrize("kind", ["noise", "drone"])
def test_chance_counts_only_the_compared_pairs(rng, kind):
    """A query ending in silence must not borrow the bit statistics of what the reference has there.

    R2-M01: the chance error rate took each side's bit frequencies over its
    own non-silent items, so the reference's music facing the query's silence
    (never compared) raised the chance rate of the pairs that were compared,
    and independent noise (BER 0.18 to 0.19 against a chance of 0.42 to 0.46)
    or two unrelated drones matched.
    """
    make = biased_fp if kind == "noise" else (lambda g, n: fp(_few_values(g, n)))
    first, second = make(rng, 200), make(rng, 200)
    alone = compare_fingerprints(first, second)
    assert not alone.is_match
    music = random_fp(rng, 400).values
    silence = (SILENCE_VALUE,) * 400
    for query, reference in (
        (first.values + silence, second.values + music),
        (first.values + music, second.values + silence),
        (silence + first.values, music + second.values),
    ):
        match = compare_fingerprints(fp(query), fp(reference))
        assert not match.is_match
        assert match.chance_bit_error_rate == pytest.approx(alone.chance_bit_error_rate, abs=0.02)
    # A real copy behind the same asymmetry still matches.
    copy = noisy_clip(first, 0, 200, rng, flip_rate=0.01)
    match = compare_fingerprints(fp(copy.values + silence), fp(first.values + music))
    assert match.is_match
    assert match.offset_items == 0


# Codes fpcalc gave two independent 25 s drones (a 50 Hz hum, an organ chord)
# from round 2's repro: one steady code, then 16 items as the drone ends, the
# query's into digital silence and the reference's into unrelated music.
_DRONE_ENDS = {
    "hum": (
        0x0D65F17B,
        (0xD65B17B, 0xD65B1FB, 0xD65B5FB, 0xF65BDFB, 0xF65BDFA, 0xE65B9FA, 0xE65F9FA, 0xE65FDFA,
         0xE6735FA, 0xE67313A, 0x627313A, 0x626313E, 0x626312E, 0x62C212E, 0x162C212E, 0x366C056E),
        (0x965F17B, 0x965F17B, 0xD65F17B, 0xD65F5FB, 0x565F5FA, 0x565F5BA, 0x165F5BA, 0x165F5BA,
         0x36DF5BA, 0x26DF1BA, 0x26DF1BA, 0x22FF1BA, 0x23FB1BA, 0x21FA1BA, 0x20FA5AA, 0x20FADAB),
    ),
    "organ": (
        0x2541F977,
        (0x2541F9F7, 0x254179F7, 0x254139F7, 0x254139F7, 0x254139FF, 0x254139FF, 0x254139FF,
         0x254139FF, 0x2541397F, 0x2581393F, 0x2583391F, 0x2587391F, 0x2587391F, 0x25843907,
         0x25842907, 0x259C0947),
        (0x2541F977, 0x2541F977, 0x2541F977, 0x24417975, 0x24417D71, 0x24413571, 0x244133F3,
         0x244133F3, 0x344133F3, 0x344133F3, 0x34C733D3, 0x17863153, 0x16863DC3, 0x128619C3,
         0x128EC883, 0x169E8883),
    ),
}  # fmt: skip


@pytest.mark.parametrize("kind", sorted(_DRONE_ENDS))
def test_steady_codes_do_not_make_their_end_look_related(rng, kind):
    """Two drones that end at the same time: a run of identical pairs counts once.

    About 183 identical pairs, then the 16 items where each drone ends. Over
    all pairs the bit error rate (0.018) was 0.38 to 0.50 of its chance rate
    and 8 of 12 such fpcalc pairs matched; with the run counted once in the
    chance test it is 0.60 to 0.76 of it, and none does.
    """
    steady, query_end, reference_end = _DRONE_ENDS[kind]
    query = (steady,) * 183 + query_end + (SILENCE_VALUE,) * 300
    reference = (steady,) * 183 + reference_end + random_fp(rng, 300).values
    match = compare_fingerprints(fp(query), fp(reference))
    assert match.offset_items == 0
    assert match.bit_error_rate < 0.02  # the steady pairs agree, and dominate
    assert not match.is_match
    # The drone and its end, copied, still match.
    copy = noisy_clip(fp(query[:199]), 0, 199, rng, flip_rate=0.01)
    assert compare_fingerprints(fp(copy.values + (SILENCE_VALUE,) * 300), fp(query)).is_match


def _chance_test_by_hand(q: np.ndarray, r: np.ndarray, offset: int) -> tuple[float, int, int]:
    """The chance test's definition: compared pairs, a run of identical pairs counted once."""
    pairs: list[tuple[int, int]] = []
    previous = None
    for i in range(max(0, -offset), min(q.size, r.size - offset)):
        pair = (int(q[i]), int(r[i + offset]))
        if SILENCE_VALUE not in pair and pair != previous:
            pairs.append(pair)
        previous = pair
    if not pairs:
        return 0.0, 0, 0
    total = 0.0
    for bit in range(32):
        p_q = sum((a >> bit) & 1 for a, _ in pairs) / len(pairs)
        p_r = sum((b >> bit) & 1 for _, b in pairs) / len(pairs)
        total += p_q * (1 - p_r) + p_r * (1 - p_q)
    errors = sum(bin(a ^ b).count("1") for a, b in pairs)
    return total / 32, errors, len(pairs)


@pytest.mark.parametrize("budget", [None, 0])
def test_the_chance_test_matches_its_definition(rng, monkeypatch, budget):
    # Both code paths: a few offsets directly, many (or a budget of 0) by FFT.
    if budget is not None:
        monkeypatch.setattr(fingerprint, "_DIRECT_CHANCE_BUDGET", budget)
    q = np.array(biased_fp(rng, 60).values, dtype=np.uint32)
    r = np.array(random_fp(rng, 90).values, dtype=np.uint32)
    q[10:25] = SILENCE_VALUE
    r[40:70] = SILENCE_VALUE
    q[30:45] = q[30]  # steady sections: runs of one code on each side
    r[0:20] = r[0]
    r[75:90:2] = r[75]
    for offsets in (np.array([-20, 0, 7, 30]), np.arange(-59, 90)):
        chance, errors, pairs = fingerprint._chance_test(q, r, offsets)
        expected = [_chance_test_by_hand(q, r, int(d)) for d in offsets]
        np.testing.assert_allclose(chance, [e[0] for e in expected], atol=1e-12)
        assert errors.tolist() == [e[1] for e in expected]
        assert pairs.tolist() == [e[2] for e in expected]


def test_a_genuine_overlap_is_found_behind_many_low_diversity_offsets(rng):
    """R2-M08: "a match when any offset passes" must hold past the 64 diversity checks.

    The reference is a two-value alternation (a two-tone alarm) and then a
    song; the query starts with the song's last 65 items and continues with
    the same alternation. Hundreds of offsets pair the alternations with a
    better z-score than the genuine overlap and pass the bit error rate and
    chance tests; only the distinct-value test rejects them.
    """
    a, b = (int(v) for v in rng.integers(0, 2**32, size=2, dtype=np.uint64))
    song = random_fp(rng, 1000)
    tail = noisy_clip(song, 935, 1000, rng, flip_rate=0.04)
    query = fp(tail.values + (a, b) * 250)
    # The round-2 repro's layout: the alternation runs straight into the song.
    # Offsets that pair the alternations plus 6 or more song items also hold
    # 8 distinct values and pass, with a better z-score than the genuine one.
    reference = fp((a, b) * 500 + song.values)
    assert compare_fingerprints(tail, reference).is_match
    assert compare_fingerprints(query, reference).is_match  # was no match at all
    # With silence between them, the genuine overlap is the only one that passes.
    reference = fp((a, b) * 500 + (SILENCE_VALUE,) * 600 + song.values)
    match = compare_fingerprints(query, reference)
    assert match.is_match
    assert match.offset_items == 1000 + 600 + 935
    assert match.overlap_items == 65
    # More offsets pass the bit error rate and chance tests than are checked.
    errors, compared = fingerprint._offset_counts(
        np.array(query.values, np.uint32), np.array(reference.values, np.uint32)
    )
    passing = (compared >= 40) & (errors <= 0.11 * 32 * compared)
    assert np.count_nonzero(passing) > 2 * fingerprint._MAX_DIVERSITY_CHECKS


def test_the_distinct_value_filter_matches_its_definition(rng):
    values = np.array(
        [SILENCE_VALUE] * 5 + [1, 1, 2, 1, 2] * 4 + list(range(10, 22)) + [3, SILENCE_VALUE] * 9,
        dtype=np.uint32,
    )
    starts, stops = np.meshgrid(np.arange(values.size), np.arange(values.size + 1))
    starts, stops = starts.ravel(), stops.ravel()
    keep = stops > starts
    starts, stops = starts[keep], stops[keep]
    got = fingerprint._holds_distinct(values, starts, stops)
    expected = [
        len({int(v) for v in values[s:e] if v != SILENCE_VALUE}) >= MIN_DISTINCT_VALUES
        for s, e in zip(starts, stops, strict=True)
    ]
    assert got.tolist() == expected
    assert not fingerprint._holds_distinct(
        np.full(10, SILENCE_VALUE, np.uint32), np.array([0]), np.array([10])
    ).any()


def test_short_queries_never_match(rng):
    reference = random_fp(rng, 4000)
    for count in (1, 3, MIN_COMPARED_ITEMS - 1):
        exact = noisy_clip(reference, 1000, 1000 + count, rng, flip_rate=0.0)
        match = compare_fingerprints(exact, reference)
        assert not match.is_match
        assert match.overlap_items == 0
    at_floor = noisy_clip(reference, 1000, 1000 + MIN_COMPARED_ITEMS, rng, flip_rate=0.05)
    match = compare_fingerprints(at_floor, reference)
    assert match.is_match
    assert match.offset_items == 1000


# ------------------------------------------------------------ length-aware threshold


def test_overlap_threshold_is_stricter_for_short_overlaps():
    short = round(SHORT_OVERLAP_SECONDS / ITEM_SECONDS)  # 40 items, 4.95 s of span
    long = math.ceil(LONG_OVERLAP_SECONDS / ITEM_SECONDS)  # 162 items
    assert overlap_threshold(0) == pytest.approx(0.11)
    assert overlap_threshold(short) == pytest.approx(SHORT_OVERLAP_FACTOR * 0.20)
    assert overlap_threshold(long) == pytest.approx(DEFAULT_MAX_BIT_ERROR_RATE)
    assert overlap_threshold(10_000) == pytest.approx(0.20)
    middle = round(12.5 / ITEM_SECONDS)  # halfway: 0.155
    assert overlap_threshold(middle) == pytest.approx(0.155, abs=0.001)
    assert overlap_threshold(0, max_bit_error_rate=0.30) == pytest.approx(0.165)
    with pytest.raises(ValueError):
        overlap_threshold(-1)
    with pytest.raises(ValueError):
        overlap_threshold(10, max_bit_error_rate=1.5)
    with pytest.raises(TypeError):
        overlap_threshold(1.5)  # type: ignore[arg-type]


def test_short_overlaps_need_a_closer_agreement(rng):
    # A flat 0.20 let 5 to 10 s windows of signals that sound alike match
    # (12 to 36 % of the suite's synthetic song windows); the same bit error
    # rate is still accepted over a long overlap.
    reference = random_fp(rng, 2000)
    short = noisy_clip(reference, 500, 540, rng, flip_rate=0.16)  # 4.95 s of span
    match = compare_fingerprints(short, reference)
    assert 0.13 < match.bit_error_rate < 0.19
    assert match.bit_error_threshold == pytest.approx(0.11)
    assert not match.is_match
    assert match.offset_items == 500  # still reported as the best alignment
    long = noisy_clip(reference, 500, 800, rng, flip_rate=0.16)  # 37 s of span
    match = compare_fingerprints(long, reference)
    assert match.is_match
    assert match.bit_error_threshold == pytest.approx(0.20)
    assert match.offset_items == 500
    lenient = compare_fingerprints(short, reference, max_bit_error_rate=0.40)
    assert lenient.is_match  # the short threshold scales with it: 0.22
    assert lenient.bit_error_threshold == pytest.approx(0.22)


# ------------------------------------------------------------ offset choice (P05)


def test_a_genuine_edge_overlap_beats_a_long_unrelated_one(rng):
    # The upload starts with the original's last 60 items, then other
    # material; both have biased bits, so long overlaps of the two reach a
    # high z-score at a bit error rate of about 0.32.
    original = biased_fp(rng, 800, strength=0.6)
    tail = noisy_clip(original, 740, 800, rng, flip_rate=0.01)
    upload = fp(tail.values + biased_fp(rng, 1200, strength=0.6).values)
    match = compare_fingerprints(upload, original)
    assert match.is_match
    assert match.offset_items == 740
    assert match.overlap_items == 60
    assert match.bit_error_rate < 0.03


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


@pytest.mark.parametrize("name", ["min_overlap_seconds", "max_offset_seconds"])
def test_huge_durations_are_refused_as_out_of_range(rng, name):
    # R2-M09: 1e308 is finite, but seconds / ITEM_SECONDS overflowed to inf
    # and math.floor / math.ceil raised an undocumented OverflowError.
    query, reference = random_fp(rng, 40), random_fp(rng, 100)
    for value in (1e308, MAX_COMPARISON_SECONDS * 1.001):
        with pytest.raises(ValueError, match=name):
            compare_fingerprints(query, reference, **{name: value})
    # The maximum itself is accepted: no fingerprint is longer.
    match = compare_fingerprints(query, query, **{name: MAX_COMPARISON_SECONDS})
    assert match.is_match
    assert pytest.approx(MAX_FINGERPRINT_ITEMS * ITEM_SECONDS) == MAX_COMPARISON_SECONDS


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
