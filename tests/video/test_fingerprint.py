"""Unit tests of video fingerprints with a fake frame hasher."""

from __future__ import annotations

import importlib
import math
import struct
from dataclasses import dataclass

import numpy as np
import pytest

from provenance_kit.errors import BackendUnavailableError
from provenance_kit.video import fingerprint as fp_module
from provenance_kit.video.fingerprint import (
    FORMAT_VERSION,
    MAX_DISTINCT_FRAMES,
    MIN_FINGERPRINT_FPS,
    VideoFingerprint,
    default_frame_hasher,
    fingerprint_frames,
    fingerprint_video,
    match_fingerprints,
)
from provenance_kit.video.frames import VideoFrame
from tests.video.fakes import MP4_MOOV_FIRST, FakeTools, probe_json


@dataclass(frozen=True)
class FakeHash:
    value: bytes
    quality: int


class FakeHasher:
    """Hashes an image to 32 bytes derived from its mean pixel value."""

    def __init__(self, quality: int = 100) -> None:
        self.quality = quality
        self.sizes: list[tuple[int, int]] = []

    def hash_image(self, image):
        self.sizes.append(image.size)
        level = int(np.asarray(image).mean())
        rng = np.random.default_rng(level)
        return FakeHash(rng.integers(0, 256, 32, dtype=np.uint8).tobytes(), self.quality)


def random_fingerprint(rng: np.random.Generator, count: int, fps: float = 1.0) -> VideoFingerprint:
    hashes = tuple(int.from_bytes(rng.bytes(32), "big") for _ in range(count))
    return VideoFingerprint(fps=fps, hash_bits=256, hashes=hashes, qualities=(100,) * count)


def flip_bits(value: int, rng: np.random.Generator, count: int) -> int:
    for bit in rng.choice(256, size=count, replace=False):
        value ^= 1 << int(bit)
    return value


def sub_clip(fp: VideoFingerprint, start: int, stop: int, rng, noise_bits: int) -> VideoFingerprint:
    hashes = tuple(flip_bits(h, rng, noise_bits) for h in fp.hashes[start:stop])
    return VideoFingerprint(
        fps=fp.fps, hash_bits=256, hashes=hashes, qualities=fp.qualities[start:stop]
    )


# ------------------------------------------------------------ serialization


def test_serialization_round_trip(rng):
    fp = random_fingerprint(rng, 40, fps=2.0)
    data = fp.to_bytes()
    assert len(data) == 20 + 40 * 33
    assert data[:4] == b"PKVF"
    assert data[4] == FORMAT_VERSION
    assert VideoFingerprint.from_bytes(data) == fp
    assert VideoFingerprint.from_bytes(memoryview(data)) == fp


def test_empty_fingerprint_round_trip():
    fp = VideoFingerprint(fps=1.0, hash_bits=256, hashes=(), qualities=())
    assert VideoFingerprint.from_bytes(fp.to_bytes()) == fp
    assert len(fp) == 0
    assert fp.duration_seconds == 0.0


def header(*, magic=b"PKVF", version=1, hash_bytes=32, reserved=0, fps=1.0, count=0) -> bytes:
    return struct.pack(">4sBBHdI", magic, version, hash_bytes, reserved, fps, count)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"PKVF",
        header(magic=b"XXXX"),
        header(version=2),
        header(hash_bytes=0),
        header(hash_bytes=65),
        header(reserved=1),
        header(fps=float("nan")),
        header(fps=0.0),
        header(fps=1e9),
        header(fps=1e-9),  # below MIN_FINGERPRINT_FPS
        header(fps=5e-324),
        header(count=1),
        header(count=1) + b"\x00" * 33 + b"\x00",
        header(count=1) + b"\x00" * 32 + b"\x65",  # quality 101
        header(count=2**31),
    ],
)
def test_from_bytes_rejects_invalid_data(data):
    with pytest.raises(ValueError):
        VideoFingerprint.from_bytes(data)


def test_from_bytes_rejects_non_bytes():
    with pytest.raises(TypeError):
        VideoFingerprint.from_bytes("PKVF")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 0.0},
        {"fps": 5e-324},  # gave an infinite duration
        {"fps": MIN_FINGERPRINT_FPS / 2},
        {"fps": 10**400},  # beyond the float range
        {"fps": float("inf")},
        {"hash_bits": 12},
        {"hash_bits": 1024},
        {"hashes": (2**256,), "qualities": (1,)},
        {"hashes": (-1,), "qualities": (1,)},
        {"hashes": (1,), "qualities": ()},
        {"hashes": (1,), "qualities": (101,)},
    ],
)
def test_fingerprint_validation(kwargs):
    base = {"fps": 1.0, "hash_bits": 256, "hashes": (), "qualities": ()}
    with pytest.raises(ValueError):
        VideoFingerprint(**{**base, **kwargs})


def test_timestamps_and_hex():
    fp = VideoFingerprint(fps=4.0, hash_bits=256, hashes=(1, 2**255), qualities=(10, 20))
    assert fp.timestamps == (0.0, 0.25)
    assert fp.hex_hashes() == ["0" * 63 + "1", "8" + "0" * 63]


# ------------------------------------------------------------ matching


def test_identical_fingerprints_match_at_offset_zero(rng):
    fp = random_fingerprint(rng, 30)
    match = match_fingerprints(fp, fp)
    assert match.is_match
    assert match.offset_frames == 0
    assert match.matched_frames == match.compared_frames == 30
    assert match.similarity == 1.0
    assert match.query_coverage == 1.0


@pytest.mark.parametrize("start", [0, 7, 45])
def test_a_noisy_sub_clip_is_found_at_its_offset(rng, start):
    reference = random_fingerprint(rng, 60)
    clip = sub_clip(reference, start, start + 15, rng, noise_bits=20)
    match = match_fingerprints(clip, reference)
    assert match.is_match
    assert match.offset_frames == start
    assert match.offset_seconds == float(start)
    assert match.matched_frames == 15
    assert 0.9 < match.similarity < 0.95


def test_a_clip_extending_past_the_end_matches_its_overlap(rng):
    reference = random_fingerprint(rng, 20)
    tail = random_fingerprint(rng, 10)
    clip = VideoFingerprint(
        fps=1.0,
        hash_bits=256,
        hashes=reference.hashes[14:] + tail.hashes,
        qualities=(100,) * 16,
    )
    match = match_fingerprints(clip, reference)
    assert match.offset_frames == 14
    assert match.matched_frames == 6
    assert match.compared_frames == 6
    assert match.query_coverage == pytest.approx(6 / 16)


def test_unrelated_fingerprints_do_not_match(rng):
    match = match_fingerprints(random_fingerprint(rng, 20), random_fingerprint(rng, 50))
    assert not match.is_match
    assert match.matched_frames == 0
    assert 0.4 < match.similarity < 0.6


def test_low_quality_frames_are_ignored(rng):
    fp = random_fingerprint(rng, 10)
    dull = VideoFingerprint(fps=1.0, hash_bits=256, hashes=fp.hashes, qualities=(10,) * 10)
    assert match_fingerprints(dull, fp).compared_frames == 0
    assert not match_fingerprints(dull, fp).is_match
    assert match_fingerprints(dull, fp, min_quality=0).is_match


def test_decision_thresholds(rng):
    reference = random_fingerprint(rng, 10)
    noisy = sub_clip(reference, 0, 10, rng, noise_bits=50)
    assert not match_fingerprints(noisy, reference).is_match
    assert match_fingerprints(noisy, reference, max_distance=60).is_match
    short = sub_clip(reference, 0, 2, rng, noise_bits=0)
    assert not match_fingerprints(short, reference).is_match  # 2 frames, 2 seconds
    assert not match_fingerprints(short, reference, min_distinct_frames=2).is_match  # < 5 s
    lenient = match_fingerprints(short, reference, min_distinct_frames=2, min_matched_seconds=2)
    assert lenient.is_match
    assert lenient.matched_seconds == 2.0


def test_max_offset_restricts_the_search(rng):
    reference = random_fingerprint(rng, 40)
    clip = sub_clip(reference, 30, 40, rng, noise_bits=0)
    assert match_fingerprints(clip, reference).offset_frames == 30
    assert not match_fingerprints(clip, reference, max_offset_seconds=10).is_match


def test_a_query_longer_than_the_reference_is_aligned_the_same_way(rng):
    # The scan runs over the shorter fingerprint's rows; swapping the roles
    # must only negate the offset and swap the segment's times.
    reference = random_fingerprint(rng, 50)
    clip = sub_clip(reference, 20, 35, rng, noise_bits=10)
    forward = match_fingerprints(clip, reference)
    backward = match_fingerprints(reference, clip)
    assert forward.offset_frames == 20
    assert backward.offset_frames == -20
    for match in (forward, backward):
        assert match.is_match
        assert (match.matched_frames, match.distinct_frames) == (15, 15)
        assert match.similarity == forward.similarity
    assert (forward.query_start_seconds, forward.query_end_seconds) == (0.0, 15.0)
    assert (forward.reference_start_seconds, forward.reference_end_seconds) == (20.0, 35.0)
    assert (backward.query_start_seconds, backward.query_end_seconds) == (20.0, 35.0)
    assert (backward.reference_start_seconds, backward.reference_end_seconds) == (0.0, 15.0)


def test_matching_validates_its_arguments(rng):
    fp = random_fingerprint(rng, 5)
    other_rate = random_fingerprint(rng, 5, fps=2.0)
    with pytest.raises(ValueError, match="rates"):
        match_fingerprints(fp, other_rate)
    small = VideoFingerprint(fps=1.0, hash_bits=64, hashes=(1,), qualities=(100,))
    with pytest.raises(ValueError, match="hash sizes"):
        match_fingerprints(fp, small)
    with pytest.raises(TypeError):
        match_fingerprints(fp, fp.to_bytes())  # type: ignore[arg-type]
    for kwargs in (
        {"max_distance": 257},
        {"min_quality": 101},
        {"min_distinct_frames": 0},
        {"min_matched_seconds": -1.0},
        {"min_match_fraction": 1.5},
        {"max_gap_seconds": -0.5},
        {"static_distance": 257},
        {"static_distance": -1},
        {"max_offset_seconds": -1.0},
        {"min_distinct_frames": MAX_DISTINCT_FRAMES + 1},
        # Finite but huge: the products with fps overflowed math.floor/ceil.
        {"max_gap_seconds": 1e308},
        {"min_matched_seconds": 1e308},
        {"max_offset_seconds": 1e308},
        {"max_gap_seconds": 10**400},
        {"max_offset_seconds": 10**400},
    ):
        with pytest.raises(ValueError):
            match_fingerprints(fp, fp, **kwargs)


def test_the_lowest_and_highest_sampling_rates_are_accepted(rng):
    for fps in (MIN_FINGERPRINT_FPS, 240):
        fp = random_fingerprint(rng, 10, fps=fps)
        assert fp.duration_seconds == 10 / fps
        match = match_fingerprints(fp, fp, max_gap_seconds=1e9, max_offset_seconds=1e9)
        assert match.is_match == (10 / fps >= 5)  # at 240 fps, 10 samples last 0.04 s
        assert match.query_end_seconds == 10 / fps
    with pytest.raises(ValueError, match="fps"):
        fingerprint_frames([], fps=MIN_FINGERPRINT_FPS / 2, hasher=FakeHasher())


# ------------------------------------------------------------ segments (boilerplate, excerpts)


def make_fingerprint(hashes, qualities=None) -> VideoFingerprint:
    hashes = tuple(hashes)
    qualities = (100,) * len(hashes) if qualities is None else tuple(qualities)
    return VideoFingerprint(fps=1.0, hash_bits=256, hashes=hashes, qualities=qualities)


def random_hashes(rng, count: int) -> list[int]:
    return [int.from_bytes(rng.bytes(32), "big") for _ in range(count)]


def still(rng, picture: int, seconds: int, noise_bits: int = 6) -> list[int]:
    """A still picture (end card, title) as sampled from one encode: tiny noise."""
    return [flip_bits(picture, rng, noise_bits) for _ in range(seconds)]


@pytest.mark.parametrize("card_seconds", [3, 8, 30])
@pytest.mark.parametrize("where", ["outro", "intro"])
def test_a_shared_still_card_alone_is_not_a_match(rng, card_seconds, where):
    # Two unrelated videos that share only a platform's end card (or a title
    # card): 3/6 frames matched under the old rule, reported as a match.
    card = random_hashes(rng, 1)[0]
    mine, theirs = random_hashes(rng, 60), random_hashes(rng, 3)
    card_a, card_b = still(rng, card, card_seconds), still(rng, card, card_seconds)
    if where == "outro":
        reference, clip = make_fingerprint(mine + card_a), make_fingerprint(theirs + card_b)
    else:
        reference, clip = make_fingerprint(card_a + mine), make_fingerprint(card_b + theirs)
    match = match_fingerprints(clip, reference)
    assert not match.is_match
    # The card is still reported, as one distinct frame, for inspection.
    assert match.matched_frames == card_seconds
    assert match.distinct_frames == 1
    assert match.matched_seconds == card_seconds


def test_shared_moving_boilerplate_shorter_than_five_seconds_is_not_a_match(rng):
    intro = random_hashes(rng, 4)  # an animated logo: 4 distinct frames, 4 s
    reference = make_fingerprint(intro + random_hashes(rng, 60))
    clip = make_fingerprint([flip_bits(h, rng, 8) for h in intro] + random_hashes(rng, 20))
    match = match_fingerprints(clip, reference)
    assert match.distinct_frames == 4
    assert match.matched_seconds == 4.0
    assert not match.is_match
    assert match_fingerprints(clip, reference, min_matched_seconds=4).is_match


EXCERPT_LAYOUTS = {
    "pure trim": (0, 20, 0),
    "intro + start of the reference": (30, 0, 0),
    "end of the reference + outro": (0, 45, 30),
    "start of the reference + outro": (0, 0, 30),
    "intro + end of the reference": (30, 45, 0),
    "compilation": (30, 20, 30),
}


@pytest.mark.parametrize("layout", sorted(EXCERPT_LAYOUTS))
@pytest.mark.parametrize("direction", ["clip-in-reference", "reference-in-clip"])
def test_an_excerpt_is_found_whatever_surrounds_it(rng, layout, direction):
    # A 15 s excerpt re-encoded (8 of 256 bits off per frame) next to
    # unrelated material: the old rule missed it whenever the unrelated part
    # overlapped the reference at the excerpt's offset (fraction 15/45).
    before, start, after = EXCERPT_LAYOUTS[layout]
    reference = random_fingerprint(rng, 60)
    excerpt = [flip_bits(h, rng, 8) for h in reference.hashes[start : start + 15]]
    clip = make_fingerprint(random_hashes(rng, before) + excerpt + random_hashes(rng, after))
    if direction == "clip-in-reference":
        match = match_fingerprints(clip, reference)
        clip_times = (match.query_start_seconds, match.query_end_seconds)
        reference_times = (match.reference_start_seconds, match.reference_end_seconds)
        expected_offset = start - before
    else:
        match = match_fingerprints(reference, clip)
        clip_times = (match.reference_start_seconds, match.reference_end_seconds)
        reference_times = (match.query_start_seconds, match.query_end_seconds)
        expected_offset = before - start
    assert match.is_match
    assert match.offset_frames == expected_offset
    assert (match.matched_frames, match.distinct_frames, match.compared_frames) == (15, 15, 15)
    assert match.matched_seconds == 15.0
    assert clip_times == (before, before + 15)
    assert reference_times == (start, start + 15)
    assert match.similarity == pytest.approx(1 - 8 / 256)


def test_short_gaps_do_not_split_a_segment(rng):
    reference = random_fingerprint(rng, 40)
    hashes = list(reference.hashes[:30])
    for position in (10, 11, 20, 21):  # two gaps of two noisy samples
        hashes[position] = random_hashes(rng, 1)[0]
    match = match_fingerprints(make_fingerprint(hashes), reference)
    assert match.is_match
    assert (match.matched_frames, match.compared_frames) == (26, 30)
    assert match.matched_seconds == 30.0
    # A gap of three samples splits it: the better piece (samples 13 to 29,
    # with its own 2-sample gap) is kept.
    hashes[12] = random_hashes(rng, 1)[0]
    split = match_fingerprints(make_fingerprint(hashes), reference)
    assert (split.matched_frames, split.compared_frames) == (15, 17)
    assert (split.query_start_seconds, split.query_end_seconds) == (13.0, 30.0)
    assert (
        match_fingerprints(make_fingerprint(hashes), reference, max_gap_seconds=3).matched_frames
        == 25
    )


def test_the_match_fraction_is_measured_on_the_segment(rng):
    # One matched sample in three over 30 s: 10 distinct frames, but sparse.
    reference = random_fingerprint(rng, 30)
    hashes = [h if i % 3 == 0 else random_hashes(rng, 1)[0] for i, h in enumerate(reference.hashes)]
    match = match_fingerprints(make_fingerprint(hashes), reference)
    assert match.distinct_frames == 10
    assert match.match_fraction == pytest.approx(10 / 28)
    assert not match.is_match
    assert match_fingerprints(make_fingerprint(hashes), reference, min_match_fraction=0.3).is_match


def test_still_runs_count_once_on_each_side(rng):
    # The query shows three stills (5 s each); the reference shows the same
    # three pictures as one sample each, then holds the last one.
    pictures = random_hashes(rng, 3)
    query = make_fingerprint(
        still(rng, pictures[0], 5) + still(rng, pictures[1], 5) + still(rng, pictures[2], 5)
    )
    reference = make_fingerprint(
        still(rng, pictures[0], 5) + still(rng, pictures[1], 5) + still(rng, pictures[2], 5)
    )
    match = match_fingerprints(query, reference)
    assert (match.matched_frames, match.distinct_frames) == (15, 3)
    assert match.is_match
    assert not match_fingerprints(query, reference, min_distinct_frames=4).is_match
    # static_distance=0: only bit-identical samples are static.
    assert match_fingerprints(query, reference, static_distance=0).distinct_frames == 15


def test_a_short_fade_through_black_stays_in_the_segment_and_counts_against_it(rng):
    reference = random_fingerprint(rng, 30)
    qualities = [100] * 30
    qualities[10:12] = [10] * 2  # two black samples in the query: within max_gap
    match = match_fingerprints(make_fingerprint(reference.hashes, qualities), reference)
    assert match.is_match
    assert (match.matched_frames, match.compared_frames) == (28, 28)
    assert match.matched_seconds == 30.0
    assert match.match_fraction == pytest.approx(28 / 30)  # black samples are not matches


def test_a_long_fade_through_black_ends_the_segment(rng):
    # Unusable samples used to be neither matches nor misses, so a segment
    # stretched over any amount of black; now a gap of more than
    # max_gap_seconds of any kind ends it. Both halves still qualify.
    reference = random_fingerprint(rng, 30)
    qualities = [100] * 30
    qualities[10:20] = [10] * 10
    match = match_fingerprints(make_fingerprint(reference.hashes, qualities), reference)
    assert match.is_match
    assert (match.matched_frames, match.compared_frames) == (10, 10)
    assert (match.query_start_seconds, match.query_end_seconds) == (0.0, 10.0)
    assert match.matched_seconds == 10.0


@pytest.mark.parametrize("side", ["both", "query", "reference"])
def test_black_or_flat_samples_do_not_pad_a_short_shared_sting(rng, side):
    # Three shared pictures (a 3 s sting) separated by 10 s fades through
    # black: matched_seconds was 23 from 3 matched frames (43 with black on
    # one side only, whatever the other video showed there).
    sting = random_hashes(rng, 3)
    black = [0] * 10

    def video(with_black: bool) -> tuple[list[int], list[int]]:
        hashes, qualities = [], []
        for picture in sting:
            hashes.append(picture)
            qualities.append(100)
            filler = black if with_black else random_hashes(rng, 10)
            hashes += filler
            qualities += [0 if with_black else 100] * 10
        return hashes + random_hashes(rng, 30), qualities + [100] * 30

    query = make_fingerprint(*video(side in ("both", "query")))
    reference = make_fingerprint(*video(side in ("both", "reference")))
    match = match_fingerprints(query, reference)
    assert not match.is_match
    assert match.matched_frames == 1
    assert match.matched_seconds == 1.0


def test_a_sting_with_fades_shorter_than_the_gap_is_still_too_sparse(rng):
    # s1 x x s2 x x s3: one segment of 7 samples with 3 matches (< half).
    sting = random_hashes(rng, 3)
    layout = [sting[0], 0, 0, sting[1], 0, 0, sting[2]]
    qualities = [100, 0, 0, 100, 0, 0, 100]
    query = make_fingerprint(layout + random_hashes(rng, 20), qualities + [100] * 20)
    reference = make_fingerprint(random_hashes(rng, 20) + layout, [100] * 20 + qualities)
    match = match_fingerprints(query, reference)
    assert not match.is_match
    assert (match.matched_frames, match.matched_seconds) == (3, 7.0)
    assert match.match_fraction == pytest.approx(3 / 7)
    assert match.distinct_frames == 3
    assert match_fingerprints(query, reference, min_match_fraction=0.4).is_match


def grainy_still(rng, picture: int, seconds: int, grain_bits: int) -> list[int]:
    """A still picture with film grain: every sample ``grain_bits`` bits off the picture."""
    return [flip_bits(picture, rng, grain_bits) for _ in range(seconds)]


@pytest.mark.parametrize("grain_bits", [12, 14, 18])
def test_a_grainy_still_card_is_one_picture(rng, grain_bits):
    # Consecutive samples of a still with grain differ by 17 to 40 bits:
    # above the old 16-bit static distance, within the 40-bit match
    # threshold, so a grainy end card counted as many pictures and made two
    # unrelated videos match. Each video has grain of its own.
    card = random_hashes(rng, 1)[0]
    reference = make_fingerprint(random_hashes(rng, 30) + grainy_still(rng, card, 8, grain_bits))
    query = make_fingerprint(random_hashes(rng, 3) + grainy_still(rng, card, 8, grain_bits))
    consecutive = [
        (x ^ y).bit_count() for x, y in zip(query.hashes[3:], query.hashes[4:], strict=False)
    ]
    assert min(consecutive) > 16
    match = match_fingerprints(query, reference)
    assert not match.is_match
    assert match.matched_frames == 8
    assert match.distinct_frames == 1
    # The old behaviour, for comparison: a lower static distance splits it.
    assert match_fingerprints(query, reference, static_distance=16).is_match


@pytest.mark.parametrize(
    "pattern", ["ABABAB", "AABBAABB", "ABABABABABAB", "AAABBBAAABBB", "ABBA" * 3]
)
def test_alternating_stills_are_two_pictures(rng, pattern):
    # Two pictures shown alternately (a blinking end card, a 2-slide loop):
    # each re-appearance was counted as another distinct frame.
    pictures = {"A": random_hashes(rng, 1)[0], "B": random_hashes(rng, 1)[0]}

    def card() -> list[int]:
        return [flip_bits(pictures[c], rng, int(rng.integers(0, 3))) for c in pattern]

    reference = make_fingerprint(random_hashes(rng, 40) + card())
    query = make_fingerprint(random_hashes(rng, 40) + card())
    match = match_fingerprints(query, reference)
    assert not match.is_match
    assert match.distinct_frames == 2
    assert match.matched_frames == len(pattern)


def test_three_alternating_pictures_count_three_times_at_most(rng):
    pictures = random_hashes(rng, 3)
    card = [pictures[k % 3] for k in range(12)]
    match = match_fingerprints(
        make_fingerprint(random_hashes(rng, 5) + card),
        make_fingerprint(card + random_hashes(rng, 5)),
    )
    assert match.is_match
    assert (match.matched_frames, match.distinct_frames) == (12, 3)
    assert not match_fingerprints(
        make_fingerprint(card), make_fingerprint(card), min_distinct_frames=4
    ).is_match


def test_distinct_pictures_are_counted_exactly_beyond_the_ranking_cap(rng):
    reference = random_fingerprint(rng, 60)
    match = match_fingerprints(sub_clip(reference, 5, 45, rng, noise_bits=8), reference)
    assert (match.matched_frames, match.distinct_frames) == (40, 40)


# ------------------------------------------------------------ brute-force reference


def _anchors(fp: VideoFingerprint, min_quality: int, static_distance: int) -> list[int]:
    """The index of each sample's run anchor (-1 for unusable samples)."""
    anchors, anchor = [], None
    for index, (value, quality) in enumerate(zip(fp.hashes, fp.qualities, strict=True)):
        if quality < min_quality:
            anchors.append(-1)
            continue
        if anchor is None or (value ^ fp.hashes[anchor]).bit_count() > static_distance:
            anchor = index
        anchors.append(anchor)
    return anchors


def _brute_pictures(fp: VideoFingerprint, samples, anchors, static_distance: int) -> int:
    """Greedy picture count along matched samples: a run is new unless close to a kept one."""
    kept, previous = [], None
    for sample in samples:
        if anchors[sample] == previous:
            continue  # same run as the previous match
        previous = anchors[sample]
        if all((fp.hashes[previous] ^ fp.hashes[k]).bit_count() > static_distance for k in kept):
            kept.append(previous)
    return len(kept)


def _brute_segments(hits, max_gap):
    """Splits the matched pairs of one offset where consecutive matches are too far apart."""
    segments = []
    for pair in hits:
        if segments and pair[0] - segments[-1][-1][0] - 1 <= max_gap:
            segments[-1].append(pair)
        else:
            segments.append([pair])
    return segments


def _brute_describe(segment, usable_pairs, *, q, r, anchors, rule):
    first, last = segment[0][0], segment[-1][0]
    span = last - first + 1
    compared = [p for p in usable_pairs if first <= p[0] <= last]
    exact = min(
        _brute_pictures(q, [p[0] for p in segment], anchors[0], rule["static_distance"]),
        _brute_pictures(r, [p[1] for p in segment], anchors[1], rule["static_distance"]),
    )
    capped = min(exact, max(rule["min_distinct"], 16))
    matched = len(segment)
    passes = (
        exact >= rule["min_distinct"]
        and span >= rule["min_span"]
        and matched >= math.ceil(span * rule["min_fraction"] - 1e-9)
    )
    mean = sum(p[2] for p in compared) / len(compared)
    return (passes, capped, matched, matched / span, -mean), (exact, len(compared), first, last)


BRUTE_RULE = {
    "max_distance": 40,
    "min_quality": 50,
    "min_distinct": 3,
    "min_span": 5,
    "min_fraction": 0.5,
    "max_gap": 2,
    "static_distance": 40,
}


def brute_force(q, r, *, max_offset=None, rule=BRUTE_RULE):
    """A direct, slow transcription of match_fingerprints' documented rule."""
    anchors = (
        _anchors(q, rule["min_quality"], rule["static_distance"]),
        _anchors(r, rule["min_quality"], rule["static_distance"]),
    )
    best = None  # (key, offset, details)
    overlap = None  # (key, offset, compared): the result when nothing matches
    for d in range(-(len(q) - 1), len(r)):
        if max_offset is not None and abs(d) > max_offset:
            continue
        pairs = [
            (i, i + d, (q.hashes[i] ^ r.hashes[i + d]).bit_count())
            for i in range(len(q))
            if 0 <= i + d < len(r)
            and min(q.qualities[i], r.qualities[i + d]) >= rule["min_quality"]
        ]
        if len(pairs) >= rule["min_distinct"]:
            key = (-sum(p[2] for p in pairs) / len(pairs), -abs(d), -d)
            if overlap is None or key > overlap[0]:
                overlap = (key, d, len(pairs))
        hits = [p for p in pairs if p[2] <= rule["max_distance"]]
        diagonal_best = None  # the earliest of equal segments wins
        for segment in _brute_segments(hits, rule["max_gap"]):
            key, details = _brute_describe(segment, pairs, q=q, r=r, anchors=anchors, rule=rule)
            if diagonal_best is None or key > diagonal_best[0]:
                diagonal_best = (key, details)
        if diagonal_best is not None:
            key = (*diagonal_best[0], -abs(d), -d)
            if best is None or key > best[0]:
                best = (key, d, diagonal_best[1])
    if best is None:
        return (
            (False, 0, 0, 0, 0, 0.0, 0.0)
            if overlap is None
            else (False, overlap[1], 0, 0, overlap[2], 0.0, 0.0)
        )
    key, d, (exact, compared, first, last) = best
    return (key[0], d, key[2], exact, compared, float(first), float(last + 1))


def test_the_scan_agrees_with_a_brute_force_transcription_of_the_rule(rng):
    disagreements = []
    for trial in range(300):
        n_ref, n_query = int(rng.integers(1, 45)), int(rng.integers(1, 45))
        palette = random_hashes(rng, int(rng.integers(1, 6)))  # pictures that come back
        base = random_hashes(rng, n_ref)
        for position in range(n_ref):
            draw = rng.random()
            if position and draw < 0.3:  # a still: close to the previous sample
                base[position] = flip_bits(base[position - 1], rng, int(rng.integers(0, 30)))
            elif draw < 0.5:  # a re-appearing picture
                picture = palette[int(rng.integers(0, len(palette)))]
                base[position] = flip_bits(picture, rng, int(rng.integers(0, 25)))
        start = int(rng.integers(-8, max(n_ref, 1)))
        hashes = []
        for i in range(n_query):
            j = start + i
            if 0 <= j < n_ref and rng.random() < 0.8:
                hashes.append(flip_bits(base[j], rng, int(rng.integers(0, 60))))
            else:
                hashes.append(random_hashes(rng, 1)[0])
        # Qualities from 0: runs of unusable samples (black frames) included.
        query = make_fingerprint(hashes, [int(x) for x in rng.integers(0, 101, n_query)])
        reference = make_fingerprint(base, [int(x) for x in rng.integers(0, 101, n_ref)])
        max_offset = None if rng.random() < 0.7 else int(rng.integers(0, 10))
        rule = dict(BRUTE_RULE)
        if trial % 2:  # other settings on every other trial
            rule["min_distinct"] = int(rng.integers(1, 6))
            rule["min_span"] = int(rng.integers(1, 8))
            rule["min_fraction"] = float(rng.choice([0.0, 0.3, 0.5, 0.75, 1.0]))
            rule["max_gap"] = int(rng.integers(0, 5))
            rule["static_distance"] = int(rng.choice([0, 16, 40, 60]))
        got = match_fingerprints(
            query,
            reference,
            min_distinct_frames=rule["min_distinct"],
            min_matched_seconds=float(rule["min_span"]),
            min_match_fraction=rule["min_fraction"],
            max_gap_seconds=float(rule["max_gap"]),
            static_distance=rule["static_distance"],
            max_offset_seconds=None if max_offset is None else float(max_offset),
        )
        expected = brute_force(query, reference, max_offset=max_offset, rule=rule)
        actual = (
            got.is_match,
            got.offset_frames,
            got.matched_frames,
            got.distinct_frames,
            got.compared_frames,
            got.query_start_seconds,
            got.query_end_seconds,
        )
        if actual != expected:
            disagreements.append((trial, actual, expected))
    assert disagreements == []


# ------------------------------------------------------------ hashing frames


def test_fingerprint_frames_resizes_and_hashes(rng):
    hasher = FakeHasher(quality=77)
    frames = [rng.integers(0, 256, (36, 64, 3), dtype=np.uint8) for _ in range(3)]
    fp = fingerprint_frames(
        [VideoFrame(index=0, timestamp=0.0, image=frames[0]), *frames[1:]],
        fps=1.0,
        hasher=hasher,
        frame_size=32,
    )
    assert len(fp) == 3
    assert fp.hash_bits == 256
    assert fp.qualities == (77, 77, 77)
    assert hasher.sizes == [(32, 32)] * 3


@pytest.mark.parametrize(
    "frame",
    [np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.float32), np.zeros((0, 4, 3), np.uint8)],
)
def test_fingerprint_frames_rejects_bad_frames(frame):
    with pytest.raises(ValueError):
        fingerprint_frames([frame], fps=1.0, hasher=FakeHasher())


def test_fingerprint_frames_rejects_inconsistent_hashers():
    class Shrinking:
        def __init__(self):
            self.n = 33

        def hash_image(self, image):
            self.n -= 1
            return FakeHash(b"\x00" * self.n, 50)

    frame = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(ValueError, match="different sizes"):
        fingerprint_frames([frame, frame], fps=1.0, hasher=Shrinking())

    class BadQuality:
        def hash_image(self, image):
            return FakeHash(b"\x00" * 32, 500)

    with pytest.raises(ValueError, match="quality"):
        fingerprint_frames([frame], fps=1.0, hasher=BadQuality())


def test_fingerprint_video_decodes_square_frames():
    tools = FakeTools(ffprobe=probe_json(), ffmpeg=bytes(16 * 16 * 3) * 3)
    fp = fingerprint_video(
        MP4_MOOV_FIRST,
        hasher=FakeHasher(),
        frame_size=16,
        tools=tools,  # type: ignore[arg-type]
    )
    assert len(fp) == 3
    assert fp.fps == 1.0
    (args,) = tools.args_of("ffmpeg")
    assert "scale=16:16" in args[args.index("-vf") + 1]


def test_default_hasher_reports_a_missing_image_module(monkeypatch):
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "provenance_kit.image.hashing":
            raise ImportError("not built yet")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(fp_module.importlib, "import_module", fake_import)
    with pytest.raises(BackendUnavailableError) as info:
        default_frame_hasher()
    assert info.value.extra == "video"
