"""Unit tests of video fingerprints with a fake frame hasher."""

from __future__ import annotations

import importlib
import struct
from dataclasses import dataclass

import numpy as np
import pytest

from provenance_kit.errors import BackendUnavailableError
from provenance_kit.video import fingerprint as fp_module
from provenance_kit.video.fingerprint import (
    FORMAT_VERSION,
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
    assert not match_fingerprints(short, reference).is_match  # fewer than 3 frames
    assert match_fingerprints(short, reference, min_matched_frames=2).is_match


def test_max_offset_restricts_the_search(rng):
    reference = random_fingerprint(rng, 40)
    clip = sub_clip(reference, 30, 40, rng, noise_bits=0)
    assert match_fingerprints(clip, reference).offset_frames == 30
    assert not match_fingerprints(clip, reference, max_offset_seconds=10).is_match


def test_large_fingerprints_are_matched_in_blocks(rng, monkeypatch):
    monkeypatch.setattr(fp_module, "_CHUNK_CELLS", 64)
    reference = random_fingerprint(rng, 50)
    clip = sub_clip(reference, 20, 35, rng, noise_bits=10)
    match = match_fingerprints(clip, reference)
    assert match.offset_frames == 20
    assert match.matched_frames == 15


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
        {"min_matched_frames": 0},
        {"min_match_fraction": 1.5},
        {"max_offset_seconds": -1.0},
    ):
        with pytest.raises(ValueError):
            match_fingerprints(fp, fp, **kwargs)


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
