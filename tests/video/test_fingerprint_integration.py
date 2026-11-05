"""Video fingerprint robustness with real encodes and real PDQ hashes.

Measured on the reference machine (2026-11-05; 40 s of the LIFE source,
640x360, 1 frame per second, PDQ via pdqhash 0.2.8):

=============================  =======  =====================  ==========
Variant                        Offset   Matched / compared     Similarity
=============================  =======  =====================  ==========
VP9 at 150 kb/s                0 s      40 / 40                0.970
Scaled to 320x180, 300 kb/s    0 s      40 / 40                0.970
15 fps instead of 30           0 s      40 / 40                0.954
Trim 12.4 s to 27.4 s          12 s     15 / 15                0.889
Trim 5 s to 20 s, VP9 480x270  5 s      15 / 15                0.976
Unrelated (testsrc2)           n/a      0 matched              0.528
=============================  =======  =====================  ==========

With the default per-frame threshold of 40 bits. The 12.4 s trim is the worst
case by design: its samples fall 0.4 s off the reference's sampling grid in
content whose consecutive seconds differ by about 50 PDQ bits (median distance
30, maximum 34). The image module's PDQ hasher gives the same hashes as
pdqhash on these frames (similarity 1.0).
The tests below assert the decisions and loose bounds; run with
``-s`` to print the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from provenance_kit.tools import ExternalTools, Tool
from provenance_kit.video.fingerprint import (
    VideoFingerprint,
    fingerprint_video,
    match_fingerprints,
)
from tests.support import find_tools, require_module, require_tool
from tests.video.media import LIFE, TESTSRC2, ffmpeg, lavfi

pytestmark = pytest.mark.integration

MP4 = ["-f", "mp4", "-movflags", "frag_keyframe+empty_moov"]


@dataclass(frozen=True)
class _Hash:
    value: bytes
    quality: int


class PdqhashAdapter:
    """A FrameHasher over the pdqhash package (independent of the image module)."""

    def __init__(self) -> None:
        self._pdqhash = require_module("pdqhash", extra="image-hash")

    def hash_image(self, image):
        vector, quality = self._pdqhash.compute(np.asarray(image.convert("RGB")))
        return _Hash(np.packbits(np.asarray(vector, dtype=np.uint8)).tobytes(), int(quality))


@pytest.fixture(scope="module")
def tools() -> ExternalTools:
    require_tool(Tool.FFMPEG)
    require_tool(Tool.FFPROBE)
    return find_tools()


@pytest.fixture(scope="module")
def reference(tools) -> bytes:
    return ffmpeg(
        tools,
        [
            *lavfi(LIFE, width=640, height=360, rate=30),
            "-t",
            "40",
            "-c:v",
            "libopenh264",
            "-b:v",
            "2M",
            "-pix_fmt",
            "yuv420p",
            *MP4,
            "pipe:1",
        ],
    )


@pytest.fixture(scope="module")
def reference_fp(reference, tools) -> VideoFingerprint:
    return fingerprint_video(reference, hasher=PdqhashAdapter(), tools=tools)


VARIANTS = {
    "vp9-150k": (
        [
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "150k",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-f",
            "webm",
        ],
        0,
    ),
    "scale-320x180-300k": (
        ["-vf", "scale=320:180", "-c:v", "libopenh264", "-b:v", "300k", *MP4],
        0,
    ),
    "fps-15": (["-r", "15", "-c:v", "libopenh264", "-b:v", "800k", *MP4], 0),
    "trim-12.4-27.4": (
        ["-ss", "12.4", "-t", "15", "-c:v", "libopenh264", "-b:v", "800k", *MP4],
        12,
    ),
    "trim-5-20-vp9-480x270": (
        [
            "-ss",
            "5",
            "-t",
            "15",
            "-vf",
            "scale=480:270",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "200k",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-f",
            "webm",
        ],
        5,
    ),
}


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_derived_videos_match_at_the_right_offset(name, reference, reference_fp, tools):
    args, offset = VARIANTS[name]
    derived = ffmpeg(tools, ["-i", "pipe:0", *args, "pipe:1"], stdin=reference)
    query = fingerprint_video(derived, hasher=PdqhashAdapter(), tools=tools)
    match = match_fingerprints(query, reference_fp)
    print(f"\n{name}: {match}")
    assert match.is_match
    assert match.offset_seconds == offset
    assert match.match_fraction >= 0.5
    assert match.similarity > 0.85


def test_unrelated_video_does_not_match(reference_fp, tools):
    other = ffmpeg(
        tools,
        [
            *lavfi(TESTSRC2, width=640, height=360, rate=25),
            "-t",
            "15",
            "-c:v",
            "libopenh264",
            "-b:v",
            "800k",
            *MP4,
            "pipe:1",
        ],
    )
    match = match_fingerprints(
        fingerprint_video(other, hasher=PdqhashAdapter(), tools=tools), reference_fp
    )
    print(f"\nunrelated: {match}")
    assert not match.is_match
    assert match.matched_frames == 0
    assert match.similarity < 0.65


def test_fingerprints_survive_serialization(reference_fp):
    data = reference_fp.to_bytes()
    assert len(data) == 20 + 33 * len(reference_fp)
    assert VideoFingerprint.from_bytes(data) == reference_fp


def test_the_image_module_hasher_works_when_available(reference, reference_fp, tools):
    hashing = require_module("provenance_kit.image.hashing", extra="video")
    if not hasattr(hashing, "PdqHasher"):
        pytest.skip("provenance_kit.image.hashing has no PdqHasher yet")
    fp = fingerprint_video(reference, tools=tools)  # the default hasher
    assert len(fp) == len(reference_fp)
    assert fp.hash_bits == 256
    match = match_fingerprints(fp, reference_fp)
    print(f"\nimage-module PDQ vs pdqhash: {match}")
    assert match.is_match
    assert match.offset_frames == 0
