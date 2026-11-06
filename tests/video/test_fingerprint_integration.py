"""Video fingerprint robustness with real encodes and real PDQ hashes.

PDQ comes from the core install (the image module's NumPy port), so these
tests run wherever ffmpeg and ffprobe are available; only the cross-check
against the native ``pdqhash`` backend needs the ``pdq-native`` extra.

Measured on the reference machine (2026-11-05; 40 s of the LIFE source,
640x360, 1 frame per second, PDQ via the NumPy port, default thresholds):

===============================  =======  ==================  ========  ==========
Variant                          Offset   Matched / compared  Distinct  Similarity
===============================  =======  ==================  ========  ==========
VP9 at 150 kb/s                  0 s      40 / 40             38        0.967
Scaled to 320x180, 300 kb/s      0 s      40 / 40             39        0.971
15 fps instead of 30             0 s      39 / 40             38        0.956
Trim 12.4 s to 27.4 s            12 s     15 / 15             14        0.902
Trim 5 s to 20 s, VP9 480x270    5 s      15 / 15             15        0.977
Unrelated (testsrc2)             n/a      0 matched           0         0.539
15 s excerpt, then 25 s other    0 s      15 / 15             15        0.981
25 s other, then 15 s excerpt    -25 s    15 / 15             15        0.975
Shared 3 s end card only         n/a      3 / 3 (the card)    1         0.995
Shared 8 s end card only         n/a      8 / 8 (the card)    1         0.993
Shared grainy card, sigma 8      n/a      8 / 8 (the card)    1         0.897
Shared grainy card, sigma 16     n/a      5 / 5 (the card)    2         0.850
Shared blinking card             n/a      8 / 8 (the card)    1         0.995
Shared sting, 2 s black fades    n/a      3 / 3 over 7 s      3         0.997
===============================  =======  ==================  ========  ==========

The 12.4 s trim is the worst case by design: its samples fall 0.4 s off the
reference's sampling grid in content whose consecutive seconds differ by
about 54 PDQ bits (median distance 24, maximum 34 within the segment). The
15 fps encode has one sample 44 bits away (the frame rate halving shifts it
by one source frame), which the segment rule tolerates. The
tests below assert the decisions and loose bounds; run with ``-s`` to print
the numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from provenance_kit.image.hashing import PdqHasher
from provenance_kit.tools import ExternalTools, Tool
from provenance_kit.video.fingerprint import (
    VideoFingerprint,
    fingerprint_video,
    match_fingerprints,
)
from tests.support import find_tools, require_module, require_tool
from tests.video.media import (
    LIFE,
    MANDELBROT,
    TESTSRC2,
    encode_frames,
    ffmpeg,
    lavfi,
    raw_frames,
)

pytestmark = pytest.mark.integration

MP4 = ["-f", "mp4", "-movflags", "frag_keyframe+empty_moov"]

#: The core install's PDQ backend: identical results on every machine.
HASHER = PdqHasher("numpy")


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
    return fingerprint_video(reference, hasher=HASHER, tools=tools)


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
    query = fingerprint_video(derived, hasher=HASHER, tools=tools)
    match = match_fingerprints(query, reference_fp)
    print(f"\n{name}: {match}")
    assert match.is_match
    assert match.offset_seconds == offset
    assert match.match_fraction >= 0.5
    assert match.distinct_frames >= 14
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
    match = match_fingerprints(fingerprint_video(other, hasher=HASHER, tools=tools), reference_fp)
    print(f"\nunrelated: {match}")
    assert not match.is_match
    assert match.matched_frames == 0
    assert match.similarity < 0.65


@pytest.mark.parametrize("order", ["excerpt-first", "excerpt-last"])
def test_an_excerpt_next_to_other_content_is_found(order, reference, reference_fp, tools):
    # The first 15 s of the reference, re-encoded, then (or after) 25 s of
    # unrelated content. The old rule found only the excerpt-last case: in
    # the other one the unrelated 25 s overlapped the reference at offset 0.
    graph = (
        "[0:v]trim=0:15,setpts=PTS-STARTPTS,scale=640:360,setsar=1,fps=30,format=yuv420p[x];"
        "[1:v]scale=640:360,setsar=1,fps=30,format=yuv420p[y];"
        + ("[x][y]" if order == "excerpt-first" else "[y][x]")
        + "concat=n=2:v=1[v]"
    )
    query = ffmpeg(
        tools,
        [
            "-i",
            "pipe:0",
            "-f",
            "lavfi",
            "-t",
            "25",  # an input option: 25 s of the unrelated source
            "-i",
            MANDELBROT.format(w=320, h=240, r=25),
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "400k",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-f",
            "webm",
            "pipe:1",
        ],
        stdin=reference,
    )
    match = match_fingerprints(fingerprint_video(query, hasher=HASHER, tools=tools), reference_fp)
    print(f"\n{order}: {match}")
    assert match.is_match
    start = 0.0 if order == "excerpt-first" else 25.0
    assert match.offset_seconds == -start
    assert match.query_start_seconds == pytest.approx(start, abs=1)
    assert match.query_end_seconds == pytest.approx(start + 15, abs=1)
    assert match.reference_start_seconds == pytest.approx(0, abs=1)
    assert match.matched_seconds >= 13


def _end_card(path) -> str:
    card = Image.new("RGB", (640, 360), (20, 20, 30))
    draw = ImageDraw.Draw(card)
    draw.ellipse([250, 70, 390, 210], fill=(240, 60, 90))
    draw.rectangle([300, 120, 340, 260], fill=(250, 250, 250))
    draw.text((230, 290), "made with ExampleApp", fill=(230, 230, 230))
    card.save(path)
    return str(path)


def _with_end_card(tools, source: str, *, seconds: int, card: str, card_seconds: int, size, codec):
    width, height = size
    graph = (
        f"[0:v]scale={width}:{height},setsar=1,fps=25,format=yuv420p[a];"
        f"[1:v]scale={width}:{height},setsar=1,fps=25,format=yuv420p[b];[a][b]concat=n=2:v=1[v]"
    )
    return ffmpeg(
        tools,
        [
            "-f",
            "lavfi",
            "-t",
            str(seconds),
            "-i",
            source,
            "-loop",
            "1",
            "-framerate",
            "25",
            "-t",
            str(card_seconds),
            "-i",
            card,
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            *codec,
            "pipe:1",
        ],
    )


@pytest.mark.parametrize("card_seconds", [3, 8])
def test_unrelated_videos_sharing_only_an_end_card_do_not_match(tools, tmp_path, card_seconds):
    # 20 s of LIFE + the card (MPEG-4) against 3 s of Mandelbrot + the same
    # card (VP9, another size): 3 of 6 frames matched, a match under the old rule.
    card = _end_card(tmp_path / "card.png")
    long_video = _with_end_card(
        tools,
        LIFE.format(w=320, h=240, r=25, seed=3),
        seconds=20,
        card=card,
        card_seconds=card_seconds,
        size=(640, 360),
        codec=["-c:v", "mpeg4", "-q:v", "4", *MP4],
    )
    clip = _with_end_card(
        tools,
        MANDELBROT.format(w=320, h=240, r=25),
        seconds=3,
        card=card,
        card_seconds=card_seconds,
        size=(480, 270),
        codec=["-c:v", "libvpx-vp9", "-b:v", "300k", "-deadline", "realtime", "-f", "webm"],
    )
    match = match_fingerprints(
        fingerprint_video(clip, hasher=HASHER, tools=tools),
        fingerprint_video(long_video, hasher=HASHER, tools=tools),
    )
    print(f"\nshared {card_seconds} s end card: {match}")
    assert not match.is_match
    assert match.distinct_frames == 1  # the card, counted once
    assert match.matched_frames >= card_seconds - 1
    assert match.reference_start_seconds == pytest.approx(20, abs=1)


TAIL_RATE = 10  # frames per second of the generated tails (enough at 1 sample per second)


def _card_picture(kind: str, size=(480, 270)) -> np.ndarray:
    card = Image.new("RGB", (640, 360), (20, 20, 30))
    draw = ImageDraw.Draw(card)
    if kind in ("end", "banner"):
        draw.ellipse([250, 70, 390, 210], fill=(240, 60, 90))
        draw.rectangle([300, 120, 340, 260], fill=(250, 250, 250))
        draw.text((230, 290), "made with ExampleApp", fill=(230, 230, 230))
    if kind == "banner":
        draw.rectangle([0, 0, 640, 60], fill=(250, 210, 40))
        draw.rectangle([20, 280, 620, 350], fill=(220, 30, 30))
    elif kind == "sting-1":
        draw.rectangle([80, 60, 300, 300], fill=(60, 140, 240))
        draw.ellipse([360, 80, 560, 280], fill=(250, 250, 250))
    elif kind == "sting-2":
        draw.polygon([(320, 30), (600, 330), (40, 330)], fill=(240, 200, 60))
        draw.rectangle([260, 200, 380, 320], fill=(30, 30, 30))
    elif kind == "sting-3":
        for i in range(8):
            draw.rectangle([i * 80, 0, i * 80 + 40, 360], fill=(200, 60 + 20 * i, 90))
        draw.ellipse([220, 100, 420, 300], fill=(20, 220, 120))
    return np.asarray(card.resize(size, Image.Resampling.BICUBIC))


def _shared_tail(name: str, size, seed: int) -> list[np.ndarray]:
    """The boilerplate both unrelated videos end with, frame by frame."""
    height, width = size[1], size[0]
    rng = np.random.default_rng(seed)
    if name.startswith("grain-"):  # a still end card with film grain of its own
        sigma = int(name.split("-")[1])
        card = _card_picture("end", size).astype(np.float32)
        frames = []
        for _ in range(8 * TAIL_RATE):
            grain = rng.normal(0, sigma, (height // 2, width // 2, 1)).repeat(2, 0).repeat(2, 1)
            frames.append(np.clip(card + grain, 0, 255).astype(np.uint8))
        return frames
    if name == "blinking":  # two pictures alternating every second
        states = [_card_picture("end", size), _card_picture("banner", size)]
        return [states[k // TAIL_RATE % 2] for k in range(8 * TAIL_RATE)]
    if name == "sting-with-fades":  # 3 pictures of 1 s with 2 s of black between them
        black = np.zeros((height, width, 3), np.uint8)
        frames = []
        for index, kind in enumerate(("sting-1", "sting-2", "sting-3")):
            frames += [_card_picture(kind, size)] * TAIL_RATE
            if index < 2:
                frames += [black] * (2 * TAIL_RATE)
        return frames
    raise AssertionError(name)


@pytest.mark.parametrize(
    "boilerplate", ["grain-8", "grain-12", "grain-16", "blinking", "sting-with-fades"]
)
def test_unrelated_videos_sharing_animated_looking_boilerplate_do_not_match(tools, boilerplate):
    # Round-2 findings: a grainy still card counted as many distinct frames
    # (consecutive samples 17 to 40 bits apart), a blinking card counted each
    # re-appearance, and black fades padded a 3 s sting into a 7 s segment.
    long_frames = raw_frames(tools, LIFE, width=480, height=270, rate=TAIL_RATE, count=100)
    short_frames = raw_frames(tools, MANDELBROT, width=320, height=180, rate=TAIL_RATE, count=30)
    reference = encode_frames(
        tools,
        long_frames + _shared_tail(boilerplate, (480, 270), seed=1),
        ["-c:v", "mpeg4", "-q:v", "4", *MP4],
        rate=TAIL_RATE,
    )
    clip = encode_frames(
        tools,
        short_frames + _shared_tail(boilerplate, (320, 180), seed=2),
        ["-c:v", "libvpx-vp9", "-b:v", "200k", "-deadline", "realtime", "-f", "webm"],
        rate=TAIL_RATE,
    )
    match = match_fingerprints(
        fingerprint_video(clip, hasher=HASHER, tools=tools),
        fingerprint_video(reference, hasher=HASHER, tools=tools),
    )
    print(f"\n{boilerplate}: {match}")
    assert not match.is_match
    assert match.distinct_frames <= (3 if boilerplate == "sting-with-fades" else 2)
    # The segment reported is the boilerplate (reference seconds 10 to 18).
    assert match.matched_frames >= 3
    assert 9 <= match.reference_start_seconds < match.reference_end_seconds <= 19


def test_fingerprints_survive_serialization(reference_fp):
    data = reference_fp.to_bytes()
    assert len(data) == 20 + 33 * len(reference_fp)
    assert VideoFingerprint.from_bytes(data) == reference_fp


def test_the_default_hasher_matches_the_reference(reference, reference_fp, tools):
    fp = fingerprint_video(reference, tools=tools)  # PdqHasher("auto")
    assert len(fp) == len(reference_fp)
    assert fp.hash_bits == 256
    match = match_fingerprints(fp, reference_fp)
    assert match.is_match
    assert match.offset_frames == 0


def test_the_native_pdq_backend_gives_the_same_fingerprint(reference, reference_fp, tools):
    require_module("pdqhash", extra="pdq-native")
    native = fingerprint_video(reference, hasher=PdqHasher("pdqhash"), tools=tools)
    match = match_fingerprints(native, reference_fp)
    print(f"\npdqhash vs NumPy port: {match}")
    assert match.is_match
    assert match.offset_frames == 0
    assert match.similarity > 0.99
