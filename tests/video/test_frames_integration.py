"""Probing and frame extraction against the real ffmpeg/ffprobe binaries."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from provenance_kit._ffmpeg import needs_seekable_input
from provenance_kit.errors import (
    MediaDecodeError,
    MediaLimitExceededError,
    UnsupportedMediaError,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType, sniff_media_type
from provenance_kit.tools import ExternalTools, Tool
from provenance_kit.video.frames import FrameSampling, extract_frames, iter_frames
from provenance_kit.video.probe import probe_video
from tests.support import find_tools, require_tool
from tests.video.media import TESTSRC2, ffmpeg, lavfi, to_file

pytestmark = pytest.mark.integration

W, H, RATE = 160, 120, 25


@pytest.fixture(scope="module")
def tools() -> ExternalTools:
    require_tool(Tool.FFMPEG)
    require_tool(Tool.FFPROBE)
    return find_tools()


@pytest.fixture(scope="module")
def media(tools, tmp_path_factory) -> dict[str, bytes]:
    """One 2-second H.264 encode in several containers, plus other codecs."""
    directory = tmp_path_factory.mktemp("media")
    source = [*lavfi(TESTSRC2, width=W, height=H, rate=RATE), "-t", "2"]
    tone = ["-f", "lavfi", "-i", "sine=frequency=440:duration=2"]
    h264 = ["-c:v", "libopenh264", "-g", "10", "-pix_fmt", "yuv420p"]
    out: dict[str, bytes] = {}
    out["moov_last"] = to_file(
        tools, [*source, *tone, *h264, "-c:a", "aac", "-shortest"], directory / "a.mp4"
    )
    first = str(directory / "a.mp4")
    out["faststart"] = to_file(
        tools, ["-i", first, "-c", "copy", "-movflags", "+faststart"], directory / "b.mp4"
    )
    out["fragmented"] = ffmpeg(
        tools,
        [
            "-i",
            "pipe:0",
            "-c",
            "copy",
            "-f",
            "mp4",
            "-movflags",
            "frag_keyframe+empty_moov",
            "pipe:1",
        ],
        stdin=out["faststart"],
    )
    out["mkv"] = ffmpeg(
        tools, ["-i", "pipe:0", "-c", "copy", "-f", "matroska", "pipe:1"], stdin=out["faststart"]
    )
    out["mpegts"] = ffmpeg(
        tools, ["-i", "pipe:0", "-c", "copy", "-f", "mpegts", "pipe:1"], stdin=out["faststart"]
    )
    out["webm"] = ffmpeg(
        tools, [*source, "-c:v", "libvpx-vp9", "-deadline", "realtime", "-f", "webm", "pipe:1"]
    )
    # AVI must be written to a file: written to a pipe, its header keeps
    # placeholder values (2**30 frames) that the limits refuse (see below).
    out["avi"] = to_file(tools, [*source, "-c:v", "mpeg4"], directory / "c.avi")
    out["avi_from_pipe"] = ffmpeg(tools, [*source, "-c:v", "mpeg4", "-f", "avi", "pipe:1"])
    out["rotated"] = to_file(
        tools, ["-display_rotation", "90", "-i", first, "-c", "copy"], directory / "r.mp4"
    )
    out["audio_only"] = to_file(tools, [*tone, "-c:a", "aac"], directory / "s.m4a")
    return out


def test_moov_at_the_end_is_decoded_from_a_temporary_file(media, tools, tmp_path):
    data = media["moov_last"]
    assert needs_seekable_input(data, MediaType.MP4)
    frames = extract_frames(data, fps=2, tools=tools, temp_dir=tmp_path)
    assert [f.timestamp for f in frames] == [0.0, 0.5, 1.0, 1.5]
    assert frames[0].image.shape == (H, W, 3)
    assert list(tmp_path.iterdir()) == []  # the temporary copy is gone


@pytest.mark.parametrize("name", ["faststart", "fragmented", "mkv", "mpegts"])
def test_remuxed_streams_decode_to_the_same_pixels(media, tools, name):
    assert not needs_seekable_input(media[name], sniff_media_type(media[name]) or MediaType.MP4)
    # Every frame, by index: fragmented MP4 read from a pipe reports other
    # timing metadata, so time-based sampling may pick neighboring frames.
    reference = extract_frames(media["moov_last"], sampling="all", max_frames=12, tools=tools)
    frames = extract_frames(media[name], sampling="all", max_frames=12, tools=tools)
    assert len(frames) == len(reference) == 12
    for ours, theirs in zip(frames, reference, strict=True):
        assert np.array_equal(ours.image, theirs.image)


@pytest.mark.parametrize("name", ["webm", "avi"])
def test_other_codecs_and_containers(media, tools, name):
    info = probe_video(media[name], tools=tools)
    assert (info.width, info.height) == (W, H)
    assert info.duration_seconds == pytest.approx(2.0, abs=0.1)
    frames = extract_frames(media[name], sampling="all", tools=tools)
    assert len(frames) == 50
    assert frames[-1].timestamp == pytest.approx(49 / RATE)


def test_probe_reports_streams(media, tools):
    info = probe_video(media["moov_last"], tools=tools)
    assert info.video_codec == "h264"
    assert info.audio_codec == "aac"
    assert (info.video_streams, info.audio_streams) == (1, 1)
    assert info.frame_rate == 25.0
    assert info.frame_count == 50
    assert info.rotation == 0


def test_rotation_matches_ffmpeg_autorotation(media, tools):
    data = media["rotated"]
    info = probe_video(data, tools=tools)
    assert info.rotation == 270  # display matrix of +90 degrees counterclockwise
    assert (info.display_width, info.display_height) == (H, W)
    ours = extract_frames(data, sampling="all", max_frames=3, tools=tools)
    raw = ffmpeg(
        tools,
        ["-i", "pipe:0", "-frames:v", "3", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdin=media["faststart"],
    )
    stored = np.frombuffer(raw, np.uint8).reshape(3, H, W, 3)
    for frame, original in zip(ours, stored, strict=True):
        assert frame.image.shape == (W, H, 3)
        assert np.array_equal(frame.image, np.rot90(original, k=1))  # 90 degrees counterclockwise


@pytest.mark.parametrize(
    "orientation",
    [
        (90, False, False),
        (270, False, False),
        (0, True, False),  # a phone's "flip" edit
        (90, True, False),
        (270, True, False),
        (180, True, False),
        (0, False, True),
        (90, False, True),
    ],
)
def test_display_matrices_match_ffmpeg_autorotation(media, tools, tmp_path, orientation):
    rotation, hflip, vflip = orientation
    source = tmp_path / "source.mp4"
    source.write_bytes(media["faststart"])
    flags = ["-display_rotation:v:0", str(rotation)]
    flags += ["-display_hflip:v:0"] if hflip else []
    flags += ["-display_vflip:v:0"] if vflip else []
    data = to_file(
        tools,
        [*flags, "-i", str(source), "-c", "copy", "-movflags", "+faststart"],
        tmp_path / "oriented.mp4",
    )
    ours = extract_frames(data, sampling="all", max_frames=3, tools=tools)
    # ffmpeg decoding the same file with its default autorotation.
    raw = ffmpeg(
        tools,
        ["-i", "pipe:0", "-frames:v", "3", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdin=data,
    )
    height, width = ours[0].image.shape[:2]
    reference = np.frombuffer(raw, np.uint8).reshape(3, height, width, 3)
    for frame, expected in zip(ours, reference, strict=True):
        assert np.array_equal(frame.image, expected)


def _index_coded(tools, *, count: int, rate: int, setpts: str | None = None) -> bytes:
    """Lossless video whose frame n is a uniform grey of level n."""
    pixels = np.repeat(np.arange(count, dtype=np.uint8), 16 * 16 * 3)
    args = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "16x16", "-r", str(rate), "-i", "pipe:0"]
    if setpts is not None:
        args += ["-vf", f"setpts='{setpts}'", "-fps_mode", "passthrough"]
    args += ["-c:v", "ffv1", "-pix_fmt", "bgr0", "-f", "matroska", "pipe:1"]
    return ffmpeg(tools, args, stdin=pixels.tobytes())


def _presentation_times(tools, data: bytes) -> list[float]:
    result = tools.run(
        Tool.FFPROBE,
        [
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "matroska",
            "-i",
            "pipe:0",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "csv=p=0",
        ],
        stdin=data,
        timeout_seconds=60,
        max_stdout_bytes=1 << 20,
    )
    times = [float(line) for line in result.stdout.decode().split()]
    return [t - times[0] for t in times]


#: Variable frame rate: 30 fps for 1.5 s, then alternating gaps of 53 and 113 ms.
VFR = "if(lt(N,45),N/30,1.5+(N-45)/12+0.03*mod(N,2))/TB"


@pytest.mark.parametrize(
    ("rate", "fps", "setpts"),
    [
        (30, 1.0, None),
        (30, 2.0, None),
        (30, 7.0, None),
        (25, 3.0, None),
        (30, 1.0, VFR),
        (30, 4.0, VFR),
    ],
)
def test_fps_sampling_returns_the_frame_shown_at_each_timestamp(tools, rate, fps, setpts):
    # The frame labelled t is the last one whose presentation time is <= t:
    # frames 0, 30, 60, ... at 1 fps from 30 fps (round=near gave 14, 44, 74).
    data = _index_coded(tools, count=100, rate=rate, setpts=setpts)
    times = _presentation_times(tools, data)
    frames = extract_frames(data, fps=fps, tools=tools)
    assert len(frames) >= int(times[-1] * fps)
    for frame in frames:
        shown = max(n for n, t in enumerate(times) if t <= frame.timestamp + 1e-6)
        assert round(float(frame.image.mean())) == shown, frame.timestamp


def test_keyframes_and_their_timestamps(media, tools):
    frames = extract_frames(media["faststart"], sampling=FrameSampling.KEYFRAMES, tools=tools)
    assert [round(f.timestamp, 3) for f in frames] == [0.0, 0.4, 0.8, 1.2, 1.6]


def test_streaming_can_stop_early(media, tools):
    iterator = iter_frames(media["mkv"], sampling="all", tools=tools)
    first = next(iterator)
    assert first.index == 0
    iterator.close()  # kills ffmpeg; no error


def test_scaling_options(media, tools):
    (frame,) = extract_frames(media["webm"], max_frames=1, max_side=80, tools=tools)
    assert frame.image.shape == (60, 80, 3)
    (frame,) = extract_frames(media["webm"], max_frames=1, size=(32, 32), tools=tools)
    assert frame.image.shape == (32, 32, 3)


def test_limits_are_enforced_before_decoding(media, tools):
    limits = MediaLimits()
    with pytest.raises(MediaLimitExceededError):
        extract_frames(
            media["mkv"],
            sampling="all",
            tools=tools,
            limits=dataclasses.replace(limits, max_frames=10),
        )
    with pytest.raises(MediaLimitExceededError):
        probe_video(media["mkv"], tools=tools, limits=dataclasses.replace(limits, max_pixels=1000))
    with pytest.raises(MediaLimitExceededError):
        probe_video(
            media["faststart"],
            tools=tools,
            limits=dataclasses.replace(limits, max_duration_seconds=1.0),
        )
    # Matroska written to a pipe declares no duration: the decoder catches it.
    assert probe_video(media["mkv"], tools=tools).duration_seconds is None
    with pytest.raises(MediaLimitExceededError) as info:
        extract_frames(
            media["mkv"],
            fps=5,
            tools=tools,
            limits=dataclasses.replace(limits, max_duration_seconds=1.0),
        )
    assert info.value.limit == "max_duration_seconds"
    with pytest.raises(MediaLimitExceededError):
        probe_video(media["mkv"], tools=tools, limits=dataclasses.replace(limits, max_bytes=100))


def test_placeholder_headers_are_refused(media, tools):
    with pytest.raises(MediaLimitExceededError):
        probe_video(media["avi_from_pipe"], tools=tools)


def test_audio_only_media_has_no_video(media, tools):
    with pytest.raises(UnsupportedMediaError):
        probe_video(media["audio_only"], tools=tools)


def test_truncated_and_corrupt_media(media, tools, rng):
    with pytest.raises(MediaDecodeError):
        probe_video(media["moov_last"][: len(media["moov_last"]) // 2], tools=tools)
    corrupt = media["mkv"][:64] + rng.bytes(4096)
    with pytest.raises(MediaDecodeError):
        extract_frames(corrupt, tools=tools)


def test_playlists_and_unknown_bytes_are_refused_before_any_tool_runs(tools):
    playlist = b"#EXTM3U\n#EXTINF:1,\nfile:///etc/passwd\n"
    with pytest.raises(UnsupportedMediaError):
        probe_video(playlist, tools=tools)
    concat = b"ffconcat version 1.0\nfile '/etc/passwd'\n"
    with pytest.raises(UnsupportedMediaError):
        extract_frames(concat, tools=tools)
