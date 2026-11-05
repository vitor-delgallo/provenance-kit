"""Probing and frame extraction against the real ffmpeg/ffprobe binaries."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from provenance_kit.errors import (
    MediaDecodeError,
    MediaLimitExceededError,
    UnsupportedMediaError,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType, sniff_media_type
from provenance_kit.tools import ExternalTools, Tool
from provenance_kit.video._ffmpeg import needs_seekable_input
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
