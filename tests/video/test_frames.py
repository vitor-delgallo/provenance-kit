"""Unit tests of frame decoding with fake tools (no ffmpeg needed)."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from provenance_kit.errors import (
    MediaDecodeError,
    MediaLimitExceededError,
    UnsupportedMediaError,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType
from provenance_kit.video.frames import FrameSampling, extract_frames, iter_frames, output_size
from provenance_kit.video.probe import parse_probe_output
from tests.video.fakes import MP4_MDAT_FIRST, MP4_MOOV_FIRST, FakeTools, probe_json

W, H = 8, 6
FRAME = W * H * 3


def frames_bytes(count: int, width: int = W, height: int = H) -> bytes:
    return b"".join(bytes([i % 256]) * (width * height * 3) for i in range(count))


def run(tools: FakeTools, **kwargs) -> list:
    return extract_frames(MP4_MOOV_FIRST, tools=tools, **kwargs)  # type: ignore[arg-type]


def test_fps_sampling_reads_frames_with_exact_timestamps():
    tools = FakeTools(ffprobe=probe_json(), ffmpeg=frames_bytes(4))
    frames = run(tools, fps=2.0)
    assert [f.index for f in frames] == [0, 1, 2, 3]
    assert [f.timestamp for f in frames] == [0.0, 0.5, 1.0, 1.5]
    first = frames[0].image
    assert first.shape == (H, W, 3)
    assert first.dtype == np.uint8
    assert not first.flags.writeable
    assert int(frames[3].image[0, 0, 0]) == 3
    (args,) = tools.args_of("ffmpeg")
    assert args[:4] == ["-nostdin", "-hide_banner", "-loglevel", "error"]
    assert args[args.index("-protocol_whitelist") + 1] == "pipe"
    assert "-noautorotate" in args
    vf = args[args.index("-vf") + 1]
    assert vf == f"setpts=PTS-STARTPTS,fps=2.0,scale={W}:{H}:flags=bicubic,format=rgb24"
    assert args[args.index("-map") + 1] == "0:0"
    assert args[-1] == "pipe:1"


def test_frames_are_streamed_lazily():
    tools = FakeTools(ffprobe=probe_json(), ffmpeg=frames_bytes(3))
    iterator = iter_frames(MP4_MOOV_FIRST, tools=tools)  # type: ignore[arg-type]
    assert tools.calls == []  # nothing runs before the first next()
    assert next(iterator).index == 0
    iterator.close()


def test_max_frames_stops_early_without_error():
    tools = FakeTools(ffprobe=probe_json(), ffmpeg=frames_bytes(10))
    frames = run(tools, max_frames=3)
    assert len(frames) == 3
    (args,) = tools.args_of("ffmpeg")
    assert args[args.index("-frames:v") + 1] == "3"


def test_more_frames_than_the_limit_raise():
    limits = dataclasses.replace(MediaLimits(), max_frames=2)
    report = json.loads(probe_json(nb_frames=None, duration=None))
    report["format"].pop("duration")
    tools = FakeTools(ffprobe=json.dumps(report).encode(), ffmpeg=frames_bytes(3))
    with pytest.raises(MediaLimitExceededError) as info:
        run(tools, limits=limits)
    assert info.value.limit == "max_frames"


def test_media_running_past_the_duration_limit_raises():
    limits = dataclasses.replace(MediaLimits(), max_duration_seconds=2.0)
    report = json.loads(probe_json(duration=None))
    report["format"].pop("duration")
    tools = FakeTools(ffprobe=json.dumps(report).encode(), ffmpeg=frames_bytes(5))
    with pytest.raises(MediaLimitExceededError) as info:
        run(tools, limits=limits)  # 1 fps: the frame at 4 s is past 2 s + 1 interval
    assert info.value.limit == "max_duration_seconds"
    (args,) = tools.args_of("ffmpeg")
    assert float(args[args.index("-t") + 1]) == 3.0


def test_expected_frames_are_checked_before_decoding():
    limits = dataclasses.replace(MediaLimits(), max_frames=10)
    tools = FakeTools(ffprobe=probe_json(nb_frames="5"), ffmpeg=frames_bytes(1))
    with pytest.raises(MediaLimitExceededError):
        run(tools, fps=100.0, limits=limits)  # 2 s at 100 fps = 200 frames
    assert tools.args_of("ffmpeg") == []


def test_output_size_is_checked_against_limits():
    limits = dataclasses.replace(MediaLimits(), max_pixels=100)
    tools = FakeTools(ffprobe=probe_json(), ffmpeg=b"")
    with pytest.raises(MediaLimitExceededError):
        run(tools, size=(20, 20), limits=limits)


def test_truncated_output_is_a_decode_error():
    tools = FakeTools(ffprobe=probe_json(), ffmpeg=frames_bytes(2)[:-5])
    with pytest.raises(MediaDecodeError, match="middle of a frame"):
        run(tools)


def test_ffmpeg_failure_is_a_decode_error():
    tools = FakeTools(ffprobe=probe_json(), ffmpeg=frames_bytes(1), ffmpeg_returncode=1)
    with pytest.raises(MediaDecodeError, match="exit status 1"):
        run(tools)


def test_all_frames_use_the_nominal_frame_rate():
    tools = FakeTools(ffprobe=probe_json(avg_frame_rate="10/1"), ffmpeg=frames_bytes(3))
    frames = run(tools, sampling=FrameSampling.ALL)
    assert [f.timestamp for f in frames] == [0.0, 0.1, 0.2]
    (args,) = tools.args_of("ffmpeg")
    assert "fps=" not in args[args.index("-vf") + 1]


def test_all_frames_need_a_frame_rate():
    report = probe_json(avg_frame_rate="0/0", r_frame_rate="0/0")
    tools = FakeTools(ffprobe=report, ffmpeg=frames_bytes(1))
    with pytest.raises(UnsupportedMediaError, match="frame rate"):
        run(tools, sampling="all")


def test_keyframes_take_timestamps_from_ffprobe():
    listing = json.dumps(
        {
            "frames": [
                {"best_effort_timestamp_time": "0.500000"},
                {"best_effort_timestamp_time": "N/A"},
                {"best_effort_timestamp_time": "4.500000"},
                "garbage",
            ]
        }
    ).encode()

    def ffprobe(args: list[str]) -> bytes:
        return (
            listing if "frame=best_effort_timestamp_time" in args else probe_json(start_time="0.5")
        )

    tools = FakeTools(ffprobe=ffprobe, ffmpeg=frames_bytes(5))
    frames = run(tools, sampling=FrameSampling.KEYFRAMES)
    assert [f.timestamp for f in frames] == [0.0, 0.0, 4.0, 4.0]
    listing_args = tools.args_of("ffprobe")[1]
    assert listing_args[listing_args.index("-skip_frame") + 1] == "nokey"
    (args,) = tools.args_of("ffmpeg")
    assert args[args.index("-skip_frame") + 1] == "nokey"


def test_too_many_keyframes_raise():
    limits = dataclasses.replace(MediaLimits(), max_frames=2)
    listing = json.dumps({"frames": [{"best_effort_timestamp_time": "0"}] * 3}).encode()

    def ffprobe(args: list[str]) -> bytes:
        return listing if "-skip_frame" in args else probe_json(nb_frames="2", duration="0.08")

    tools = FakeTools(ffprobe=ffprobe, ffmpeg=b"")
    with pytest.raises(MediaLimitExceededError):
        run(tools, sampling="keyframes", limits=limits)


@pytest.mark.parametrize(
    ("rotation", "filters", "shape"),
    [(-90, "transpose=clock", (W, H, 3)), (180, "hflip,vflip", (H, W, 3))],
)
def test_rotation_is_applied_in_the_filter_graph(rotation, filters, shape):
    report = probe_json(side_data_list=[{"rotation": rotation}])
    info = parse_probe_output(report, MediaType.MP4, MediaLimits())
    width, height = output_size(info)
    tools = FakeTools(ffprobe=report, ffmpeg=frames_bytes(1, width, height))
    (frame,) = run(tools)
    assert frame.image.shape == shape
    (args,) = tools.args_of("ffmpeg")
    assert args[args.index("-vf") + 1].startswith(filters + ",setpts=PTS-STARTPTS,fps=")


def test_seekable_media_is_read_from_a_private_temporary_file(tmp_path):
    tools = FakeTools(ffprobe=probe_json(), ffmpeg=frames_bytes(1))
    frames = extract_frames(MP4_MDAT_FIRST, tools=tools, temp_dir=tmp_path)  # type: ignore[arg-type]
    assert len(frames) == 1
    for _, args, kwargs in tools.calls:
        assert args[args.index("-protocol_whitelist") + 1] == "file,pipe"
        target = args[args.index("-i") + 1]
        assert target.startswith("file:")
        assert str(tmp_path) in target
        assert kwargs["stdin"] is None
    assert list(tmp_path.iterdir()) == []


def test_output_size_options():
    info = parse_probe_output(probe_json(width=1920, height=1080), MediaType.MP4, MediaLimits())
    assert output_size(info) == (1920, 1080)
    assert output_size(info, max_side=256) == (256, 144)
    assert output_size(info, max_side=4000) == (1920, 1080)
    assert output_size(info, size=(64, 64)) == (64, 64)
    with pytest.raises(ValueError):
        output_size(info, size=(64, 64), max_side=10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 0},
        {"fps": -1.0},
        {"fps": float("nan")},
        {"fps": 1000.0},
        {"max_frames": 0},
        {"size": (0, 10)},
        {"size": (10,)},
        {"max_side": 0},
        {"size": (8, 8), "max_side": 8},
        {"sampling": "every-other"},
    ],
)
def test_invalid_arguments_raise_at_call_time(kwargs):
    with pytest.raises(ValueError):
        iter_frames(MP4_MOOV_FIRST, tools=FakeTools(), **kwargs)  # type: ignore[arg-type]


def test_wrong_argument_types_raise():
    with pytest.raises(TypeError):
        iter_frames(MP4_MOOV_FIRST, fps="1")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        iter_frames(MP4_MOOV_FIRST, max_frames=True)  # type: ignore[arg-type]
