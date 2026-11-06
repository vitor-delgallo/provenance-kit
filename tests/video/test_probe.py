"""Unit tests of video probing: hostile ffprobe reports and input staging."""

from __future__ import annotations

import dataclasses
import json

import pytest

from provenance_kit.errors import (
    MediaDecodeError,
    MediaLimitExceededError,
    UnsupportedMediaError,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType
from provenance_kit.video.probe import MAX_STREAMS, VideoInfo, _parse_probe_output, probe_video
from tests.support import assert_imports_lightly
from tests.video.fakes import MP4_MDAT_FIRST, MP4_MOOV_FIRST, WEBM, FakeTools, probe_json

LIMITS = MediaLimits()


def parse(raw: bytes, limits: MediaLimits = LIMITS) -> VideoInfo:
    return _parse_probe_output(raw, MediaType.MP4, limits)


def test_video_modules_import_lightly():
    assert_imports_lightly(
        [
            "provenance_kit.video.probe",
            "provenance_kit.video.frames",
            "provenance_kit.video.fingerprint",
            "provenance_kit.video.watermark",
        ]
    )


def test_parses_a_normal_report():
    info = parse(probe_json())
    assert (info.width, info.height, info.rotation) == (8, 6, 0)
    assert info.frame_rate == 25.0
    assert info.frame_count == 50
    assert info.duration_seconds == 2.0
    assert info.video_codec == "h264"
    assert info.audio_codec == "aac"
    assert (info.video_streams, info.audio_streams, info.other_streams) == (1, 1, 0)
    assert info.stream_index == 0


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not json",
        b"[1, 2, 3]",
        b'{"streams": NaN}',
        b'{"streams": [{"width": Infinity}]}',
        b"\xff\xfe garbage",
        b"[" * 100_000 + b"]" * 100_000,
        b'{"streams": "nope"}',
        b'{"format": {}}',
    ],
)
def test_unusable_reports_are_decode_errors(raw):
    with pytest.raises(MediaDecodeError):
        parse(raw)


def test_no_video_stream_is_unsupported():
    raw = json.dumps({"streams": [{"index": 0, "codec_type": "audio"}], "format": {}}).encode()
    with pytest.raises(UnsupportedMediaError, match="no video stream"):
        parse(raw)


def test_cover_art_is_not_a_video_stream():
    raw = probe_json(disposition={"attached_pic": 1})
    with pytest.raises(UnsupportedMediaError):
        parse(raw)


def test_too_many_streams_are_refused():
    streams = [{"index": i, "codec_type": "data"} for i in range(MAX_STREAMS + 1)]
    with pytest.raises(UnsupportedMediaError, match="too many streams"):
        parse(json.dumps({"streams": streams}).encode())


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": None},
        {"width": 0},
        {"width": "abc"},
        {"width": True},
        {"height": -6},
        {"width": "9" * 40},
        {"index": 5000},
        {"index": "x"},
    ],
)
def test_missing_or_invalid_fields_are_decode_errors(overrides):
    with pytest.raises(MediaDecodeError):
        parse(probe_json(**overrides))


def test_numeric_strings_are_accepted():
    info = parse(probe_json(width="16", height="10"))
    assert (info.width, info.height) == (16, 10)


@pytest.mark.parametrize(
    ("overrides", "limits", "field"),
    [
        ({"width": 20_000}, LIMITS, "max_width"),
        ({"width": 9000, "height": 9000}, LIMITS, "max_pixels"),
        ({"nb_frames": "1000000"}, LIMITS, "max_frames"),
        ({"avg_frame_rate": "5000/1"}, dataclasses.replace(LIMITS, max_frames=1000), "max_frames"),
    ],
)
def test_declared_values_are_checked_against_limits(overrides, limits, field):
    with pytest.raises(MediaLimitExceededError) as info:
        parse(probe_json(**overrides), limits)
    assert info.value.limit == field


def test_declared_duration_is_checked_against_limits():
    report = json.loads(probe_json())
    report["format"]["duration"] = "99999"
    with pytest.raises(MediaLimitExceededError) as info:
        parse(json.dumps(report).encode(), dataclasses.replace(LIMITS, max_duration_seconds=10.0))
    assert info.value.limit == "max_duration_seconds"


@pytest.mark.parametrize("value", ["N/A", "", "-3", "nan", None, [], {"a": 1}])
def test_bad_durations_are_ignored(value):
    raw = json.loads(probe_json(duration=value))
    raw["format"]["duration"] = value  # both the container and the stream
    info = parse(json.dumps(raw).encode())
    assert info.duration_seconds is None


@pytest.mark.parametrize(
    ("rate", "expected"),
    [("30000/1001", 30000 / 1001), ("0/0", None), ("25", None), ("-5/1", None), ("1/0", None)],
)
def test_frame_rates(rate, expected):
    info = parse(probe_json(avg_frame_rate=rate, r_frame_rate="0/0"))
    assert info.frame_rate == expected


def test_absurd_frame_rates_are_dropped():
    info = parse(probe_json(avg_frame_rate="1000000/1", r_frame_rate="1000000/1"))
    assert info.frame_rate is None


@pytest.mark.parametrize(
    ("side_data", "tags", "expected"),
    [
        ([{"side_data_type": "Display Matrix", "rotation": -90}], {}, 90),
        ([{"rotation": 90}], {}, 270),
        ([{"rotation": 180}], {}, 180),
        ([{"rotation": -180.0}], {}, 180),
        ([{"rotation": 45}], {}, 0),
        ([{"rotation": "abc"}], {"rotate": "90"}, 90),
        ([], {"rotate": "270"}, 270),
        ("garbage", {"rotate": 12345678}, 0),
        ([{"rotation": 1e300}], {}, 0),
        ([{"rotation": int("9" * 400)}], {}, 0),  # too large for a float: ignored
    ],
)
def test_rotation(side_data, tags, expected):
    info = parse(probe_json(side_data_list=side_data, tags=tags))
    assert info.rotation == expected
    if expected in (90, 270):
        assert (info.display_width, info.display_height) == (6, 8)


@pytest.mark.parametrize("name", ["h264; rm -rf /", "x" * 100, 7, "../etc"])
def test_unsafe_codec_names_are_replaced(name):
    assert parse(probe_json(codec_name=name)).video_codec == "unknown"


def display_matrix(a: int, b: int, c: int, d: int) -> str:
    """ffprobe's text form of a display matrix (16.16 fixed point, as printed)."""
    rows = ((a, b, 0), (c, d, 0), (0, 0, 1 << 30))
    return "\n" + "".join(
        f"{index:08x}: {x:11d} {y:11d} {z:11d}\n" for index, (x, y, z) in enumerate(rows)
    )


ONE = 1 << 16

# (ffprobe rotation, matrix) -> (clockwise rotation, mirrored), as ffmpeg's own
# autorotation transforms the frames (measured with ffmpeg 8.1 for every
# -display_rotation / -display_hflip / -display_vflip combination).
DISPLAY_MATRICES = [
    (-180, (-ONE, 0, 0, ONE), (0, True)),  # -display_hflip
    (0, (ONE, 0, 0, -ONE), (180, True)),  # -display_vflip
    (90, (0, -ONE, -ONE, 0), (90, True)),  # rotation 90 + hflip
    (-90, (0, ONE, ONE, 0), (270, True)),  # rotation 270 + hflip
    (-180, (-ONE, 0, 0, -ONE), (180, False)),  # rotation 180 (or hflip + vflip)
    (90, (0, -ONE, ONE, 0), (270, False)),  # rotation 90
    (-90, (0, ONE, -ONE, 0), (90, False)),  # rotation 270
]


@pytest.mark.parametrize(("rotation", "matrix", "expected"), DISPLAY_MATRICES)
def test_mirrored_display_matrices(rotation, matrix, expected):
    entry = {
        "side_data_type": "Display Matrix",
        "displaymatrix": display_matrix(*matrix),
        "rotation": rotation,
    }
    info = parse(probe_json(side_data_list=[entry]))
    assert (info.rotation, info.mirrored) == expected


@pytest.mark.parametrize(("rotation", "matrix", "expected"), DISPLAY_MATRICES)
def test_display_matrix_without_a_rotation_field(rotation, matrix, expected):
    entry = {"side_data_type": "Display Matrix", "displaymatrix": display_matrix(*matrix)}
    info = parse(probe_json(side_data_list=[entry]))
    assert (info.rotation, info.mirrored) == expected


@pytest.mark.parametrize(
    "text",
    [
        "garbage",
        display_matrix(0, 0, 0, 0),  # degenerate
        "\n00000000: 1 2\n",  # too few values
        "x" * 10_000,
        display_matrix(-ONE, 0, 0, ONE) + "00000003: 1 2 3\n",  # too many values
        "\n00000000: 99999999999 0 0\n00000001: 0 1 0\n00000002: 0 0 1\n",  # not int32
        7,
    ],
)
def test_unusable_display_matrices_are_ignored(text):
    entry = {"displaymatrix": text, "rotation": -90}
    info = parse(probe_json(side_data_list=[entry]))
    assert (info.rotation, info.mirrored) == (90, False)


def test_legacy_rotate_tag_is_never_mirrored():
    info = parse(probe_json(tags={"rotate": "90"}))
    assert (info.rotation, info.mirrored) == (90, False)


def test_video_info_validates_rotation():
    info = parse(probe_json())
    with pytest.raises(ValueError):
        dataclasses.replace(info, rotation=45)
    with pytest.raises(TypeError):
        dataclasses.replace(info, mirrored=1)


# ------------------------------------------------------------ probe_video


def test_probe_video_passes_safe_arguments():
    tools = FakeTools(ffprobe=probe_json())
    info = probe_video(MP4_MOOV_FIRST, tools=tools)  # type: ignore[arg-type]
    assert info.width == 8
    (args,) = tools.args_of("ffprobe")
    assert "-nostdin" not in args  # ffprobe rejects it
    assert args[args.index("-protocol_whitelist") + 1] == "pipe"
    assert args[args.index("-f") + 1] == "mov"
    assert args[args.index("-i") + 1] == "pipe:0"
    assert tools.calls[0][2]["max_stdout_bytes"] <= 1024 * 1024


def test_probe_video_reads_a_trailing_moov_mp4_from_a_private_file(tmp_path):
    tools = FakeTools(ffprobe=probe_json())
    probe_video(MP4_MDAT_FIRST, tools=tools, temp_dir=tmp_path)  # type: ignore[arg-type]
    ((_, args, kwargs),) = tools.calls
    assert args[args.index("-protocol_whitelist") + 1] == "file,pipe"
    assert args[args.index("-i") + 1].startswith("file:")
    assert kwargs["stdin"] is None
    assert list(tmp_path.iterdir()) == []  # deleted after the call


def test_probe_video_refuses_non_video_bytes():
    tools = FakeTools(ffprobe=probe_json())
    with pytest.raises(UnsupportedMediaError):
        probe_video(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, tools=tools)  # type: ignore[arg-type]
    with pytest.raises(UnsupportedMediaError):
        probe_video(b"random bytes" * 10, tools=tools)  # type: ignore[arg-type]
    assert tools.calls == []


def test_probe_video_checks_the_byte_limit_first():
    tools = FakeTools(ffprobe=probe_json())
    with pytest.raises(MediaLimitExceededError):
        probe_video(
            MP4_MOOV_FIRST,
            tools=tools,  # type: ignore[arg-type]
            limits=dataclasses.replace(LIMITS, max_bytes=10),
        )
    assert tools.calls == []


def test_probe_video_rejects_non_bytes():
    with pytest.raises(TypeError):
        probe_video("video.mp4")  # type: ignore[arg-type]


def test_ffprobe_failures_become_decode_errors():
    tools = FakeTools(ffprobe_returncode=1)
    with pytest.raises(MediaDecodeError, match="exit status 1"):
        probe_video(WEBM, tools=tools)  # type: ignore[arg-type]
