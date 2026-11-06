"""The shared ffmpeg/ffprobe plumbing: input staging and defensive report parsing."""

from __future__ import annotations

import dataclasses

import pytest

from provenance_kit._ffmpeg import (
    DEMUXERS,
    FFMPEG_COMMON,
    FFPROBE_COMMON,
    as_media_error,
    check_media_input,
    json_float,
    json_int,
    json_name,
    json_rate,
    load_json_object,
    needs_seekable_input,
    run_ffprobe_json,
    staged_input,
)
from provenance_kit.errors import (
    ExternalToolError,
    ExternalToolTimeoutError,
    MediaDecodeError,
    MediaLimitExceededError,
    UnsupportedMediaError,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType
from provenance_kit.tools import Tool, ToolResult


def box(kind: bytes, payload: bytes = b"", size: int | None = None) -> bytes:
    length = 8 + len(payload) if size is None else size
    return length.to_bytes(4, "big") + kind + payload


FTYP = box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
MOOV_FIRST = FTYP + box(b"moov") + box(b"mdat")
MDAT_FIRST = FTYP + box(b"mdat") + box(b"moov")
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64


def test_common_arguments():
    assert FFMPEG_COMMON == ("-nostdin", "-hide_banner", "-loglevel", "error")
    # ffprobe rejects -nostdin.
    assert FFPROBE_COMMON == ("-hide_banner", "-loglevel", "error")


def test_every_demuxer_is_a_plain_container_demuxer():
    assert DEMUXERS[MediaType.MP4] == DEMUXERS[MediaType.M4A] == "mov"
    assert DEMUXERS[MediaType.WEBM] == "matroska"
    assert not {"hls", "concat", "applehttp", "image2"} & set(DEMUXERS.values())
    with pytest.raises(TypeError):
        DEMUXERS[MediaType.JPEG] = "image2"  # type: ignore[index]


# ------------------------------------------------------------ input checks and staging


def test_check_media_input():
    assert check_media_input(MOOV_FIRST, MediaLimits(), {MediaType.MP4}) is MediaType.MP4
    with pytest.raises(UnsupportedMediaError):
        check_media_input(WAV, MediaLimits(), {MediaType.MP4})
    with pytest.raises(MediaLimitExceededError):
        check_media_input(WAV, dataclasses.replace(MediaLimits(), max_bytes=8), {MediaType.WAV})
    with pytest.raises(TypeError):
        check_media_input("text", MediaLimits(), {MediaType.WAV})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (FTYP + box(b"moov") + box(b"mdat"), False),
        (FTYP + box(b"free") + box(b"mdat", b"x" * 10) + box(b"moov"), True),
        (FTYP + box(b"moof") + box(b"mdat"), False),
        (FTYP + box(b"mdat", size=0) + b"\x00" * 64, True),
        (
            FTYP
            + (1).to_bytes(4, "big")
            + b"free"
            + (24).to_bytes(8, "big")
            + b"\x00" * 8
            + box(b"moov"),
            False,
        ),
        (FTYP + box(b"free", size=3), True),
        (FTYP + b"\x00\x00", True),
        (FTYP + box(b"free") * 5000, True),
    ],
)
def test_needs_seekable_input(data, expected):
    assert needs_seekable_input(data, MediaType.MP4) is expected
    assert needs_seekable_input(data, MediaType.M4A) is expected


def test_non_bmff_containers_always_stream():
    assert needs_seekable_input(MDAT_FIRST, MediaType.WEBM) is False
    assert needs_seekable_input(WAV, MediaType.WAV) is False


def test_staged_input_uses_a_pipe_or_a_private_temporary_file(tmp_path):
    with staged_input(MOOV_FIRST, MediaType.MP4, temp_dir=tmp_path) as staged:
        assert staged.target == "pipe:0"
        assert not staged.uses_file
        assert staged.input_args()[:4] == ["-protocol_whitelist", "pipe", "-f", "mov"]
        assert staged.input_args("-noautorotate")[-3:] == ["-noautorotate", "-i", "pipe:0"]
    with staged_input(MDAT_FIRST, MediaType.M4A, temp_dir=tmp_path) as staged:
        assert staged.stdin is None
        assert staged.uses_file
        assert staged.target.startswith("file:")
        assert staged.input_args()[1] == "file,pipe"
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].read_bytes() == MDAT_FIRST
    assert list(tmp_path.iterdir()) == []
    with staged_input(WAV, MediaType.WAV, temp_dir=tmp_path) as staged:
        assert staged.demuxer == "wav"
        assert bytes(staged.stdin or b"") == WAV


# ------------------------------------------------------------ tool failures


def test_as_media_error_classifies_failures():
    decode = as_media_error(ExternalToolError("ffmpeg", "failed", returncode=1), "decoding")
    assert isinstance(decode, MediaDecodeError)
    assert "exit status 1" in str(decode)
    timeout = ExternalToolTimeoutError("ffmpeg", 5)
    assert as_media_error(timeout, "decoding") is timeout
    overflow = ExternalToolError("ffmpeg", "output exceeded")
    assert as_media_error(overflow, "decoding") is overflow


class FakeProbe:
    def __init__(self, stdout: bytes = b"{}", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.calls: list[tuple[str, list[str], dict]] = []

    def run(self, tool, args, **kwargs):
        self.calls.append((Tool(tool).value, list(args), kwargs))
        if self.returncode:
            raise ExternalToolError("ffprobe", "failed", returncode=self.returncode)
        return ToolResult("ffprobe", 0, self.stdout, b"", 0.0)


def test_run_ffprobe_json_builds_the_arguments_and_parses_the_report(tmp_path):
    tools = FakeProbe(b'{"streams": []}')
    with staged_input(WAV, MediaType.WAV, temp_dir=tmp_path) as staged:
        report = run_ffprobe_json(
            staged,
            ["-print_format", "json"],
            tools=tools,  # type: ignore[arg-type]
            what="probe",
            timeout_seconds=5,
            max_output_bytes=1000,
            before_input=("-skip_frame", "nokey"),
        )
    assert report == {"streams": []}
    ((name, args, kwargs),) = tools.calls
    assert name == "ffprobe"
    assert args == [
        *FFPROBE_COMMON,
        "-protocol_whitelist",
        "pipe",
        "-f",
        "wav",
        "-skip_frame",
        "nokey",
        "-i",
        "pipe:0",
        "-print_format",
        "json",
    ]
    assert kwargs["max_stdout_bytes"] == 1000
    assert kwargs["timeout_seconds"] == 5


def test_run_ffprobe_json_turns_failures_into_media_errors(tmp_path):
    with staged_input(WAV, MediaType.WAV, temp_dir=tmp_path) as staged:
        for tools in (FakeProbe(returncode=1), FakeProbe(b"[1, 2]"), FakeProbe(b"\xff")):
            with pytest.raises(MediaDecodeError):
                run_ffprobe_json(
                    staged,
                    [],
                    tools=tools,  # type: ignore[arg-type]
                    what="probe",
                    timeout_seconds=5,
                    max_output_bytes=1000,
                )


# ------------------------------------------------------------ defensive JSON


@pytest.mark.parametrize(
    "raw",
    [b"", b"not json", b"[]", b'"text"', b'{"a": NaN}', b'{"a": -Infinity}', b"\xff\xfe"],
)
def test_load_json_object_refuses_anything_but_a_finite_object(raw):
    with pytest.raises(MediaDecodeError):
        load_json_object(raw, "report")


def test_load_json_object_refuses_deep_nesting():
    with pytest.raises(MediaDecodeError):
        load_json_object(b"[" * 100_000 + b"]" * 100_000, "report")


def test_json_int():
    assert json_int(7) == 7
    assert json_int(" 12 ") == 12
    for bad in (True, 1.5, "1.5", "", "9" * 40, None, [1]):
        assert json_int(bad) is None


def test_json_float():
    assert json_float(2) == 2.0
    assert json_float("1.25") == 1.25
    for bad in (False, "nan", "inf", float("nan"), "", "1" * 40, None, {}):
        assert json_float(bad) is None


def test_json_float_ignores_integers_too_large_for_a_float():
    # json.loads returns an int of any size; float() of one above ~1.8e308
    # raises OverflowError, which must not escape the report parser.
    assert json_float(10**308) == 1e308
    assert json_float(int("9" * 400)) is None
    assert json_float(-(10**400)) is None


def test_json_rate():
    assert json_rate("30000/1001", maximum=1000) == pytest.approx(29.97, abs=0.01)
    assert json_rate("25:1", maximum=1000) == 25.0
    for bad in ("0/1", "1/0", "-25/1", "25", 25, "2000/1", "1" * 20 + "/1"):
        assert json_rate(bad, maximum=1000) is None


def test_json_name():
    assert json_name("h264") == "h264"
    assert json_name("mov,mp4,m4a") == "mov,mp4,m4a"
    for bad in ("", "a b", "x" * 65, "<script>", 3, None):
        assert json_name(bad) is None
