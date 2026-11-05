"""Probe untrusted video with ffprobe before decoding it.

:func:`probe_video` returns a :class:`VideoInfo`: duration, dimensions,
rotation, frame rate, codecs and stream counts of the first video stream
(cover-art pictures are skipped). It enforces
:class:`~provenance_kit.limits.MediaLimits` on what the container declares,
so that :mod:`provenance_kit.video.frames` can refuse a decompression bomb
before decoding a single frame.

Both the media and ffprobe's report are treated as untrusted:

* the container is detected from the bytes and passed to ffprobe explicitly
  (``-f mov``, ``-f matroska``, ...), with ``-protocol_whitelist pipe`` (or
  ``file,pipe`` for a temporary file), so the media cannot make ffprobe open
  URLs or other files;
* the media goes through a pipe, except ISO base media files whose ``moov``
  box follows the media data, which need a seekable private temporary file
  (see :mod:`provenance_kit._ffmpeg`);
* ffprobe is asked for the few fields used here, its output is capped at
  1 MiB, and the JSON is parsed defensively: wrong types, missing fields,
  ``NaN``/``Infinity``, absurd values and too many streams are refused or
  ignored, never trusted.

Example::

    from provenance_kit.video.probe import probe_video

    info = probe_video(data)  # data: bytes of an uploaded video
    print(info.duration_seconds, info.display_width, info.display_height)

Thread safety: the functions are thread-safe; each call runs its own ffprobe
process.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from provenance_kit._ffmpeg import (
    StagedInput,
    check_media_input,
    json_float,
    json_int,
    json_name,
    json_rate,
    load_json_object,
    run_ffprobe_json,
    staged_input,
)
from provenance_kit._logging import get_logger
from provenance_kit._typing import BytesLike, StrPath
from provenance_kit.errors import MediaDecodeError, UnsupportedMediaError
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType
from provenance_kit.tools import ExternalTools

__all__ = ["MAX_STREAMS", "VideoInfo", "probe_video"]

_log = get_logger(__name__)

#: Containers declaring more streams than this are refused.
MAX_STREAMS = 64

#: The containers the video modules accept (detected from the bytes).
VIDEO_CONTAINERS: frozenset[MediaType] = frozenset(
    {
        MediaType.MP4,
        MediaType.QUICKTIME,
        MediaType.WEBM,
        MediaType.MATROSKA,
        MediaType.AVI,
        MediaType.MPEG_TS,
    }
)

_MAX_PROBE_OUTPUT = 1024 * 1024
_MAX_FRAME_RATE = 10_000.0

_N = TypeVar("_N", int, float)

_PROBE_ENTRIES = (
    "format=format_name,duration,bit_rate,nb_streams"
    ":stream=index,codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,"
    "r_frame_rate,nb_frames,duration,start_time"
    ":stream_disposition=attached_pic"
    ":stream_tags=rotate"
    ":stream_side_data=rotation,displaymatrix"
)
_MAX_DISPLAY_MATRIX_TEXT = 512  # ffprobe prints about 130 characters
_INT32 = 1 << 31


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """What ffprobe reports about a video, after validation.

    Dimensions are those of the stored (coded) frames. The display transform
    is: mirror the frames horizontally if ``mirrored``, then turn them
    clockwise by ``rotation``. Every orientation a display matrix can encode
    (the four rotations, each with or without a mirror, such as a phone's
    "flip" edit) is one such pair. The ``display_width`` / ``display_height``
    properties account for it, and :mod:`provenance_kit.video.frames`
    applies it exactly as ffmpeg's own autorotation does.

    Attributes:
        container: The container detected from the bytes.
        format_name: ffprobe's demuxer name list, for example
            ``"mov,mp4,m4a,3gp,3g2,mj2"``.
        duration_seconds: The declared duration, or ``None`` if unknown.
        width: Stored frame width in pixels.
        height: Stored frame height in pixels.
        rotation: Clockwise display rotation: 0, 90, 180 or 270 degrees.
        frame_rate: Average frame rate in frames per second, or ``None``.
        frame_count: Declared number of frames of the stream, or ``None``.
        video_codec: Codec of the selected video stream, for example
            ``"h264"``.
        pixel_format: Pixel format, for example ``"yuv420p"``, or ``None``.
        audio_codec: Codec of the first audio stream, or ``None``.
        video_streams: Number of video streams (cover-art pictures included).
        audio_streams: Number of audio streams.
        subtitle_streams: Number of subtitle streams.
        other_streams: Number of data, attachment and unknown streams.
        bit_rate: Declared overall bit rate in bits per second, or ``None``.
        stream_index: ffprobe index of the selected video stream (used to map
            it in ffmpeg).
        start_time: Start time of the selected stream in seconds (0.0 when
            not declared).
        mirrored: Whether the display matrix mirrors the frames (negative
            determinant); the horizontal flip is applied before ``rotation``.
    """

    container: MediaType
    format_name: str
    duration_seconds: float | None
    width: int
    height: int
    rotation: int
    frame_rate: float | None
    frame_count: int | None
    video_codec: str
    pixel_format: str | None
    audio_codec: str | None
    video_streams: int
    audio_streams: int
    subtitle_streams: int
    other_streams: int
    bit_rate: int | None
    stream_index: int
    start_time: float = 0.0
    mirrored: bool = False

    def __post_init__(self) -> None:
        """Validates the fields."""
        if self.width < 1 or self.height < 1:
            raise ValueError("width and height must be positive")
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError("rotation must be 0, 90, 180 or 270")
        if not isinstance(self.mirrored, bool):
            raise TypeError("mirrored must be a bool")

    @property
    def display_width(self) -> int:
        """Width of the frames as displayed (after rotation)."""
        return self.height if self.rotation in (90, 270) else self.width

    @property
    def display_height(self) -> int:
        """Height of the frames as displayed (after rotation)."""
        return self.width if self.rotation in (90, 270) else self.height


def probe_video(
    data: BytesLike,
    *,
    tools: ExternalTools | None = None,
    limits: MediaLimits | None = None,
    temp_dir: StrPath | None = None,
    timeout_seconds: float | None = None,
) -> VideoInfo:
    """Probes untrusted video bytes with ffprobe and checks the limits.

    Args:
        data: The encoded video (untrusted).
        tools: Where the binaries are. Defaults to
            :meth:`ExternalTools.from_env() <provenance_kit.tools.ExternalTools.from_env>`.
        limits: The limits to enforce. Defaults to ``MediaLimits()``.
        temp_dir: Directory for the private temporary file that MP4 files with
            a trailing ``moov`` box need; ``None`` uses the platform default.
        timeout_seconds: The ffprobe timeout. Defaults to
            ``limits.subprocess_timeout_seconds``.

    Returns:
        The validated stream information.

    Raises:
        TypeError: If ``data`` is not bytes-like.
        MediaLimitExceededError: If the size, dimensions, duration or declared
            frame count exceed ``limits``.
        UnsupportedMediaError: If the bytes are not a supported video
            container, or contain no video stream.
        MediaDecodeError: If ffprobe cannot read the media or reports
            something unusable.
        BackendUnavailableError: If ffprobe is not available.
        ExternalToolTimeoutError: If ffprobe times out.
    """
    limits = MediaLimits() if limits is None else limits
    tools = ExternalTools.from_env() if tools is None else tools
    media_type = check_media_input(data, limits, VIDEO_CONTAINERS)
    with staged_input(data, media_type, temp_dir=temp_dir) as staged:
        return probe_staged(
            staged, media_type, tools=tools, limits=limits, timeout_seconds=timeout_seconds
        )


def probe_staged(
    staged: StagedInput,
    media_type: MediaType,
    *,
    tools: ExternalTools,
    limits: MediaLimits,
    timeout_seconds: float | None,
) -> VideoInfo:
    """Runs ffprobe on an already staged input (internal, shared with frames).

    Args:
        staged: The staged input.
        media_type: The sniffed container.
        tools: Where the binaries are.
        limits: The limits to enforce.
        timeout_seconds: The ffprobe timeout, or ``None`` for the limits'.

    Returns:
        The validated stream information.
    """
    timeout = limits.subprocess_timeout_seconds if timeout_seconds is None else timeout_seconds
    document = run_ffprobe_json(
        staged,
        ["-print_format", "json", "-show_entries", _PROBE_ENTRIES],
        tools=tools,
        what="video probe",
        timeout_seconds=timeout,
        max_output_bytes=_MAX_PROBE_OUTPUT,
    )
    return validate_probe_report(document, media_type, limits)


def parse_probe_output(raw: bytes, media_type: MediaType, limits: MediaLimits) -> VideoInfo:
    """Parses and validates ffprobe's JSON report (internal; untrusted input).

    Args:
        raw: ffprobe's standard output.
        media_type: The sniffed container.
        limits: The limits to enforce.

    Returns:
        The validated stream information.

    Raises:
        MediaDecodeError: If the report is not usable JSON or lacks required
            fields.
        UnsupportedMediaError: If there is no video stream or too many streams.
        MediaLimitExceededError: If a declared value exceeds ``limits``.
    """
    return validate_probe_report(load_json_object(raw, "video probe"), media_type, limits)


def validate_probe_report(
    document: Mapping[str, Any], media_type: MediaType, limits: MediaLimits
) -> VideoInfo:
    """Validates ffprobe's decoded report (internal; untrusted input).

    Args:
        document: The decoded JSON object.
        media_type: The sniffed container.
        limits: The limits to enforce.

    Returns:
        The validated stream information.

    Raises:
        MediaDecodeError: If the report lacks required fields.
        UnsupportedMediaError: If there is no video stream or too many streams.
        MediaLimitExceededError: If a declared value exceeds ``limits``.
    """
    summary = _summarize_streams(document.get("streams"))
    selected = summary.selected
    fmt = document.get("format")
    fmt = fmt if isinstance(fmt, dict) else {}

    index = json_int(selected.get("index"))
    if index is None or not 0 <= index < MAX_STREAMS:
        raise MediaDecodeError("video probe: the video stream has no valid index")
    width = json_int(selected.get("width"))
    height = json_int(selected.get("height"))
    if width is None or height is None or width < 1 or height < 1:
        raise MediaDecodeError("video probe: the video stream declares no frame size")
    limits.check_dimensions(width, height)

    duration = _non_negative(json_float(fmt.get("duration")))
    if duration is None:
        duration = _non_negative(json_float(selected.get("duration")))
    if duration is not None:
        limits.check_duration(duration)

    frame_rate = json_rate(selected.get("avg_frame_rate"), maximum=_MAX_FRAME_RATE) or json_rate(
        selected.get("r_frame_rate"), maximum=_MAX_FRAME_RATE
    )
    frame_count = _non_negative(json_int(selected.get("nb_frames")))
    if frame_count is not None:
        limits.check_frames(frame_count)
    if frame_rate is not None and duration is not None:
        estimate = duration * frame_rate
        if estimate > limits.max_frames:
            limits.check_frames(math.ceil(estimate))

    start_time = _non_negative(json_float(selected.get("start_time")))
    if start_time is None or start_time > limits.max_duration_seconds:
        start_time = 0.0
    rotation, mirrored = _display_transform(selected)

    return VideoInfo(
        container=media_type,
        format_name=json_name(fmt.get("format_name")) or "unknown",
        duration_seconds=duration,
        width=width,
        height=height,
        rotation=rotation,
        frame_rate=frame_rate,
        frame_count=frame_count,
        video_codec=json_name(selected.get("codec_name")) or "unknown",
        pixel_format=json_name(selected.get("pix_fmt")),
        audio_codec=summary.audio_codec,
        video_streams=summary.video,
        audio_streams=summary.audio,
        subtitle_streams=summary.subtitle,
        other_streams=summary.other,
        bit_rate=_non_negative(json_int(fmt.get("bit_rate"))),
        stream_index=index,
        start_time=start_time,
        mirrored=mirrored,
    )


@dataclass(frozen=True, slots=True)
class _StreamSummary:
    selected: Mapping[str, Any]
    audio_codec: str | None
    video: int
    audio: int
    subtitle: int
    other: int


def _summarize_streams(streams_raw: object) -> _StreamSummary:
    """Counts the streams and selects the first non-cover-art video stream."""
    if not isinstance(streams_raw, list):
        raise MediaDecodeError("video probe: the report lists no streams")
    if len(streams_raw) > MAX_STREAMS:
        raise UnsupportedMediaError(
            f"the media has too many streams ({len(streams_raw)} > {MAX_STREAMS})"
        )
    counts = {"video": 0, "audio": 0, "subtitle": 0, "other": 0}
    selected: Mapping[str, Any] | None = None
    audio_codec: str | None = None
    for stream in streams_raw:
        if not isinstance(stream, dict):
            continue
        kind = stream.get("codec_type")
        counts[kind if kind in ("video", "audio", "subtitle") else "other"] += 1
        if kind == "audio" and audio_codec is None:
            audio_codec = json_name(stream.get("codec_name"))
        if kind == "video" and selected is None and not _is_attached_picture(stream):
            selected = stream
    if selected is None:
        raise UnsupportedMediaError("the media contains no video stream")
    return _StreamSummary(
        selected=selected,
        audio_codec=audio_codec,
        video=counts["video"],
        audio=counts["audio"],
        subtitle=counts["subtitle"],
        other=counts["other"],
    )


def _is_attached_picture(stream: Mapping[str, Any]) -> bool:
    disposition = stream.get("disposition")
    return isinstance(disposition, dict) and json_int(disposition.get("attached_pic")) == 1


def _non_negative(value: _N | None) -> _N | None:
    return value if value is not None and value >= 0 else None


def _display_transform(stream: Mapping[str, Any]) -> tuple[int, bool]:
    """Clockwise display rotation and mirroring from the display matrix or the legacy tag.

    ffprobe reports a display matrix's rotation counterclockwise (``r``, from
    ``av_display_rotation_get``). Without a mirror, ffmpeg's autorotation
    turns the frames clockwise by ``-r``; with one (negative determinant), it
    flips them horizontally and turns them clockwise by ``180 - r``. Both
    were checked against ffmpeg 8.1 for every combination of
    ``-display_rotation``, ``-display_hflip`` and ``-display_vflip``.
    """
    degrees: float | None = None
    mirrored = False
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for entry in side_data[:16]:
            if not isinstance(entry, dict):
                continue
            matrix = _display_matrix(entry.get("displaymatrix"))
            value = json_float(entry.get("rotation"))
            if value is None and matrix is not None:
                value = _matrix_rotation(matrix)
            if value is not None:
                mirrored = matrix is not None and _determinant(matrix) < 0
                degrees = 180.0 - value if mirrored else -value
                break
    if degrees is None:
        tags = stream.get("tags")
        if isinstance(tags, dict):
            degrees = json_float(tags.get("rotate"))  # legacy tag: clockwise
    if degrees is None or abs(degrees) > 3600:
        return 0, False
    quarter = round(degrees / 90.0)
    if abs(degrees - 90.0 * quarter) > 2.0:
        _log.debug("ignoring a display rotation that is not a multiple of 90 degrees")
        return 0, False
    return (quarter % 4) * 90, mirrored


def _display_matrix(text: object) -> tuple[int, int, int, int] | None:
    """The 2 x 2 linear part ``(a, b, c, d)`` of ffprobe's printed display matrix.

    ffprobe prints three rows such as ``00000000:  0  65536  0`` (16.16 fixed
    point; the third column and row are the translation and the scale).
    Anything else (wrong count, values outside 32 bits, a singular matrix)
    gives ``None``.
    """
    if not isinstance(text, str) or len(text) > _MAX_DISPLAY_MATRIX_TEXT:
        return None
    try:
        values = _display_matrix_values(text)
    except ValueError:
        return None
    a, b, _, c, d, _, _, _, _ = values
    if a * d - b * c == 0:
        return None
    return a, b, c, d


def _display_matrix_values(text: str) -> list[int]:
    """The nine integers of a printed display matrix; ``ValueError`` if malformed."""
    values: list[int] = []
    for line in text.splitlines():
        label, separator, rest = line.partition(":")
        if not separator:
            if line.strip():
                raise ValueError("a display matrix row has no label")
            continue
        if not label.strip().isalnum():
            raise ValueError("bad display matrix row label")
        for token in rest.split():
            number = int(token)  # ValueError if not an integer
            if not -_INT32 <= number < _INT32:
                raise ValueError("display matrix value out of range")
            values.append(number)
    if len(values) != 9:
        raise ValueError("a display matrix has nine values")
    return values


def _determinant(matrix: tuple[int, int, int, int]) -> int:
    a, b, c, d = matrix
    return a * d - b * c


def _matrix_rotation(matrix: tuple[int, int, int, int]) -> float:
    """Counterclockwise rotation in degrees, as ``av_display_rotation_get`` computes it."""
    a, b, c, d = matrix
    scale_0 = math.hypot(a, c)
    scale_1 = math.hypot(b, d)
    return -math.degrees(math.atan2(b / scale_1, a / scale_0))
