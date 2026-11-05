"""Decode video frames from untrusted media through an ffmpeg pipe.

:func:`iter_frames` streams decoded frames as :class:`VideoFrame` objects:
RGB ``uint8`` NumPy arrays shaped ``(height, width, 3)`` with a timestamp.
Memory stays bounded: one frame is read from ffmpeg at a time, and ffmpeg
scales every frame to a fixed output size, so a stream that changes
resolution mid-way cannot inflate the records.

Sampling (:class:`FrameSampling`):

* ``FPS`` (default): frames at a fixed rate (``fps`` per second) through
  ffmpeg's ``fps`` filter, which picks the frame shown at each instant.
  Timestamps are rebased to the stream's first frame first, so the sampling
  grid does not depend on the container's start offset (Matroska and MPEG-TS
  often start at a few milliseconds). Timestamps are exact by construction:
  ``index / fps`` seconds from the first frame.
* ``KEYFRAMES``: only key frames (``-skip_frame nokey``, so non-key frames
  are not even decoded). Timestamps come from a first ffprobe pass over the
  key frames, relative to the stream's start time.
* ``ALL``: every frame, without dropping or duplicating. Timestamps are
  nominal, ``index / frame_rate`` with the declared average frame rate, which
  is exact for constant-frame-rate video only.

Rotation: the container's display rotation (display matrix or legacy
``rotate`` tag, see :class:`~provenance_kit.video.probe.VideoInfo`) is applied
explicitly in the filter graph (ffmpeg's implicit autorotation is disabled),
so frames come out in display orientation with known dimensions. The sample
aspect ratio is not applied: frames have the stored pixel grid, rotated.

Limits (:class:`~provenance_kit.limits.MediaLimits`): the probe checks the
byte size, dimensions, declared duration and declared frame count before
decoding; the expected number of output frames is checked before decoding
too; ffmpeg runs with ``-frames:v`` and ``-t`` caps (headers can lie), and the
frames and their timestamps are checked while reading, so media that runs
past ``max_duration_seconds`` or ``max_frames`` raises
:class:`~provenance_kit.errors.MediaLimitExceededError` whatever its header
declared. The ``max_frames`` argument of :func:`iter_frames` is a soft stop
(the caller wants no more), whereas exceeding ``limits.max_frames`` raises.
Containers whose header carries placeholder values, such as AVI files that
were written to a pipe (2**30 frames, about 4.3e7 seconds), are refused by
the limits.

Thread safety: the functions are thread-safe; each iteration runs its own
ffmpeg process. A single iterator must be consumed by one thread.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from provenance_kit._checks import check_int, check_number
from provenance_kit._ffmpeg import (
    FFMPEG_COMMON,
    StagedInput,
    as_media_error,
    check_media_input,
    json_float,
    run_ffprobe_json,
    staged_input,
)
from provenance_kit._logging import get_logger
from provenance_kit._typing import BytesLike, StrPath
from provenance_kit.errors import (
    ExternalToolError,
    ExternalToolTimeoutError,
    MediaDecodeError,
    MediaLimitExceededError,
    UnsupportedMediaError,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType
from provenance_kit.tools import ExternalTools, Tool
from provenance_kit.video.probe import VIDEO_CONTAINERS, VideoInfo, probe_staged

__all__ = [
    "MAX_SAMPLING_FPS",
    "FrameSampling",
    "VideoFrame",
    "extract_frames",
    "iter_frames",
    "output_size",
]

_log = get_logger(__name__)

#: The highest sampling rate accepted for ``FrameSampling.FPS``.
MAX_SAMPLING_FPS = 240.0

_DURATION_SLACK_SECONDS = 1.0  # ffmpeg decodes a little past the limit, to detect overflow
_KEYFRAME_ENTRY_BYTES = 96  # generous bound on one ffprobe JSON frame entry
_ROTATION_FILTERS = {90: "transpose=clock", 180: "hflip,vflip", 270: "transpose=cclock"}


class FrameSampling(enum.StrEnum):
    """Which frames :func:`iter_frames` decodes."""

    FPS = "fps"
    KEYFRAMES = "keyframes"
    ALL = "all"


@dataclass(frozen=True, slots=True, eq=False)
class VideoFrame:
    """One decoded frame.

    Attributes:
        index: The position of the frame in the output sequence (0-based).
        timestamp: Seconds from the start of the stream (see the module
            docstring for how each sampling mode computes it).
        image: The pixels, ``uint8``, shape ``(height, width, 3)``, RGB,
            C-contiguous. The array is **read-only** (it wraps the bytes read
            from ffmpeg); copy it before modifying it.
    """

    index: int
    timestamp: float
    image: npt.NDArray[np.uint8] = field(repr=False)

    @property
    def width(self) -> int:
        """The frame width in pixels."""
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        """The frame height in pixels."""
        return int(self.image.shape[0])


def output_size(
    info: VideoInfo,
    *,
    size: tuple[int, int] | None = None,
    max_side: int | None = None,
) -> tuple[int, int]:
    """Computes the output frame size for :func:`iter_frames`.

    Args:
        info: The probed stream.
        size: An exact ``(width, height)``; the aspect ratio is not kept.
        max_side: Downscale (never upscale) so that the longer side is at most
            this many pixels, keeping the display aspect ratio.

    Returns:
        ``(width, height)`` in display orientation.

    Raises:
        ValueError: If both ``size`` and ``max_side`` are given, or a value is
            not a positive integer.
    """
    if size is not None and max_side is not None:
        raise ValueError("pass size or max_side, not both")
    if size is not None:
        return _check_size(size)
    width, height = info.display_width, info.display_height
    if max_side is not None:
        check_int("max_side", max_side, minimum=1)
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / longest
            width = max(1, round(width * scale))
            height = max(1, round(height * scale))
    return width, height


def iter_frames(
    data: BytesLike,
    *,
    sampling: FrameSampling | str = FrameSampling.FPS,
    fps: float = 1.0,
    max_frames: int | None = None,
    size: tuple[int, int] | None = None,
    max_side: int | None = None,
    tools: ExternalTools | None = None,
    limits: MediaLimits | None = None,
    temp_dir: StrPath | None = None,
    timeout_seconds: float | None = None,
) -> Iterator[VideoFrame]:
    """Decodes frames from untrusted video bytes, one at a time.

    Arguments, the byte limit and the container are checked when this
    function is called; probing and decoding start on the first iteration, so
    the media errors listed below are raised by ``next()``. Closing the
    iterator early kills ffmpeg and deletes any temporary file.

    Args:
        data: The encoded video (untrusted).
        sampling: Which frames to decode (see :class:`FrameSampling`).
        fps: The sampling rate for ``FrameSampling.FPS``, in frames per
            second (0 < fps <= :data:`MAX_SAMPLING_FPS`). Ignored otherwise.
        max_frames: Stop after this many frames (not an error).
        size: Exact output ``(width, height)``; see :func:`output_size`.
        max_side: Downscale so the longer side is at most this; see
            :func:`output_size`.
        tools: Where the binaries are. Defaults to ``ExternalTools.from_env()``.
        limits: The limits to enforce. Defaults to ``MediaLimits()``.
        temp_dir: Directory for a private temporary file when the container
            needs seeking (MP4 with a trailing ``moov`` box).
        timeout_seconds: Timeout of each tool run, including the time the
            caller spends between frames. Defaults to
            ``limits.subprocess_timeout_seconds``.

    Returns:
        An iterator of frames in presentation order.

    Raises:
        TypeError: If ``data`` is not bytes-like.
        ValueError: If an argument is out of range.
        MediaLimitExceededError: If the media exceeds ``limits`` (declared or
            counted while decoding).
        UnsupportedMediaError: If the media is not a supported video, or the
            frame rate needed by ``FrameSampling.ALL`` is unknown.
        MediaDecodeError: If ffprobe or ffmpeg cannot read the media.
        BackendUnavailableError: If ffmpeg or ffprobe is not available.
        ExternalToolTimeoutError: If a tool times out.
    """
    mode = FrameSampling(sampling)
    if mode is FrameSampling.FPS:
        check_number("fps", fps, minimum=0.0, maximum=MAX_SAMPLING_FPS, exclusive_minimum=True)
    if max_frames is not None:
        check_int("max_frames", max_frames, minimum=1)
    if size is not None and max_side is not None:
        raise ValueError("pass size or max_side, not both")
    if size is not None:
        _check_size(size)
    if max_side is not None:
        check_int("max_side", max_side, minimum=1)
    limits = MediaLimits() if limits is None else limits
    tools = ExternalTools.from_env() if tools is None else tools
    media_type = check_media_input(data, limits, VIDEO_CONTAINERS)
    return _generate(
        data,
        media_type=media_type,
        mode=mode,
        fps=float(fps),
        max_frames=max_frames,
        size=size,
        max_side=max_side,
        tools=tools,
        limits=limits,
        temp_dir=temp_dir,
        timeout_seconds=timeout_seconds,
    )


def extract_frames(
    data: BytesLike,
    *,
    sampling: FrameSampling | str = FrameSampling.FPS,
    fps: float = 1.0,
    max_frames: int | None = None,
    size: tuple[int, int] | None = None,
    max_side: int | None = None,
    tools: ExternalTools | None = None,
    limits: MediaLimits | None = None,
    temp_dir: StrPath | None = None,
    timeout_seconds: float | None = None,
) -> list[VideoFrame]:
    """Decodes frames into a list; see :func:`iter_frames` for the arguments.

    Prefer :func:`iter_frames` for long videos: this function keeps every
    frame in memory.

    Args:
        data: The encoded video (untrusted).
        sampling: Which frames to decode.
        fps: The sampling rate for ``FrameSampling.FPS``.
        max_frames: Stop after this many frames.
        size: Exact output ``(width, height)``.
        max_side: Downscale so the longer side is at most this.
        tools: Where the binaries are.
        limits: The limits to enforce.
        temp_dir: Directory for a private temporary file, if one is needed.
        timeout_seconds: Timeout of each tool run.

    Returns:
        The frames in presentation order.

    Raises:
        TypeError: See :func:`iter_frames`.
        ValueError: See :func:`iter_frames`.
        MediaLimitExceededError: See :func:`iter_frames`.
        UnsupportedMediaError: See :func:`iter_frames`.
        MediaDecodeError: See :func:`iter_frames`.
        BackendUnavailableError: See :func:`iter_frames`.
        ExternalToolTimeoutError: See :func:`iter_frames`.
    """
    return list(
        iter_frames(
            data,
            sampling=sampling,
            fps=fps,
            max_frames=max_frames,
            size=size,
            max_side=max_side,
            tools=tools,
            limits=limits,
            temp_dir=temp_dir,
            timeout_seconds=timeout_seconds,
        )
    )


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _generate(
    data: BytesLike,
    *,
    media_type: MediaType,
    mode: FrameSampling,
    fps: float,
    max_frames: int | None,
    size: tuple[int, int] | None,
    max_side: int | None,
    tools: ExternalTools,
    limits: MediaLimits,
    temp_dir: StrPath | None,
    timeout_seconds: float | None,
) -> Iterator[VideoFrame]:
    with staged_input(data, media_type, temp_dir=temp_dir) as staged:
        info = probe_staged(
            staged, media_type, tools=tools, limits=limits, timeout_seconds=timeout_seconds
        )
        width, height = output_size(info, size=size, max_side=max_side)
        limits.check_dimensions(width, height)

        keyframe_times: list[float] | None = None
        if mode is FrameSampling.KEYFRAMES:
            keyframe_times = _keyframe_times(staged, info, tools, limits, timeout_seconds)
            limits.check_frames(len(keyframe_times))
        else:
            _check_expected_frames(info, mode, fps, limits)
        if mode is FrameSampling.ALL and info.frame_rate is None:
            raise UnsupportedMediaError(
                "the frame rate is unknown, so frame timestamps cannot be computed; "
                "use FrameSampling.FPS"
            )

        hard_cap = limits.max_frames + 1  # one more than allowed, to detect overflow
        cap = hard_cap if max_frames is None else min(max_frames, hard_cap)
        if keyframe_times is not None:
            cap = min(cap, max(len(keyframe_times), 1))
        args = _ffmpeg_args(
            staged, info, mode=mode, fps=fps, size=(width, height), cap=cap, limits=limits
        )
        frame_bytes = width * height * 3
        timeout = limits.subprocess_timeout_seconds if timeout_seconds is None else timeout_seconds

        index = 0
        try:
            with tools.stream(
                Tool.FFMPEG,
                args,
                stdin=staged.stdin,
                timeout_seconds=timeout,
                max_stdout_bytes=cap * frame_bytes,
            ) as proc:
                while max_frames is None or index < max_frames:
                    if keyframe_times is not None and index >= len(keyframe_times):
                        _log.debug("ffmpeg produced more key frames than ffprobe listed")
                        break
                    record = proc.read_exact(frame_bytes)
                    if record is None:
                        break
                    if index >= limits.max_frames:
                        raise MediaLimitExceededError(
                            "max_frames", limits.max_frames, limits.max_frames + 1
                        )
                    timestamp = _timestamp(index, mode, fps, info, keyframe_times)
                    if timestamp > limits.max_duration_seconds + _interval(mode, fps, info):
                        raise MediaLimitExceededError(
                            "max_duration_seconds", limits.max_duration_seconds, timestamp
                        )
                    image = np.frombuffer(record, dtype=np.uint8).reshape(height, width, 3)
                    yield VideoFrame(index=index, timestamp=timestamp, image=image)
                    index += 1
        except ExternalToolTimeoutError:
            raise
        except ExternalToolError as exc:
            if exc.returncode is None:
                raise MediaDecodeError(
                    "frame decoding failed: ffmpeg output ended in the middle of a frame"
                ) from exc
            raise as_media_error(exc, "frame decoding") from exc


def _ffmpeg_args(
    staged: StagedInput,
    info: VideoInfo,
    *,
    mode: FrameSampling,
    fps: float,
    size: tuple[int, int],
    cap: int,
    limits: MediaLimits,
) -> list[str]:
    width, height = size
    before_input = ["-noautorotate"]
    if mode is FrameSampling.KEYFRAMES:
        before_input += ["-skip_frame", "nokey"]
    filters: list[str] = []
    rotation = _ROTATION_FILTERS.get(info.rotation)
    if rotation:
        filters.append(rotation)
    if mode is FrameSampling.FPS:
        filters += ["setpts=PTS-STARTPTS", f"fps={fps!r}"]
    filters += [f"scale={width}:{height}:flags=bicubic", "format=rgb24"]
    return [
        *FFMPEG_COMMON,
        *staged.input_args(*before_input),
        "-map",
        f"0:{info.stream_index}",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        ",".join(filters),
        "-fps_mode",
        "passthrough",
        "-frames:v",
        str(cap),
        "-t",
        f"{limits.max_duration_seconds + _DURATION_SLACK_SECONDS:.3f}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]


def _check_expected_frames(
    info: VideoInfo, mode: FrameSampling, fps: float, limits: MediaLimits
) -> None:
    if info.duration_seconds is None:
        return
    if mode is FrameSampling.FPS:
        expected = math.ceil(info.duration_seconds * fps)
    elif info.frame_count is not None:
        expected = info.frame_count
    elif info.frame_rate is not None:
        expected = math.ceil(info.duration_seconds * info.frame_rate)
    else:
        return
    limits.check_frames(expected)


def _keyframe_times(
    staged: StagedInput,
    info: VideoInfo,
    tools: ExternalTools,
    limits: MediaLimits,
    timeout_seconds: float | None,
) -> list[float]:
    """Lists the key frame timestamps with ffprobe (decodes key frames only)."""
    timeout = limits.subprocess_timeout_seconds if timeout_seconds is None else timeout_seconds
    document = run_ffprobe_json(
        staged,
        [
            "-select_streams",
            str(info.stream_index),
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-print_format",
            "json",
        ],
        tools=tools,
        what="key frame listing",
        timeout_seconds=timeout,
        max_output_bytes=(limits.max_frames + 1) * _KEYFRAME_ENTRY_BYTES + 4096,
        before_input=("-skip_frame", "nokey"),
    )
    frames = document.get("frames")
    if not isinstance(frames, list):
        raise MediaDecodeError("key frame listing: the report lists no frames")
    if len(frames) > limits.max_frames:
        raise MediaLimitExceededError("max_frames", limits.max_frames, len(frames))
    times: list[float] = []
    previous = info.start_time
    for entry in frames:
        value = entry.get("best_effort_timestamp_time") if isinstance(entry, dict) else None
        seconds = json_float(value)
        if seconds is None:
            # A key frame without a timestamp: keep the sequence aligned with
            # the decoded frames by repeating the previous time.
            seconds = previous
        previous = seconds
        times.append(max(seconds - info.start_time, 0.0))
    return times


def _timestamp(
    index: int,
    mode: FrameSampling,
    fps: float,
    info: VideoInfo,
    keyframe_times: list[float] | None,
) -> float:
    if mode is FrameSampling.FPS:
        return index / fps
    if keyframe_times is not None:
        return keyframe_times[index]
    assert info.frame_rate is not None  # noqa: S101 - checked before decoding
    return index / info.frame_rate


def _interval(mode: FrameSampling, fps: float, info: VideoInfo) -> float:
    """Tolerance for the duration check: one sampling interval."""
    if mode is FrameSampling.FPS:
        return 1.0 / fps
    if mode is FrameSampling.ALL and info.frame_rate is not None:
        return 1.0 / info.frame_rate
    return 0.0


def _check_size(size: object) -> tuple[int, int]:
    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError("size must be a (width, height) tuple of positive ints")
    width = check_int("size[0]", size[0], minimum=1)
    height = check_int("size[1]", size[1], minimum=1)
    return width, height
