"""Decode untrusted audio to float32 PCM through an ffmpeg pipe.

:func:`decode_pcm` returns a :class:`PcmAudio`: a ``float32`` NumPy array
shaped ``(channels, samples)`` in ``[-1, 1]`` plus its sample rate (the
library's audio convention, ARCHITECTURE.md section 4). The target sample
rate and channel count are chosen by the caller (ffmpeg resamples and
down- or up-mixes); by default the source's are kept.

Any supported container works, including the audio track of a video file
(MP4, Matroska/WebM, AVI, MPEG-TS). The first audio stream is decoded.

Limits (:class:`~provenance_kit.limits.MediaLimits`), in order:

1. the byte size, before anything else;
2. what ffprobe declares: channels, sample rate and duration;
3. the requested output layout (channels, sample rate);
4. while decoding: ffmpeg runs with ``-t`` slightly above
   ``max_duration_seconds`` and the samples are counted, so a container that
   under-declares its duration is refused with
   :class:`~provenance_kit.errors.MediaLimitExceededError` instead of
   decoding without bound.

Untrusted input never reaches a shell; ffprobe's report is parsed
defensively (wrong types, ``NaN``, absurd values are refused or ignored).

Thread safety: the functions are thread-safe; each call runs its own tools.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from provenance_kit._checks import check_int, check_number
from provenance_kit._ffmpeg import (
    FFMPEG_COMMON,
    StagedInput,
    as_media_error,
    check_media_input,
    json_float,
    json_int,
    json_name,
    load_json_object,
    run_ffprobe_json,
    staged_input,
)
from provenance_kit._logging import get_logger
from provenance_kit._typing import BytesLike, StrPath
from provenance_kit.errors import (
    ExternalToolError,
    MediaDecodeError,
    MediaLimitExceededError,
    UnsupportedMediaError,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType
from provenance_kit.tools import ExternalTools, Tool

__all__ = [
    "MAX_STREAMS",
    "MIN_SAMPLE_RATE",
    "AudioInfo",
    "PcmAudio",
    "decode_pcm",
    "probe_audio",
]

_log = get_logger(__name__)

#: Containers declaring more streams than this are refused.
MAX_STREAMS = 64

#: The lowest output sample rate :func:`decode_pcm` accepts.
MIN_SAMPLE_RATE = 1000

#: The containers the audio modules accept (detected from the bytes): audio
#: files, and video containers for their audio track.
AUDIO_CONTAINERS: frozenset[MediaType] = frozenset(
    {
        MediaType.WAV,
        MediaType.MP3,
        MediaType.AAC,
        MediaType.FLAC,
        MediaType.OGG,
        MediaType.M4A,
        MediaType.AIFF,
        MediaType.MP4,
        MediaType.QUICKTIME,
        MediaType.WEBM,
        MediaType.MATROSKA,
        MediaType.AVI,
        MediaType.MPEG_TS,
    }
)

_MAX_PROBE_OUTPUT = 1024 * 1024
_DURATION_SLACK_SECONDS = 1.0
_READ_CHUNK = 1024 * 1024
_PROBE_ENTRIES = (
    "format=format_name,duration,bit_rate"
    ":stream=index,codec_type,codec_name,sample_rate,channels,duration"
)


@dataclass(frozen=True, slots=True)
class AudioInfo:
    """What ffprobe reports about the first audio stream, after validation.

    Attributes:
        container: The container detected from the bytes.
        format_name: ffprobe's demuxer name list.
        duration_seconds: The declared duration, or ``None`` if unknown.
        sample_rate: The stream's sample rate in hertz.
        channels: The stream's channel count.
        codec: The stream's codec, for example ``"aac"``.
        audio_streams: Number of audio streams.
        video_streams: Number of video streams.
        stream_index: ffprobe index of the selected audio stream.
        bit_rate: Declared overall bit rate in bits per second, or ``None``.
    """

    container: MediaType
    format_name: str
    duration_seconds: float | None
    sample_rate: int
    channels: int
    codec: str
    audio_streams: int
    video_streams: int
    stream_index: int
    bit_rate: int | None


@dataclass(frozen=True, slots=True, eq=False)
class PcmAudio:
    """Decoded audio samples.

    Attributes:
        samples: ``float32``, shape ``(channels, samples)``, C-contiguous,
            values in ``[-1, 1]``.
        sample_rate: Samples per second per channel.

    Raises:
        TypeError: If ``samples`` is not a float32 NumPy array.
        ValueError: If the shape or sample rate is invalid.
    """

    samples: npt.NDArray[np.float32] = field(repr=False)
    sample_rate: int

    def __post_init__(self) -> None:
        """Validates the fields."""
        _check_float32_matrix(self.samples)
        check_int("sample_rate", self.sample_rate, minimum=1)

    @property
    def channels(self) -> int:
        """The number of channels."""
        return int(self.samples.shape[0])

    @property
    def frames(self) -> int:
        """The number of samples per channel."""
        return int(self.samples.shape[1])

    @property
    def duration_seconds(self) -> float:
        """The duration, from the sample count."""
        return self.frames / self.sample_rate

    def __repr__(self) -> str:
        """Returns a summary without the samples."""
        return (
            f"PcmAudio(channels={self.channels}, frames={self.frames}, "
            f"sample_rate={self.sample_rate})"
        )


def _check_float32_matrix(samples: object) -> None:
    if not isinstance(samples, np.ndarray) or samples.dtype != np.float32:
        raise TypeError("samples must be a float32 NumPy array")
    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError("samples must be shaped (channels, samples)")


def probe_audio(
    data: BytesLike,
    *,
    tools: ExternalTools | None = None,
    limits: MediaLimits | None = None,
    temp_dir: StrPath | None = None,
    timeout_seconds: float | None = None,
) -> AudioInfo:
    """Probes untrusted media with ffprobe and checks the audio limits.

    Args:
        data: The encoded media (untrusted).
        tools: Where the binaries are. Defaults to ``ExternalTools.from_env()``.
        limits: The limits to enforce. Defaults to ``MediaLimits()``.
        temp_dir: Directory for the private temporary file that MP4/M4A files
            with a trailing ``moov`` box need.
        timeout_seconds: The ffprobe timeout. Defaults to
            ``limits.subprocess_timeout_seconds``.

    Returns:
        The validated audio stream information.

    Raises:
        TypeError: If ``data`` is not bytes-like.
        MediaLimitExceededError: If the size, channels, sample rate or
            declared duration exceed ``limits``.
        UnsupportedMediaError: If the container is not supported or holds no
            audio stream.
        MediaDecodeError: If ffprobe cannot read the media.
        BackendUnavailableError: If ffprobe is not available.
        ExternalToolTimeoutError: If ffprobe times out.
    """
    limits = MediaLimits() if limits is None else limits
    tools = ExternalTools.from_env() if tools is None else tools
    media_type = check_media_input(data, limits, AUDIO_CONTAINERS)
    with staged_input(data, media_type, temp_dir=temp_dir) as staged:
        return probe_staged(
            staged, media_type, tools=tools, limits=limits, timeout_seconds=timeout_seconds
        )


def decode_pcm(
    data: BytesLike,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
    max_duration_seconds: float | None = None,
    tools: ExternalTools | None = None,
    limits: MediaLimits | None = None,
    temp_dir: StrPath | None = None,
    timeout_seconds: float | None = None,
) -> PcmAudio:
    """Decodes the first audio stream of untrusted media to float32 PCM.

    Args:
        data: The encoded media (untrusted).
        sample_rate: Output sample rate in hertz (ffmpeg resamples), or
            ``None`` to keep the source rate.
        channels: Output channel count (1 = mono downmix, 2 = stereo, ...),
            or ``None`` to keep the source layout. ffmpeg's default matrix
            is used: stereo to mono sums the channels at -3 dB each.
        max_duration_seconds: Decode at most this much audio (truncation, not
            an error).
        tools: Where the binaries are. Defaults to ``ExternalTools.from_env()``.
        limits: The limits to enforce. Defaults to ``MediaLimits()``.
        temp_dir: Directory for a private temporary file, if the container
            needs one.
        timeout_seconds: Timeout of each tool run. Defaults to
            ``limits.subprocess_timeout_seconds``.

    Returns:
        The decoded audio.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``sample_rate``, ``channels`` or
            ``max_duration_seconds`` is out of range.
        MediaLimitExceededError: If the media or the requested layout exceeds
            ``limits`` (declared, or counted while decoding).
        UnsupportedMediaError: If the media holds no audio stream.
        MediaDecodeError: If the media cannot be decoded.
        BackendUnavailableError: If ffmpeg or ffprobe is not available.
        ExternalToolTimeoutError: If a tool times out.
    """
    limits = MediaLimits() if limits is None else limits
    tools = ExternalTools.from_env() if tools is None else tools
    if sample_rate is not None:
        check_int("sample_rate", sample_rate, minimum=MIN_SAMPLE_RATE)
    if channels is not None:
        check_int("channels", channels, minimum=1)
    if max_duration_seconds is not None:
        check_number(
            "max_duration_seconds", max_duration_seconds, minimum=0.0, exclusive_minimum=True
        )
    media_type = check_media_input(data, limits, AUDIO_CONTAINERS)
    with staged_input(data, media_type, temp_dir=temp_dir) as staged:
        info = probe_staged(
            staged, media_type, tools=tools, limits=limits, timeout_seconds=timeout_seconds
        )
        out_rate = info.sample_rate if sample_rate is None else sample_rate
        out_channels = info.channels if channels is None else channels
        if out_rate < MIN_SAMPLE_RATE:
            raise UnsupportedMediaError(
                f"sample rates below {MIN_SAMPLE_RATE} Hz are not supported"
            )
        limits.check_audio(out_channels, out_rate)
        raw = read_pcm(
            staged,
            info,
            tools=tools,
            limits=limits,
            sample_format="f32le",
            sample_rate=out_rate,
            channels=out_channels,
            max_duration_seconds=max_duration_seconds,
            timeout_seconds=timeout_seconds,
        )
    frame_bytes = 4 * out_channels
    usable = len(raw) - len(raw) % frame_bytes
    interleaved = np.frombuffer(raw, dtype="<f4", count=usable // 4).reshape(-1, out_channels)
    samples = np.array(interleaved.T, dtype=np.float32, order="C")  # always a writable copy
    np.clip(samples, -1.0, 1.0, out=samples)
    np.nan_to_num(samples, copy=False, nan=0.0)
    return PcmAudio(samples=samples, sample_rate=out_rate)


# --------------------------------------------------------------------------
# Internals shared with the fingerprint module
# --------------------------------------------------------------------------


def pcm_args(
    staged: StagedInput,
    info: AudioInfo,
    *,
    limits: MediaLimits,
    sample_format: str,
    sample_rate: int,
    channels: int,
    max_duration_seconds: float | None,
) -> list[str]:
    """Builds the ffmpeg arguments that decode the audio stream to raw PCM.

    Args:
        staged: The staged input.
        info: The probed stream.
        limits: The limits in force.
        sample_format: ``"f32le"`` or ``"s16le"``.
        sample_rate: Output rate.
        channels: Output channels.
        max_duration_seconds: Caller's truncation, if any.

    Returns:
        The argument list.
    """
    cap = limits.max_duration_seconds + _DURATION_SLACK_SECONDS
    if max_duration_seconds is not None:
        cap = min(cap, max_duration_seconds)
    codec = {"f32le": "pcm_f32le", "s16le": "pcm_s16le"}[sample_format]
    return [
        *FFMPEG_COMMON,
        *staged.input_args(),
        "-map",
        f"0:{info.stream_index}",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-t",
        f"{cap:.6f}",
        "-f",
        sample_format,
        "-c:a",
        codec,
        "pipe:1",
    ]


def sample_cap_bytes(
    limits: MediaLimits,
    *,
    bytes_per_frame: int,
    sample_rate: int,
    max_duration_seconds: float | None,
) -> int:
    """Largest PCM output accepted: the duration limit, plus one frame of slack.

    Args:
        limits: The limits in force.
        bytes_per_frame: Bytes per sample frame (all channels).
        sample_rate: Output rate.
        max_duration_seconds: Caller's truncation, if any.

    Returns:
        The byte cap; more output means the media exceeds the limit.
    """
    seconds = limits.max_duration_seconds
    if max_duration_seconds is not None:
        seconds = min(seconds, max_duration_seconds)
    return (math.ceil(seconds * sample_rate) + 1) * bytes_per_frame


def read_pcm(
    staged: StagedInput,
    info: AudioInfo,
    *,
    tools: ExternalTools,
    limits: MediaLimits,
    sample_format: str,
    sample_rate: int,
    channels: int,
    max_duration_seconds: float | None,
    timeout_seconds: float | None,
) -> bytes:
    """Runs ffmpeg and returns the raw PCM bytes, counting them against the limits."""
    args = pcm_args(
        staged,
        info,
        limits=limits,
        sample_format=sample_format,
        sample_rate=sample_rate,
        channels=channels,
        max_duration_seconds=max_duration_seconds,
    )
    bytes_per_frame = (4 if sample_format == "f32le" else 2) * channels
    cap = sample_cap_bytes(
        limits,
        bytes_per_frame=bytes_per_frame,
        sample_rate=sample_rate,
        max_duration_seconds=max_duration_seconds,
    )
    timeout = limits.subprocess_timeout_seconds if timeout_seconds is None else timeout_seconds
    parts: list[bytes] = []
    total = 0
    try:
        with tools.stream(Tool.FFMPEG, args, stdin=staged.stdin, timeout_seconds=timeout) as proc:
            for chunk in proc.iter_chunks(_READ_CHUNK):
                total += len(chunk)
                if total > cap:
                    seconds = total / (bytes_per_frame * sample_rate)
                    raise MediaLimitExceededError(
                        "max_duration_seconds", limits.max_duration_seconds, seconds
                    )
                parts.append(chunk)
    except ExternalToolError as exc:
        raise as_media_error(exc, "audio decoding") from exc
    if max_duration_seconds is None and total > cap - bytes_per_frame:
        # Exactly at the cap: the stream ran into the duration limit.
        raise MediaLimitExceededError(
            "max_duration_seconds",
            limits.max_duration_seconds,
            total / (bytes_per_frame * sample_rate),
        )
    return b"".join(parts)


def probe_staged(
    staged: StagedInput,
    media_type: MediaType,
    *,
    tools: ExternalTools,
    limits: MediaLimits,
    timeout_seconds: float | None,
) -> AudioInfo:
    """Runs ffprobe on a staged input and validates the report (internal).

    Args:
        staged: The staged input.
        media_type: The sniffed container.
        tools: Where the binaries are.
        limits: The limits to enforce.
        timeout_seconds: The ffprobe timeout, or ``None`` for the limits'.

    Returns:
        The validated audio stream information.
    """
    timeout = limits.subprocess_timeout_seconds if timeout_seconds is None else timeout_seconds
    document = run_ffprobe_json(
        staged,
        ["-print_format", "json", "-show_entries", _PROBE_ENTRIES],
        tools=tools,
        what="audio probe",
        timeout_seconds=timeout,
        max_output_bytes=_MAX_PROBE_OUTPUT,
    )
    return validate_probe_report(document, media_type, limits)


def parse_probe_output(raw: bytes, media_type: MediaType, limits: MediaLimits) -> AudioInfo:
    """Parses and validates ffprobe's JSON report (internal; untrusted input).

    Args:
        raw: ffprobe's standard output.
        media_type: The sniffed container.
        limits: The limits to enforce.

    Returns:
        The validated audio stream information.

    Raises:
        MediaDecodeError: If the report is unusable or lacks required fields.
        UnsupportedMediaError: If there is no audio stream or too many streams.
        MediaLimitExceededError: If a declared value exceeds ``limits``.
    """
    return validate_probe_report(load_json_object(raw, "audio probe"), media_type, limits)


def validate_probe_report(
    document: Mapping[str, Any], media_type: MediaType, limits: MediaLimits
) -> AudioInfo:
    """Validates ffprobe's decoded report (internal; untrusted input).

    Args:
        document: The decoded JSON object.
        media_type: The sniffed container.
        limits: The limits to enforce.

    Returns:
        The validated audio stream information.

    Raises:
        MediaDecodeError: If the report lacks required fields.
        UnsupportedMediaError: If there is no audio stream or too many streams.
        MediaLimitExceededError: If a declared value exceeds ``limits``.
    """
    streams = document.get("streams")
    if not isinstance(streams, list):
        raise MediaDecodeError("audio probe: the report lists no streams")
    if len(streams) > MAX_STREAMS:
        raise UnsupportedMediaError(
            f"the media has too many streams ({len(streams)} > {MAX_STREAMS})"
        )
    entries = [s for s in streams if isinstance(s, dict)]
    audio = [s for s in entries if s.get("codec_type") == "audio"]
    video_count = sum(1 for s in entries if s.get("codec_type") == "video")
    if not audio:
        raise UnsupportedMediaError("the media contains no audio stream")
    selected = audio[0]
    fmt = document.get("format")
    fmt = fmt if isinstance(fmt, dict) else {}

    index = json_int(selected.get("index"))
    if index is None or not 0 <= index < MAX_STREAMS:
        raise MediaDecodeError("audio probe: the audio stream has no valid index")
    sample_rate = json_int(selected.get("sample_rate"))
    channel_count = json_int(selected.get("channels"))
    if sample_rate is None or sample_rate < 1 or channel_count is None or channel_count < 1:
        raise MediaDecodeError("audio probe: the audio stream declares no sample rate or layout")
    limits.check_audio(channel_count, sample_rate)

    duration = json_float(fmt.get("duration"))
    if duration is None or duration < 0:
        duration = json_float(selected.get("duration"))
    if duration is not None and duration < 0:
        duration = None
    if duration is not None:
        limits.check_duration(duration)
    bit_rate = json_int(fmt.get("bit_rate"))
    return AudioInfo(
        container=media_type,
        format_name=json_name(fmt.get("format_name")) or "unknown",
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channel_count,
        codec=json_name(selected.get("codec_name")) or "unknown",
        audio_streams=len(audio),
        video_streams=video_count,
        stream_index=index,
        bit_rate=bit_rate if bit_rate is not None and bit_rate >= 0 else None,
    )
