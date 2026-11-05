"""Shared ffmpeg/ffprobe plumbing for the audio modules (internal).

The same rules as the video modules (ARCHITECTURE.md, section 6): the
container is sniffed from the bytes and passed to the tool explicitly, the
protocol whitelist allows only ``pipe`` (or ``file,pipe`` for a temporary
file), and media goes through a pipe except ISO base media files (MP4, M4A,
QuickTime) whose ``moov`` box follows the ``mdat`` box: ffmpeg must seek back
to the samples after reading the tables at the end, which a pipe cannot do,
so those are written to a private temporary file for the duration of the
call. Fragmented MP4 and ``moov``-first files stream through the pipe.

(The video package has an equivalent private module; subpackages do not
import each other, see ARCHITECTURE.md, section 1.)
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

from provenance_kit._logging import get_logger
from provenance_kit._typing import BytesLike, StrPath
from provenance_kit.errors import (
    ExternalToolError,
    ExternalToolTimeoutError,
    MediaDecodeError,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType, require_media_type
from provenance_kit.tempfiles import private_temp_file

__all__ = [
    "AUDIO_CONTAINERS",
    "FFMPEG_COMMON",
    "FFPROBE_COMMON",
    "StagedInput",
    "as_media_error",
    "check_audio_input",
    "needs_seekable_input",
    "staged_input",
]

_log = get_logger(__name__)

#: Leading arguments of every ffmpeg call.
FFMPEG_COMMON: tuple[str, ...] = ("-nostdin", "-hide_banner", "-loglevel", "error")

#: Leading arguments of every ffprobe call (ffprobe has no ``-nostdin``).
FFPROBE_COMMON: tuple[str, ...] = ("-hide_banner", "-loglevel", "error")

#: Containers the audio modules accept (audio files, and video containers
#: for their audio track), with their ffmpeg demuxer.
AUDIO_CONTAINERS: dict[MediaType, str] = {
    MediaType.WAV: "wav",
    MediaType.MP3: "mp3",
    MediaType.AAC: "aac",
    MediaType.FLAC: "flac",
    MediaType.OGG: "ogg",
    MediaType.M4A: "mov",
    MediaType.AIFF: "aiff",
    MediaType.MP4: "mov",
    MediaType.QUICKTIME: "mov",
    MediaType.WEBM: "matroska",
    MediaType.MATROSKA: "matroska",
    MediaType.AVI: "avi",
    MediaType.MPEG_TS: "mpegts",
}

_BMFF_TYPES = frozenset({MediaType.MP4, MediaType.QUICKTIME, MediaType.M4A})
_MAX_TOP_LEVEL_BOXES = 4096


@dataclass(frozen=True, slots=True)
class StagedInput:
    """How a tool reads the media.

    Attributes:
        demuxer: The ffmpeg demuxer name passed with ``-f``.
        target: ``"pipe:0"`` or ``"file:<absolute temporary path>"``.
        stdin: The media bytes for standard input, or ``None`` for a file.
    """

    demuxer: str
    target: str
    stdin: memoryview | None

    @property
    def uses_file(self) -> bool:
        """True when the media was written to a temporary file."""
        return self.stdin is None

    def input_args(self, *before_input: str) -> list[str]:
        """Builds the protocol, demuxer and ``-i`` arguments.

        Args:
            *before_input: Input options that must precede ``-i``.

        Returns:
            The argument list, ending with ``-i <target>``.
        """
        whitelist = "file,pipe" if self.uses_file else "pipe"
        return [
            "-protocol_whitelist",
            whitelist,
            "-f",
            self.demuxer,
            *before_input,
            "-i",
            self.target,
        ]


def check_audio_input(data: BytesLike, limits: MediaLimits) -> MediaType:
    """Checks the byte limit and detects the container.

    Args:
        data: The encoded media (untrusted).
        limits: The limits to apply.

    Returns:
        The detected container.

    Raises:
        TypeError: If ``data`` is not bytes-like.
        MediaLimitExceededError: If ``data`` exceeds ``max_bytes``.
        UnsupportedMediaError: If the container is not supported.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"data must be bytes-like, got {type(data).__name__}")
    limits.check_bytes(memoryview(data).nbytes)
    return require_media_type(data, allowed=AUDIO_CONTAINERS.keys())


def needs_seekable_input(data: BytesLike, media_type: MediaType) -> bool:
    """Says whether ffmpeg needs a seekable (file) input for this media.

    Args:
        data: The encoded media (untrusted).
        media_type: The sniffed container.

    Returns:
        True for ISO base media files whose first ``moov``/``moof``/``mdat``
        box is ``mdat`` (or whose layout cannot be established).
    """
    if media_type not in _BMFF_TYPES:
        return False
    return _first_layout_box(memoryview(data).cast("B")) not in (b"moov", b"moof")


def _first_layout_box(view: memoryview) -> bytes | None:
    total = len(view)
    offset = 0
    for _ in range(_MAX_TOP_LEVEL_BOXES):
        if offset + 8 > total:
            break
        size = int.from_bytes(view[offset : offset + 4], "big")
        box = bytes(view[offset + 4 : offset + 8])
        header = 8
        if size == 1:
            if offset + 16 > total:
                break
            size = int.from_bytes(view[offset + 8 : offset + 16], "big")
            header = 16
        elif size == 0:
            size = total - offset
        if box in (b"moov", b"moof", b"mdat"):
            return box
        if size < header:
            break
        offset += size
    return None


@contextlib.contextmanager
def staged_input(
    data: BytesLike,
    media_type: MediaType,
    *,
    temp_dir: StrPath | None,
) -> Iterator[StagedInput]:
    """Presents the media to a tool through a pipe or a private temporary file.

    Args:
        data: The encoded media (untrusted).
        media_type: The sniffed container.
        temp_dir: Where a temporary file may be created, if one is needed.

    Yields:
        The staged input; a temporary file is deleted when the block exits.
    """
    demuxer = AUDIO_CONTAINERS[media_type]
    view = memoryview(data).cast("B")
    if not needs_seekable_input(view, media_type):
        yield StagedInput(demuxer=demuxer, target="pipe:0", stdin=view)
        return
    _log.debug("staging %s media in a temporary file (needs seeking)", media_type.mime)
    with private_temp_file(directory=temp_dir, data=view) as path:
        yield StagedInput(demuxer=demuxer, target=f"file:{path}", stdin=None)


def as_media_error(exc: ExternalToolError, what: str) -> ExternalToolError | MediaDecodeError:
    """Classifies a tool failure that happened while reading untrusted media.

    Args:
        exc: The error raised by the runner.
        what: A short, safe description.

    Returns:
        :class:`~provenance_kit.errors.MediaDecodeError` for a non-zero exit
        status; timeouts and output-cap violations unchanged.
    """
    if isinstance(exc, ExternalToolTimeoutError) or exc.returncode is None:
        return exc
    return MediaDecodeError(
        f"{what} failed: {exc.tool} could not read the media (exit status {exc.returncode})"
    )
