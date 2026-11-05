"""Shared ffmpeg/ffprobe plumbing for the video modules (internal).

* :func:`check_video_input` applies the byte limit and sniffs the container.
* :func:`staged_input` presents the media to ffmpeg or ffprobe: through a pipe
  whenever the container allows it, otherwise through a private temporary
  file (see :func:`needs_seekable_input`).
* :data:`FFMPEG_COMMON` and :data:`FFPROBE_COMMON` are the mandatory leading
  arguments (ARCHITECTURE.md, section 6).
* :func:`as_media_error` turns a failed tool run on untrusted media into a
  :class:`~provenance_kit.errors.MediaDecodeError`.

When a temporary file is used: ISO base media files (MP4, QuickTime) whose
``moov`` box (the sample tables) comes **after** the ``mdat`` box (the media
data) cannot be decoded from a pipe: ffmpeg must seek back to the samples
after reading the tables at the end, and a pipe cannot seek. ffmpeg writes
such files by default unless ``-movflags +faststart`` is given, so they are
common. Fragmented MP4 (``moof`` boxes) and files with ``moov`` first stream
fine. Every other supported container is read sequentially and always uses a
pipe.
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
    "FFMPEG_COMMON",
    "FFPROBE_COMMON",
    "VIDEO_CONTAINERS",
    "StagedInput",
    "as_media_error",
    "check_video_input",
    "needs_seekable_input",
    "staged_input",
]

_log = get_logger(__name__)

#: Leading arguments of every ffmpeg call: no interactive input, no banner,
#: errors only. ``-nostdin`` does not stop ``-i pipe:0`` from reading media.
FFMPEG_COMMON: tuple[str, ...] = ("-nostdin", "-hide_banner", "-loglevel", "error")

#: Leading arguments of every ffprobe call. ffprobe has no ``-nostdin`` option
#: (it rejects it); it never reads the terminal, and the runner connects its
#: standard input to the media pipe or to the null device.
FFPROBE_COMMON: tuple[str, ...] = ("-hide_banner", "-loglevel", "error")

#: The video containers the video modules accept, with their ffmpeg demuxer.
#: Passing the demuxer explicitly stops ffmpeg from probing for (and possibly
#: picking) a playlist or concat demuxer.
VIDEO_CONTAINERS: dict[MediaType, str] = {
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
    """How a tool reads the media: the input arguments and what to feed it.

    Attributes:
        demuxer: The ffmpeg demuxer name passed with ``-f``.
        target: ``"pipe:0"`` or ``"file:<absolute temporary path>"``.
        stdin: The media bytes to feed on standard input, or ``None`` when the
            tool reads a temporary file.
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
            *before_input: Input options that must precede ``-i``, for example
                ``"-noautorotate"`` or ``"-skip_frame", "nokey"``.

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


def check_video_input(data: BytesLike, limits: MediaLimits) -> MediaType:
    """Checks the byte limit and detects the video container.

    Args:
        data: The encoded media (untrusted).
        limits: The limits to apply.

    Returns:
        The detected container.

    Raises:
        TypeError: If ``data`` is not bytes-like.
        MediaLimitExceededError: If ``data`` exceeds ``max_bytes``.
        UnsupportedMediaError: If the bytes are not a supported video container.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"data must be bytes-like, got {type(data).__name__}")
    size = memoryview(data).nbytes
    limits.check_bytes(size)
    return require_media_type(data, allowed=VIDEO_CONTAINERS.keys())


def needs_seekable_input(data: BytesLike, media_type: MediaType) -> bool:
    """Says whether ffmpeg needs a seekable (file) input for this media.

    Only ISO base media files can need it: see the module docstring. The
    top-level boxes are walked without trusting their sizes (bounded number of
    boxes, sizes checked against the data length). When the layout cannot be
    established, the answer is ``True``: a seekable input decodes everything a
    pipe does.

    Args:
        data: The encoded media (untrusted).
        media_type: The sniffed container.

    Returns:
        True if the media should be read from a temporary file.
    """
    if media_type not in _BMFF_TYPES:
        return False
    return _first_layout_box(memoryview(data).cast("B")) not in (b"moov", b"moof")


def _first_layout_box(view: memoryview) -> bytes | None:
    """Returns the first top-level ``moov``, ``moof`` or ``mdat`` box type.

    ``None`` when none is found within the walk or the box sizes are invalid.
    """
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
            size = total - offset  # the box extends to the end of the data
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
    demuxer: str,
    *,
    temp_dir: StrPath | None,
) -> Iterator[StagedInput]:
    """Presents the media to a tool through a pipe or a private temporary file.

    Args:
        data: The encoded media (untrusted).
        media_type: The sniffed container.
        demuxer: The ffmpeg demuxer for ``media_type``.
        temp_dir: Where a temporary file may be created (``None``: the
            platform default). Used only when :func:`needs_seekable_input`.

    Yields:
        The staged input. A temporary file is deleted when the block exits.
    """
    view = memoryview(data).cast("B")
    if not needs_seekable_input(view, media_type):
        yield StagedInput(demuxer=demuxer, target="pipe:0", stdin=view)
        return
    _log.debug("staging %s media in a temporary file (needs seeking)", media_type.mime)
    with private_temp_file(directory=temp_dir, data=view) as path:
        yield StagedInput(demuxer=demuxer, target=f"file:{path}", stdin=None)


def as_media_error(exc: ExternalToolError, what: str) -> ExternalToolError | MediaDecodeError:
    """Classifies a tool failure that happened while reading untrusted media.

    A non-zero exit status while demuxing or decoding means the media could not
    be read: it becomes :class:`~provenance_kit.errors.MediaDecodeError`.
    Timeouts and output-cap violations stay as they are.

    Args:
        exc: The error raised by the runner.
        what: A short, safe description, for example ``"video probe"``.

    Returns:
        The exception to raise (``raise as_media_error(exc, ...) from exc``).
    """
    if isinstance(exc, ExternalToolTimeoutError) or exc.returncode is None:
        return exc
    return MediaDecodeError(
        f"{what} failed: {exc.tool} could not read the media (exit status {exc.returncode})"
    )
