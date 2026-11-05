"""ffmpeg and ffprobe plumbing shared by the video and audio modules (internal).

The rules of ARCHITECTURE.md, section 6, in one place:

* :func:`check_media_input` applies the byte limit and sniffs the container
  from the bytes (:mod:`provenance_kit.mediatype`); the container decides
  the demuxer (:data:`DEMUXERS`), which is always passed explicitly with
  ``-f`` so ffmpeg never probes for (and possibly picks) a playlist or
  concat demuxer.
* :func:`staged_input` presents the media to a tool: through a pipe whenever
  the container allows it, otherwise through a private temporary file (see
  :func:`needs_seekable_input`). :meth:`StagedInput.input_args` adds the
  protocol whitelist (``pipe``, or ``file,pipe`` for a temporary file).
* :data:`FFMPEG_COMMON` and :data:`FFPROBE_COMMON` are the mandatory leading
  arguments of every call.
* :func:`run_ffprobe_json` runs ffprobe with a capped output and parses its
  JSON report; :func:`load_json_object`, :func:`json_int`,
  :func:`json_float`, :func:`json_name` and :func:`json_rate` read the
  report defensively: it describes untrusted media, so wrong types,
  ``NaN``/``Infinity``, oversized strings and absurd values are refused or
  ignored, never trusted.
* :func:`as_media_error` turns a failed tool run on untrusted media into a
  :class:`~provenance_kit.errors.MediaDecodeError`.

When a temporary file is used: ISO base media files (MP4, M4A, QuickTime)
whose ``moov`` box (the sample tables) comes **after** the ``mdat`` box (the
media data) cannot be decoded from a pipe: ffmpeg must seek back to the
samples after reading the tables at the end, and a pipe cannot seek. ffmpeg
writes such files by default unless ``-movflags +faststart`` is given, so
they are common. Fragmented MP4 (``moof`` boxes) and files with ``moov``
first stream fine. Every other supported container is read sequentially and
always uses a pipe.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

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
from provenance_kit.tools import ExternalTools, Tool

__all__ = [
    "DEMUXERS",
    "FFMPEG_COMMON",
    "FFPROBE_COMMON",
    "StagedInput",
    "as_media_error",
    "check_media_input",
    "json_float",
    "json_int",
    "json_name",
    "json_rate",
    "load_json_object",
    "needs_seekable_input",
    "run_ffprobe_json",
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

#: The ffmpeg demuxer of each container the video and audio modules accept.
DEMUXERS: Mapping[MediaType, str] = MappingProxyType(
    {
        MediaType.MP4: "mov",
        MediaType.QUICKTIME: "mov",
        MediaType.M4A: "mov",
        MediaType.WEBM: "matroska",
        MediaType.MATROSKA: "matroska",
        MediaType.AVI: "avi",
        MediaType.MPEG_TS: "mpegts",
        MediaType.WAV: "wav",
        MediaType.MP3: "mp3",
        MediaType.AAC: "aac",
        MediaType.FLAC: "flac",
        MediaType.OGG: "ogg",
        MediaType.AIFF: "aiff",
    }
)

_BMFF_TYPES = frozenset({MediaType.MP4, MediaType.QUICKTIME, MediaType.M4A})
_MAX_TOP_LEVEL_BOXES = 4096
_MAX_NAME_CHARS = 64
_MAX_NUMBER_CHARS = 32
_NAME_RE = re.compile(r"^[A-Za-z0-9_.,+-]+$")
_RATIONAL_RE = re.compile(r"^(-?\d{1,12})[/:](-?\d{1,12})$")


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


def check_media_input(
    data: BytesLike, limits: MediaLimits, allowed: Collection[MediaType]
) -> MediaType:
    """Checks the byte limit and detects the container from the bytes.

    Args:
        data: The encoded media (untrusted).
        limits: The limits to apply.
        allowed: The containers the caller accepts (each must be in
            :data:`DEMUXERS`).

    Returns:
        The detected container.

    Raises:
        TypeError: If ``data`` is not bytes-like.
        MediaLimitExceededError: If ``data`` exceeds ``max_bytes``.
        UnsupportedMediaError: If the bytes are not one of ``allowed``.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"data must be bytes-like, got {type(data).__name__}")
    limits.check_bytes(memoryview(data).nbytes)
    return require_media_type(data, allowed=allowed)


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
    *,
    temp_dir: StrPath | None,
) -> Iterator[StagedInput]:
    """Presents the media to a tool through a pipe or a private temporary file.

    Args:
        data: The encoded media (untrusted).
        media_type: The sniffed container (a key of :data:`DEMUXERS`).
        temp_dir: Where a temporary file may be created (``None``: the
            platform default). Used only when :func:`needs_seekable_input`.

    Yields:
        The staged input. A temporary file is deleted when the block exits.

    Raises:
        KeyError: If ``media_type`` has no demuxer (a programming error: check
            the input with :func:`check_media_input` first).
    """
    demuxer = DEMUXERS[media_type]
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


def run_ffprobe_json(
    staged: StagedInput,
    output_args: Sequence[str],
    *,
    tools: ExternalTools,
    what: str,
    timeout_seconds: float,
    max_output_bytes: int,
    before_input: Sequence[str] = (),
) -> dict[str, Any]:
    """Runs ffprobe on a staged input and returns its JSON report.

    The arguments are :data:`FFPROBE_COMMON`, the staged input (with
    ``before_input`` ahead of ``-i``) and ``output_args``, which must ask for
    ``-print_format json``.

    Args:
        staged: The staged input.
        output_args: ffprobe's output options (``-show_entries``, ...).
        tools: Where ffprobe is.
        what: A short, safe description for error messages.
        timeout_seconds: The ffprobe timeout.
        max_output_bytes: The cap on the report's size.
        before_input: Input options such as ``"-skip_frame", "nokey"``.

    Returns:
        The decoded report (untrusted: read it with the ``json_*`` helpers).

    Raises:
        MediaDecodeError: If ffprobe cannot read the media or its report is
            not a JSON object.
        BackendUnavailableError: If ffprobe is not available.
        ExternalToolError: If the report exceeds ``max_output_bytes``.
        ExternalToolTimeoutError: If ffprobe times out.
    """
    args = [*FFPROBE_COMMON, *staged.input_args(*before_input), *output_args]
    try:
        result = tools.run(
            Tool.FFPROBE,
            args,
            stdin=staged.stdin,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_output_bytes,
        )
    except ExternalToolError as exc:
        raise as_media_error(exc, what) from exc
    return load_json_object(result.stdout, what)


# --------------------------------------------------------------------------
# Defensive JSON helpers (the report describes untrusted media)
# --------------------------------------------------------------------------


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} in the report")


def load_json_object(raw: bytes, what: str) -> dict[str, Any]:
    """Decodes a JSON object from untrusted tool output.

    ``NaN``, ``Infinity`` and ``-Infinity`` are refused.

    Args:
        raw: The tool's standard output.
        what: A short, safe description for error messages.

    Returns:
        The decoded object.

    Raises:
        MediaDecodeError: If the output is not a UTF-8 JSON object.
    """
    try:
        document = json.loads(raw.decode("utf-8"), parse_constant=_refuse_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise MediaDecodeError(f"{what}: the tool report is not valid JSON") from exc
    if not isinstance(document, dict):
        raise MediaDecodeError(f"{what}: the tool report is not a JSON object")
    return document


def json_int(value: object) -> int | None:
    """Reads an integer field: an ``int`` or a short decimal string.

    Args:
        value: The field value from the report.

    Returns:
        The integer, or ``None`` for a missing, ``bool`` or unusable value.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and 0 < len(value) <= _MAX_NUMBER_CHARS:
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def json_float(value: object) -> float | None:
    """Reads a real-number field: a number or a short decimal string.

    Args:
        value: The field value from the report.

    Returns:
        The finite number, or ``None`` for a missing, ``bool``, non-finite or
        unusable value.
    """
    if isinstance(value, bool):
        return None
    number: float
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str) and 0 < len(value) <= _MAX_NUMBER_CHARS:
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def json_rate(value: object, *, maximum: float) -> float | None:
    """Reads a rational rate such as ``"30000/1001"``.

    Args:
        value: The field value from the report.
        maximum: The largest plausible rate; larger values are ignored.

    Returns:
        The positive rate, or ``None`` when it is missing, malformed, zero,
        negative or above ``maximum``.
    """
    if not isinstance(value, str):
        return None
    match = _RATIONAL_RE.match(value.strip())
    if match is None:
        return None
    num, den = int(match.group(1)), int(match.group(2))
    if num <= 0 or den <= 0:
        return None
    rate = num / den
    return rate if 0.0 < rate <= maximum else None


def json_name(value: object) -> str | None:
    """Reads a short identifier such as a codec or format name.

    Args:
        value: The field value from the report.

    Returns:
        The name when it is a non-empty string of at most 64 letters, digits
        and ``_.,+-``, else ``None``.
    """
    if not isinstance(value, str) or not value or len(value) > _MAX_NAME_CHARS:
        return None
    return value if _NAME_RE.match(value) else None
