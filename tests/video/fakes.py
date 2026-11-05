"""Fakes for the video unit tests: an ExternalTools stand-in and tiny containers."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Iterator
from typing import Any

from provenance_kit.errors import ExternalToolError
from provenance_kit.tools import Tool, ToolResult

#: A minimal MP4 head: ftyp, then moov before mdat (streams through a pipe).
MP4_MOOV_FIRST = (
    b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
    + b"\x00\x00\x00\x08moov"
    + b"\x00\x00\x00\x08mdat"
)
#: The same with the media data first (needs a seekable temporary file).
MP4_MDAT_FIRST = (
    b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
    + b"\x00\x00\x00\x08mdat"
    + b"\x00\x00\x00\x08moov"
)
#: A Matroska/WebM head.
WEBM = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\x82\x84webm" + b"\x00" * 32


def probe_json(**stream_overrides: Any) -> bytes:
    """An ffprobe report for one 8x6 video stream (fields overridable)."""
    stream: dict[str, Any] = {
        "index": 0,
        "codec_type": "video",
        "codec_name": "h264",
        "width": 8,
        "height": 6,
        "pix_fmt": "yuv420p",
        "avg_frame_rate": "25/1",
        "r_frame_rate": "25/1",
        "nb_frames": "50",
        "duration": "2.000000",
        "start_time": "0.000000",
        "disposition": {"attached_pic": 0},
    }
    stream.update(stream_overrides)
    return json.dumps(
        {
            "streams": [stream, {"index": 1, "codec_type": "audio", "codec_name": "aac"}],
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "2.0"},
        }
    ).encode()


class FakeProcess:
    """Mimics provenance_kit.tools.ToolProcess for canned standard output."""

    def __init__(self, output: bytes) -> None:
        self._output = output
        self._pos = 0
        self.tool = "ffmpeg"
        self.stderr_tail = b""

    def read(self, size: int = 65536) -> bytes:
        chunk = self._output[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def read_exact(self, size: int) -> bytes | None:
        remaining = len(self._output) - self._pos
        if remaining == 0:
            return None
        if remaining < size:
            self._pos = len(self._output)
            raise ExternalToolError("ffmpeg", "output ended in the middle of a record")
        return self.read(size)

    def iter_chunks(self, size: int = 65536) -> Iterator[bytes]:
        while chunk := self.read(size):
            yield chunk


class FakeTools:
    """An ExternalTools stand-in: canned outputs per tool, calls recorded.

    ``ffprobe`` may be bytes or a callable ``(args) -> bytes`` (to answer the
    stream probe and the key frame listing differently).
    """

    def __init__(
        self,
        *,
        ffprobe: bytes | Callable[[list[str]], bytes] = b"{}",
        ffmpeg: bytes = b"",
        ffmpeg_returncode: int = 0,
        ffprobe_returncode: int = 0,
    ) -> None:
        self._ffprobe = ffprobe
        self._ffmpeg = ffmpeg
        self._ffmpeg_returncode = ffmpeg_returncode
        self._ffprobe_returncode = ffprobe_returncode
        self.calls: list[tuple[str, list[str], dict[str, Any]]] = []

    def run(self, tool: Tool | str, args: list[str], **kwargs: Any) -> ToolResult:
        name = Tool(tool).value
        self.calls.append((name, list(args), kwargs))
        if name != "ffprobe":
            raise AssertionError(f"unexpected run of {name}")
        if self._ffprobe_returncode:
            raise ExternalToolError(name, "failed", returncode=self._ffprobe_returncode)
        output = self._ffprobe(list(args)) if callable(self._ffprobe) else self._ffprobe
        return ToolResult(
            tool=name, returncode=0, stdout=output, stderr_tail=b"", elapsed_seconds=0.0
        )

    @contextlib.contextmanager
    def stream(self, tool: Tool | str, args: list[str], **kwargs: Any) -> Iterator[FakeProcess]:
        name = Tool(tool).value
        self.calls.append((name, list(args), kwargs))
        process = FakeProcess(self._ffmpeg)
        yield process
        if self._ffmpeg_returncode:
            raise ExternalToolError(name, "failed", returncode=self._ffmpeg_returncode)

    def args_of(self, tool: str) -> list[list[str]]:
        """The argument lists of every call to ``tool``."""
        return [args for name, args, _ in self.calls if name == tool]
