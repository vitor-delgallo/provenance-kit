"""Fakes for the audio unit tests: an ExternalTools stand-in and tiny containers."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Iterator
from typing import Any

from provenance_kit.errors import ExternalToolError
from provenance_kit.tools import Tool, ToolResult

#: A WAV head (RIFF/WAVE): sniffed as audio/wav, streams through a pipe.
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64
#: An M4A head with the media data before the sample tables.
M4A_MDAT_FIRST = (
    b"\x00\x00\x00\x18ftypM4A \x00\x00\x02\x00M4A isom"
    + b"\x00\x00\x00\x08mdat"
    + b"\x00\x00\x00\x08moov"
)


def probe_json(**stream_overrides: Any) -> bytes:
    """An ffprobe report for one stereo 48 kHz audio stream (overridable)."""
    stream: dict[str, Any] = {
        "index": 0,
        "codec_type": "audio",
        "codec_name": "pcm_s16le",
        "sample_rate": "48000",
        "channels": 2,
        "duration": "1.000000",
    }
    stream.update(stream_overrides)
    return json.dumps(
        {"streams": [stream], "format": {"format_name": "wav", "duration": "1.0"}}
    ).encode()


class FakeProcess:
    """Mimics ToolProcess for canned standard output (with its output cap)."""

    def __init__(self, output: bytes, max_stdout_bytes: int | None = None) -> None:
        self._output = output
        self._pos = 0
        self._cap = max_stdout_bytes
        self.tool = "ffmpeg"
        self.bytes_read = 0
        self.output_limit_exceeded = False
        self.stopped = False

    def read(self, size: int = 65536) -> bytes:
        chunk = self._output[self._pos : self._pos + size]
        self._pos += len(chunk)
        self.bytes_read += len(chunk)
        if self._cap is not None and self.bytes_read > self._cap:
            self.output_limit_exceeded = True
            raise ExternalToolError(self.tool, "output exceeded the limit")
        return chunk

    def iter_chunks(self, size: int = 65536) -> Iterator[bytes]:
        while not self.stopped and (chunk := self.read(size)):
            yield chunk

    def stop(self) -> None:
        self.stopped = True


class FakeTools:
    """An ExternalTools stand-in with canned outputs; records calls.

    ``fpcalc`` is a callable receiving the PCM bytes it was fed and returning
    ``(returncode, stdout)``. ``ffmpeg`` is the output of every ffmpeg run, or
    a list of outputs for successive runs (the last one repeats).
    """

    def __init__(
        self,
        *,
        ffprobe: bytes = b"{}",
        ffmpeg: bytes | list[bytes] = b"",
        fpcalc: Callable[[bytes], tuple[int, bytes]] | None = None,
        ffmpeg_returncode: int = 0,
    ) -> None:
        self._ffprobe = ffprobe
        self._ffmpeg = ffmpeg
        self._fpcalc = fpcalc
        self._ffmpeg_returncode = ffmpeg_returncode
        self.calls: list[tuple[str, list[str], dict[str, Any]]] = []
        self.fpcalc_input = b""

    def run(self, tool: Tool | str, args: list[str], **kwargs: Any) -> ToolResult:
        name = Tool(tool).value
        self.calls.append((name, list(args), kwargs))
        if name == "ffprobe":
            return ToolResult(
                tool=name, returncode=0, stdout=self._ffprobe, stderr_tail=b"", elapsed_seconds=0
            )
        if name == "fpcalc" and self._fpcalc is not None:
            stdin = kwargs.get("stdin")
            if isinstance(stdin, (bytes, bytearray, memoryview)):
                data = bytes(stdin)
            else:
                parts = []
                while chunk := stdin.read(4096):
                    parts.append(chunk)
                data = b"".join(parts)
            self.fpcalc_input = data
            code, out = self._fpcalc(data)
            if code and kwargs.get("check", True):
                raise ExternalToolError(name, "failed", returncode=code)
            return ToolResult(
                tool=name, returncode=code, stdout=out, stderr_tail=b"", elapsed_seconds=0
            )
        raise AssertionError(f"unexpected run of {name}")

    @contextlib.contextmanager
    def stream(self, tool: Tool | str, args: list[str], **kwargs: Any) -> Iterator[FakeProcess]:
        name = Tool(tool).value
        self.calls.append((name, list(args), kwargs))
        outputs = self._ffmpeg if isinstance(self._ffmpeg, list) else [self._ffmpeg]
        runs = len(self.args_of("ffmpeg"))
        yield FakeProcess(outputs[min(runs, len(outputs)) - 1], kwargs.get("max_stdout_bytes"))
        if self._ffmpeg_returncode:
            raise ExternalToolError(name, "failed", returncode=self._ffmpeg_returncode)

    def args_of(self, tool: str) -> list[list[str]]:
        """The argument lists of every call to ``tool``."""
        return [args for name, args, _ in self.calls if name == tool]
