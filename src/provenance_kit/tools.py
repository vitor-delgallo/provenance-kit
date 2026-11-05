"""Locating and running external binaries: ffmpeg, ffprobe and fpcalc.

Two pieces:

* :class:`ExternalTools` says where the binaries are: explicit paths, or a
  lookup on ``PATH``. :meth:`ExternalTools.from_env` reads
  ``PROVENANCE_KIT_FFMPEG``, ``PROVENANCE_KIT_FFPROBE`` and
  ``PROVENANCE_KIT_FPCALC``.
* :func:`run_tool` and :func:`stream_tool` run a binary safely.

The runner's guarantees, which every module relies on:

* No shell, ever. Arguments are passed as a list; strings are never parsed.
  The executable must be an absolute path (use :meth:`ExternalTools.resolve`),
  and Windows batch files are refused because ``cmd.exe`` would re-parse
  their arguments.
* The ``PATH`` lookup ignores the current directory and relative entries, so
  a binary planted next to the media cannot be picked up.
* Media goes through pipes: ``stdin`` accepts bytes or a binary file object,
  fed by a background thread; standard output can be captured
  (:func:`run_tool`) or read incrementally (:func:`stream_tool`).
* Every run has a timeout (default:
  :attr:`MediaLimits.subprocess_timeout_seconds
  <provenance_kit.limits.MediaLimits.subprocess_timeout_seconds>`). On timeout
  the whole process tree is killed: a new session and ``killpg`` on POSIX, a
  Job Object on Windows.
* Captured output is capped. Standard output beyond its cap is an error (the
  process is killed); standard error is drained continuously and only its
  last bytes are kept.
* Failures raise :class:`~provenance_kit.errors.ExternalToolError` with a
  sanitized message (tool name and exit status only). The standard error tail
  is logged at DEBUG level only and kept on the exception as
  ``stderr_tail`` for classification.
"""

from __future__ import annotations

import contextlib
import enum
import logging
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Any, BinaryIO

from provenance_kit._logging import get_logger
from provenance_kit._typing import BytesLike, StrPath
from provenance_kit.errors import (
    BackendUnavailableError,
    ExternalToolError,
    ExternalToolTimeoutError,
)
from provenance_kit.limits import MediaLimits

__all__ = [
    "DEFAULT_MAX_STDERR_BYTES",
    "DEFAULT_MAX_STDOUT_BYTES",
    "ENV_PREFIX",
    "ExternalTools",
    "Tool",
    "ToolProcess",
    "ToolResult",
    "run_tool",
    "stream_tool",
]

_log = get_logger(__name__)

#: Prefix of the environment variables read by :meth:`ExternalTools.from_env`.
ENV_PREFIX = "PROVENANCE_KIT_"

#: Default cap on standard output captured by :func:`run_tool` (64 MiB).
DEFAULT_MAX_STDOUT_BYTES = 64 * 1024 * 1024

#: Default number of trailing standard-error bytes kept (64 KiB).
DEFAULT_MAX_STDERR_BYTES = 64 * 1024

_CHUNK = 64 * 1024
_DEBUG_STDERR_CHARS = 2000
_KILL_WAIT_SECONDS = 10.0
_JOIN_SECONDS = 5.0


class Tool(enum.StrEnum):
    """The external binaries provenance-kit knows how to locate."""

    FFMPEG = "ffmpeg"
    FFPROBE = "ffprobe"
    FPCALC = "fpcalc"


# --------------------------------------------------------------------------
# Locating binaries
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExternalTools:
    """Where to find the external binaries.

    Each field is either ``None`` (look the tool up on ``PATH`` when
    ``search_path`` is true), a bare program name (looked up on ``PATH``), or a
    path to the executable.

    Attributes:
        ffmpeg: The ffmpeg executable.
        ffprobe: The ffprobe executable.
        fpcalc: The Chromaprint ``fpcalc`` executable.
        search_path: Whether unset tools are looked up on ``PATH``.
    """

    ffmpeg: StrPath | None = None
    ffprobe: StrPath | None = None
    fpcalc: StrPath | None = None
    search_path: bool = True

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        search_path: bool = True,
    ) -> ExternalTools:
        """Builds a configuration from ``PROVENANCE_KIT_<TOOL>`` variables.

        Args:
            environ: The environment to read. Defaults to ``os.environ``.
            search_path: Whether tools without a variable are looked up on
                ``PATH``.

        Returns:
            The configuration. Empty variables count as unset.
        """
        env = os.environ if environ is None else environ

        def get(tool: Tool) -> str | None:
            value = env.get(ENV_PREFIX + tool.name, "").strip()
            return value or None

        return cls(
            ffmpeg=get(Tool.FFMPEG),
            ffprobe=get(Tool.FFPROBE),
            fpcalc=get(Tool.FPCALC),
            search_path=search_path,
        )

    def resolve(self, tool: Tool | str) -> Path:
        """Returns the absolute path of a tool's executable.

        Args:
            tool: The tool, as a :class:`Tool` or its name.

        Returns:
            The absolute path to an existing executable file.

        Raises:
            ValueError: If ``tool`` is not a known tool name.
            BackendUnavailableError: If the tool is not configured and not
                found, or if the configured path is not an executable file.
        """
        tool = Tool(tool)
        configured: StrPath | None = getattr(self, tool.value)
        if configured is not None:
            text = os.fspath(configured)
            if _is_bare_name(text):
                found = _search_path(text)
                if found is None:
                    raise BackendUnavailableError.missing_binary(
                        tool.value,
                        hint=f"the configured program name was not found on PATH ({tool.value})",
                    )
                return found
            path = _executable_file(Path(text))
            if path is None:
                raise BackendUnavailableError.missing_binary(
                    tool.value,
                    hint=f"the configured {tool.value} path is not an executable file",
                )
            return path
        if self.search_path:
            found = _search_path(tool.value)
            if found is not None:
                return found
        raise BackendUnavailableError.missing_binary(tool.value)

    def available(self, tool: Tool | str) -> bool:
        """Says whether :meth:`resolve` would succeed for ``tool``.

        Args:
            tool: The tool, as a :class:`Tool` or its name.

        Returns:
            True if the executable can be located.
        """
        try:
            self.resolve(tool)
        except BackendUnavailableError:
            return False
        return True

    def run(
        self,
        tool: Tool | str,
        args: Sequence[StrPath],
        **kwargs: Any,
    ) -> ToolResult:
        """Resolves ``tool`` and runs it with :func:`run_tool`.

        Args:
            tool: The tool to run.
            args: The arguments, without the executable.
            **kwargs: Keyword arguments for :func:`run_tool`.

        Returns:
            The result of :func:`run_tool`.
        """
        name = Tool(tool).value
        return run_tool(self.resolve(tool), args, tool=name, **kwargs)

    def stream(
        self,
        tool: Tool | str,
        args: Sequence[StrPath],
        **kwargs: Any,
    ) -> contextlib.AbstractContextManager[ToolProcess]:
        """Resolves ``tool`` and starts it with :func:`stream_tool`.

        Args:
            tool: The tool to run.
            args: The arguments, without the executable.
            **kwargs: Keyword arguments for :func:`stream_tool`.

        Returns:
            The context manager returned by :func:`stream_tool`.
        """
        name = Tool(tool).value
        return stream_tool(self.resolve(tool), args, tool=name, **kwargs)


def _is_bare_name(text: str) -> bool:
    separators = {"/", os.sep}
    if os.altsep:
        separators.add(os.altsep)
    return bool(text) and not any(sep in text for sep in separators) and text not in {".", ".."}


def _executable_file(path: Path) -> Path | None:
    candidates = [path]
    if sys.platform == "win32" and path.suffix.lower() != ".exe":
        candidates.append(path.with_name(path.name + ".exe"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.absolute()
    return None


def _search_path(name: str, path_env: str | None = None) -> Path | None:
    """Looks ``name`` up on PATH, ignoring the current and relative directories.

    Unlike ``shutil.which``, this never searches the current directory (which
    ``shutil.which`` does first on Windows), skips relative ``PATH`` entries,
    and on Windows considers ``.exe`` files only.
    """
    if path_env is None:
        path_env = os.environ.get("PATH", os.defpath)
    if sys.platform == "win32":
        names = [name] if name.lower().endswith(".exe") else [name + ".exe"]
    else:
        names = [name]
    for raw_entry in path_env.split(os.pathsep):
        entry = raw_entry.strip().strip('"')
        if not entry or not os.path.isabs(entry):
            continue
        for candidate_name in names:
            candidate = Path(entry) / candidate_name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return None


# --------------------------------------------------------------------------
# Process tree handling
# --------------------------------------------------------------------------

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _kernel32_cache: list[Any] = []

    def _kernel32() -> Any:
        if not _kernel32_cache:
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.SetInformationJobObject.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            k32.SetInformationJobObject.restype = wintypes.BOOL
            k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            k32.AssignProcessToJobObject.restype = wintypes.BOOL
            k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            k32.TerminateJobObject.restype = wintypes.BOOL
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            k32.CloseHandle.restype = wintypes.BOOL
            _kernel32_cache.append(k32)
        return _kernel32_cache[0]

    def _create_job_for(pid: int) -> int | None:
        """Puts process ``pid`` in a new kill-on-close Job Object.

        Returns the job handle, or None if the job could not be set up (the
        caller then falls back to killing the single process).
        """
        k32 = _kernel32()
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = k32.SetInformationJobObject(
            job,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        process = k32.OpenProcess(_PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, pid)
        if ok and process:
            assigned = k32.AssignProcessToJobObject(job, process)
            k32.CloseHandle(process)
            if assigned:
                return int(job)
        elif process:
            k32.CloseHandle(process)
        k32.CloseHandle(job)
        return None


class _ProcessTree:
    """Kills a child process and everything it started."""

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        self._proc = proc
        self._lock = threading.Lock()
        self._killed = False
        self._job: int | None = None
        if sys.platform == "win32":
            try:
                self._job = _create_job_for(proc.pid)
            except OSError:
                self._job = None

    def kill(self) -> None:
        with self._lock:
            if self._killed:
                return
            self._killed = True
        if sys.platform == "win32":
            if self._job is not None:
                _kernel32().TerminateJobObject(self._job, 1)
            with contextlib.suppress(OSError):
                self._proc.kill()
        else:
            import signal  # noqa: PLC0415 - POSIX only

            try:
                os.killpg(self._proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                with contextlib.suppress(OSError):
                    self._proc.kill()

    def close(self) -> None:
        """Releases OS resources and kills any process left in the tree."""
        if sys.platform == "win32":
            if self._job is not None:
                # KILL_ON_JOB_CLOSE: closing the last handle ends every
                # process still in the job, including orphaned grandchildren.
                _kernel32().CloseHandle(self._job)
                self._job = None
        else:
            import signal  # noqa: PLC0415 - POSIX only

            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self._proc.pid, signal.SIGKILL)


# --------------------------------------------------------------------------
# Running tools
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of :func:`run_tool`.

    Attributes:
        tool: The tool name used in messages.
        returncode: The exit code (negative for a POSIX signal).
        stdout: The captured standard output.
        stderr_tail: The last bytes of standard error. **Untrusted**: it can
            echo metadata from the media. Do not log it above DEBUG level.
        elapsed_seconds: Wall-clock run time.
    """

    tool: str
    returncode: int
    stdout: bytes
    stderr_tail: bytes
    elapsed_seconds: float

    def __repr__(self) -> str:
        """Returns a representation without the output contents."""
        return (
            f"ToolResult(tool={self.tool!r}, returncode={self.returncode}, "
            f"stdout=<{len(self.stdout)} bytes>, stderr_tail=<{len(self.stderr_tail)} bytes>, "
            f"elapsed_seconds={self.elapsed_seconds:.3f})"
        )


class ToolProcess:
    """A running tool whose standard output is read incrementally.

    Obtained from :func:`stream_tool`; use it inside the ``with`` block only.
    Reads block until data is available and return ``b""`` (or ``None`` for
    :meth:`read_exact`) at the end of the output. When the timeout kills the
    process, the output simply ends; the
    :class:`~provenance_kit.errors.ExternalToolTimeoutError` is raised when the
    block exits (or at once by :meth:`read_exact` if a record was cut short).
    Reads raise :class:`~provenance_kit.errors.ExternalToolError` as soon as
    standard output exceeds ``max_stdout_bytes``.

    Instances are not thread-safe: read from one thread.
    """

    def __init__(
        self,
        proc: subprocess.Popen[bytes],
        *,
        tool: str,
        stdin: BytesLike | BinaryIO | None,
        timeout_seconds: float,
        max_stdout_bytes: int | None,
        max_stderr_bytes: int,
        check: bool,
    ) -> None:
        """Wires up a started process. Internal: use :func:`stream_tool`."""
        self.tool = tool
        self.pid = proc.pid
        self._proc = proc
        self._tree = _ProcessTree(proc)
        self._check = check
        self._timeout_seconds = timeout_seconds
        self._deadline = time.monotonic() + timeout_seconds
        self._max_stdout_bytes = max_stdout_bytes
        self._max_stderr_bytes = max_stderr_bytes
        self._stdout_total = 0
        self._eof = False
        self._timed_out = threading.Event()
        self._overflowed = False
        self._stopped = False
        self._stdin_error: BaseException | None = None
        self._stderr_buf = bytearray()
        self._started = time.monotonic()
        self.elapsed_seconds = 0.0

        if proc.stdout is None or proc.stderr is None:
            raise RuntimeError("the process was started without output pipes")
        self._stdout: IO[bytes] = proc.stdout
        self._threads: list[threading.Thread] = []
        self._start_thread(self._drain_stderr, proc.stderr, "stderr")
        if stdin is not None:
            if proc.stdin is None:
                raise RuntimeError("the process was started without an input pipe")
            self._start_thread(self._feed_stdin, (proc.stdin, stdin), "stdin")
        self._timer = threading.Timer(timeout_seconds, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()

    # ------------------------------------------------------------ public

    @property
    def returncode(self) -> int | None:
        """The exit code once the process has exited, else ``None``."""
        return self._proc.returncode

    @property
    def stderr_tail(self) -> bytes:
        """The last bytes of standard error (untrusted; DEBUG logging only).

        Complete only after the ``with`` block has exited.
        """
        return bytes(self._stderr_buf)

    def read(self, size: int = _CHUNK) -> bytes:
        """Reads up to ``size`` bytes; returns ``b""`` at end of output.

        Args:
            size: The maximum number of bytes to return.

        Returns:
            The bytes read, possibly fewer than ``size``.
        """
        _check_positive_int("size", size)
        reader = getattr(self._stdout, "read1", self._stdout.read)
        chunk: bytes = reader(size)
        return self._account(chunk)

    def read_exact(self, size: int) -> bytes | None:
        """Reads exactly ``size`` bytes, for fixed-size records such as frames.

        Args:
            size: The record size in bytes.

        Returns:
            The record, or ``None`` if the output ended cleanly before it.

        Raises:
            ExternalToolError: If the output ends in the middle of a record.
            ExternalToolTimeoutError: If the timeout cut the record short.
        """
        _check_positive_int("size", size)
        parts: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self._account(self._stdout.read(remaining))
            if not chunk:
                break
            parts.append(chunk)
            remaining -= len(chunk)
        if remaining and self._timed_out.is_set():
            raise self._timeout_error()
        if remaining == size:
            return None
        if remaining:
            raise ExternalToolError(
                self.tool,
                f"output ended in the middle of a {size}-byte record",
                stderr_tail=self.stderr_tail,
            )
        return b"".join(parts)

    def iter_chunks(self, size: int = _CHUNK) -> Iterator[bytes]:
        """Yields standard output in chunks of at most ``size`` bytes.

        Args:
            size: The maximum chunk size.

        Yields:
            Non-empty chunks until the end of output.
        """
        while chunk := self.read(size):
            yield chunk

    def read_all(self) -> bytes:
        """Reads standard output to the end.

        Returns:
            All remaining output (bounded by ``max_stdout_bytes``).
        """
        return b"".join(self.iter_chunks())

    def stop(self) -> None:
        """Kills the process early because the caller needs no more output.

        A stopped process is not an error: its exit status is not checked.
        """
        self._stopped = True
        self._tree.kill()

    # ------------------------------------------------------------ internals

    def _start_thread(self, target: Any, arg: Any, label: str) -> None:
        thread = threading.Thread(
            target=target,
            args=(arg,),
            name=f"provenance-kit-{self.tool}-{label}",
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)

    def _account(self, chunk: bytes) -> bytes:
        if not chunk:
            # A timeout also ends the output (the process was killed); it is
            # reported when the block exits, with the complete stderr tail.
            self._eof = True
            return chunk
        self._stdout_total += len(chunk)
        if self._max_stdout_bytes is not None and self._stdout_total > self._max_stdout_bytes:
            self._overflowed = True
            self._tree.kill()
            raise ExternalToolError(
                self.tool,
                f"output exceeded the limit of {self._max_stdout_bytes} bytes and the "
                "process was killed",
                stderr_tail=self.stderr_tail,
            )
        return chunk

    def _drain_stderr(self, stream: IO[bytes]) -> None:
        reader = getattr(stream, "read1", stream.read)
        cap = self._max_stderr_bytes
        with contextlib.suppress(OSError, ValueError):
            while chunk := reader(_CHUNK):
                self._stderr_buf += chunk
                if len(self._stderr_buf) > cap:
                    del self._stderr_buf[: len(self._stderr_buf) - cap]

    def _feed_stdin(self, pair: tuple[IO[bytes], BytesLike | BinaryIO]) -> None:
        pipe, source = pair
        try:
            if isinstance(source, (bytes, bytearray, memoryview)):
                view = memoryview(source).cast("B")
                for offset in range(0, len(view), _CHUNK):
                    if not self._write(pipe, view[offset : offset + _CHUNK]):
                        return
            else:
                while True:
                    try:
                        chunk = source.read(_CHUNK)
                        if not isinstance(chunk, (bytes, bytearray, memoryview)):
                            raise TypeError("the stdin file object must return bytes")
                    except Exception as exc:
                        # The caller's stream failed: stop the tool and report
                        # the original error from the main thread.
                        self._stdin_error = exc
                        self._tree.kill()
                        return
                    if not chunk or not self._write(pipe, chunk):
                        return
        finally:
            with contextlib.suppress(OSError):
                pipe.close()

    @staticmethod
    def _write(pipe: IO[bytes], data: BytesLike) -> bool:
        try:
            pipe.write(data)
        except (OSError, ValueError):
            # The tool closed its input (it may need only a header) or was
            # killed. Its exit status decides whether that is an error.
            return False
        return True

    def _on_timeout(self) -> None:
        if self._proc.poll() is None:
            self._timed_out.set()
            self._tree.kill()

    def _timeout_error(self) -> ExternalToolTimeoutError:
        return ExternalToolTimeoutError(
            self.tool, self._timeout_seconds, stderr_tail=self.stderr_tail
        )

    def _at_end_of_output(self) -> bool:
        if self._eof:
            return True
        peek = getattr(self._stdout, "peek", None)
        try:
            data = peek(1) if peek is not None else self._stdout.read(1)
        except (OSError, ValueError):
            return True
        if not data:
            self._eof = True
            return True
        return False

    def _finish(self, failed: bool) -> None:
        """Ends the process and raises if the run failed.

        Args:
            failed: True if the caller's block raised; the process is then
                killed and nothing else is raised here.
        """
        try:
            if failed or self._overflowed or self._stopped:
                self._tree.kill()
            elif not self._timed_out.is_set() and not self._at_end_of_output():
                # The caller stopped reading before the end: not an error.
                self._stopped = True
                self._tree.kill()
            remaining = max(self._deadline - time.monotonic(), 0.0)
            try:
                self._proc.wait(timeout=remaining + 1.0)
            except subprocess.TimeoutExpired:
                self._timed_out.set()
                self._tree.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self._proc.wait(timeout=_KILL_WAIT_SECONDS)
        finally:
            self._timer.cancel()
            self._tree.close()
            for thread in self._threads:
                thread.join(_JOIN_SECONDS)
            for pipe in (self._proc.stdout, self._proc.stderr, self._proc.stdin):
                if pipe is not None:
                    with contextlib.suppress(OSError):
                        pipe.close()
            self.elapsed_seconds = time.monotonic() - self._started

        if failed or self._overflowed:
            return
        if self._timed_out.is_set():
            self._log_failure("timed out")
            raise self._timeout_error()
        if self._stdin_error is not None:
            raise self._stdin_error
        if self._stopped:
            return
        code = self._proc.returncode
        if self._check and code != 0:
            self._log_failure(f"exit status {code}")
            raise ExternalToolError(
                self.tool,
                _describe_exit(code),
                returncode=code,
                stderr_tail=self.stderr_tail,
            )

    def _log_failure(self, what: str) -> None:
        if _log.isEnabledFor(logging.DEBUG):
            tail = self._stderr_buf[-_DEBUG_STDERR_CHARS:].decode("utf-8", "replace")
            _log.debug("%s failed (%s); stderr tail: %r", self.tool, what, tail)


def _describe_exit(code: int | None) -> str:
    if code is None:
        return "did not exit normally"
    if code < 0:
        return f"was terminated by signal {-code}"
    return f"exited with status {code}"


class _ToolContext(contextlib.AbstractContextManager[ToolProcess]):
    def __init__(self, process: ToolProcess) -> None:
        self._process = process

    def __enter__(self) -> ToolProcess:
        return self._process

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._process._finish(failed=exc_type is not None)


def stream_tool(
    executable: StrPath,
    args: Sequence[StrPath],
    *,
    tool: str | None = None,
    stdin: BytesLike | BinaryIO | None = None,
    timeout_seconds: float | None = None,
    max_stdout_bytes: int | None = None,
    max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES,
    check: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: StrPath | None = None,
) -> contextlib.AbstractContextManager[ToolProcess]:
    """Starts a tool and lets the caller read its output incrementally.

    Use it as a context manager::

        with stream_tool(ffmpeg, args, tool="ffmpeg", stdin=data) as proc:
            while (frame := proc.read_exact(frame_size)) is not None:
                ...

    On leaving the block normally, the runner waits for the process (within
    the timeout) and checks its exit status. If the caller stopped reading
    before the end of the output, or called :meth:`ToolProcess.stop`, the
    process is killed and its status is not checked. If the block raises, the
    process is killed and the caller's exception propagates.

    Args:
        executable: The absolute path of the executable.
        args: The arguments, without the executable.
        tool: The name used in messages. Defaults to the executable's stem.
        stdin: Bytes or a readable binary file object to feed to the tool's
            standard input from a background thread. ``None`` connects
            standard input to the null device.
        timeout_seconds: The wall-clock limit for the whole run. Defaults to
            ``MediaLimits().subprocess_timeout_seconds``.
        max_stdout_bytes: The total standard output allowed, or ``None`` for no
            cap (the caller bounds what it reads).
        max_stderr_bytes: How many trailing bytes of standard error to keep.
        check: Whether a non-zero exit status raises ``ExternalToolError``.
        env: The environment for the child. Defaults to the current one.
        cwd: The working directory for the child.

    Returns:
        A context manager that yields a :class:`ToolProcess`.

    Raises:
        TypeError: If an argument has the wrong type (for example a single
            string instead of a list).
        ValueError: If the executable is not absolute, is a batch file, or an
            argument contains a NUL character.
        BackendUnavailableError: If the executable does not exist.
        ExternalToolError: If the process cannot be started; on exit, if it
            failed; on read, if it exceeded the output cap.
        ExternalToolTimeoutError: On exit, if it timed out.
        Exception: On exit, whatever the ``stdin`` file object raised while
            being read (the tool is killed first).
    """
    exe = _check_executable(executable)
    argv = [str(exe), *_normalize_args(args)]
    name = tool or exe.stem
    timeout = _timeout(timeout_seconds)
    if max_stdout_bytes is not None:
        _check_positive_int("max_stdout_bytes", max_stdout_bytes)
    _check_positive_int("max_stderr_bytes", max_stderr_bytes)

    popen_kwargs: dict[str, Any] = {
        "stdin": subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": dict(env) if env is not None else None,
        "cwd": os.fspath(cwd) if cwd is not None else None,
        "close_fds": True,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        popen_kwargs["start_new_session"] = True

    _log.debug("running %s with %d arguments", name, len(argv) - 1)
    try:
        proc = subprocess.Popen(argv, **popen_kwargs)  # no shell; argv is a list
    except FileNotFoundError as exc:
        raise BackendUnavailableError.missing_binary(name) from exc
    except OSError as exc:
        raise ExternalToolError(name, f"could not be started ({type(exc).__name__})") from exc

    try:
        process = ToolProcess(
            proc,
            tool=name,
            stdin=stdin,
            timeout_seconds=timeout,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
            check=check,
        )
    except BaseException:
        with contextlib.suppress(OSError):
            proc.kill()
        proc.wait()
        raise
    return _ToolContext(process)


def run_tool(
    executable: StrPath,
    args: Sequence[StrPath],
    *,
    tool: str | None = None,
    stdin: BytesLike | BinaryIO | None = None,
    timeout_seconds: float | None = None,
    max_stdout_bytes: int = DEFAULT_MAX_STDOUT_BYTES,
    max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES,
    check: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: StrPath | None = None,
) -> ToolResult:
    """Runs a tool to completion and captures its standard output.

    Args:
        executable: The absolute path of the executable.
        args: The arguments, without the executable.
        tool: The name used in messages. Defaults to the executable's stem.
        stdin: Bytes or a readable binary file object for standard input.
        timeout_seconds: The wall-clock limit. Defaults to
            ``MediaLimits().subprocess_timeout_seconds``.
        max_stdout_bytes: The cap on captured standard output; exceeding it
            kills the process and raises ``ExternalToolError``.
        max_stderr_bytes: How many trailing bytes of standard error to keep.
        check: Whether a non-zero exit status raises ``ExternalToolError``.
            With ``check=False`` the status is returned in the result.
        env: The environment for the child. Defaults to the current one.
        cwd: The working directory for the child.

    Returns:
        The result.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the executable is not absolute, is a batch file, or an
            argument contains a NUL character.
        BackendUnavailableError: If the executable does not exist.
        ExternalToolError: If the process cannot be started, fails (with
            ``check``), or exceeds the output cap.
        ExternalToolTimeoutError: If it times out.
    """
    with stream_tool(
        executable,
        args,
        tool=tool,
        stdin=stdin,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
        check=check,
        env=env,
        cwd=cwd,
    ) as proc:
        stdout = proc.read_all()
    returncode = proc.returncode
    if returncode is None:  # pragma: no cover - _finish waits or raises
        raise ExternalToolError(proc.tool, "did not exit")
    return ToolResult(
        tool=proc.tool,
        returncode=returncode,
        stdout=stdout,
        stderr_tail=proc.stderr_tail,
        elapsed_seconds=proc.elapsed_seconds,
    )


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _check_executable(executable: StrPath) -> Path:
    if not isinstance(executable, (str, os.PathLike)):
        raise TypeError(f"executable must be a path, got {type(executable).__name__}")
    text = os.fspath(executable)
    if not isinstance(text, str):
        raise TypeError("executable must be a text path, not bytes")
    if "\x00" in text:
        raise ValueError("executable path contains a NUL character")
    path = Path(text)
    if not path.is_absolute():
        raise ValueError(
            "executable must be an absolute path; locate it with ExternalTools.resolve()"
        )
    if sys.platform == "win32" and path.suffix.lower() in {".bat", ".cmd"}:
        raise ValueError("batch files cannot be run safely: cmd.exe re-parses their arguments")
    return path


def _normalize_args(args: Sequence[StrPath]) -> list[str]:
    if isinstance(args, (str, bytes)) or not isinstance(args, Sequence):
        raise TypeError("args must be a list of strings, not a single string")
    normalized: list[str] = []
    for arg in args:
        if not isinstance(arg, (str, os.PathLike)):
            raise TypeError(f"each argument must be a str or path, got {type(arg).__name__}")
        text = os.fspath(arg)
        if not isinstance(text, str):
            raise TypeError("each argument must be text, not bytes")
        if "\x00" in text:
            raise ValueError("an argument contains a NUL character")
        normalized.append(text)
    return normalized


def _timeout(timeout_seconds: float | None) -> float:
    if timeout_seconds is None:
        return MediaLimits().subprocess_timeout_seconds
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise TypeError("timeout_seconds must be a number")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    return float(timeout_seconds)


def _check_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
