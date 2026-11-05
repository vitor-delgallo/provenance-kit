"""Unit tests for the external tool runner, using the Python interpreter as the tool."""

from __future__ import annotations

import io
import logging
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from provenance_kit.errors import (
    BackendUnavailableError,
    ExternalToolError,
    ExternalToolTimeoutError,
)
from provenance_kit.tools import (
    ENV_PREFIX,
    ExternalTools,
    Tool,
    run_tool,
    stream_tool,
)

PY = Path(sys.executable)


def py(code: str) -> list[str]:
    """Arguments that make the interpreter run ``code``."""
    return ["-I", "-c", textwrap.dedent(code)]


ECHO = py("import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())")


# ------------------------------------------------------------------ run_tool


def test_run_tool_pipes_bytes_through_stdin_and_stdout(rng):
    data = rng.integers(0, 256, size=1_000_000, dtype="uint8").tobytes()
    result = run_tool(PY, ECHO, tool="echo", stdin=data, timeout_seconds=60)
    assert result.returncode == 0
    assert result.stdout == data
    assert result.tool == "echo"
    assert result.elapsed_seconds > 0


def test_run_tool_accepts_a_binary_file_object_as_stdin(rng):
    data = rng.integers(0, 256, size=300_000, dtype="uint8").tobytes()
    result = run_tool(PY, ECHO, stdin=io.BytesIO(data), timeout_seconds=60)
    assert result.stdout == data


def test_run_tool_without_stdin_uses_the_null_device():
    result = run_tool(PY, py("import sys; print(len(sys.stdin.buffer.read()))"), timeout_seconds=60)
    assert result.stdout.strip() == b"0"


def test_tool_that_ignores_its_input_is_not_an_error():
    result = run_tool(PY, py("pass"), stdin=b"x" * 5_000_000, timeout_seconds=60)
    assert result.returncode == 0


def test_tool_name_defaults_to_the_executable_stem():
    result = run_tool(PY, py("pass"), timeout_seconds=60)
    assert result.tool == PY.stem


def test_failure_message_is_sanitized_and_stderr_is_kept_aside():
    code = """
        import sys
        sys.stderr.write("Title: SECRET-MEDIA-TITLE\\n")
        sys.exit(3)
    """
    with pytest.raises(ExternalToolError) as info:
        run_tool(PY, py(code), tool="fake", timeout_seconds=60)
    err = info.value
    assert err.returncode == 3
    assert err.tool == "fake"
    assert "SECRET" not in str(err)
    assert "SECRET" not in repr(err)
    assert b"SECRET-MEDIA-TITLE" in err.stderr_tail
    assert str(err) == "fake: exited with status 3"


def test_check_false_returns_the_exit_status():
    result = run_tool(PY, py("import sys; sys.exit(5)"), check=False, timeout_seconds=60)
    assert result.returncode == 5


def test_stderr_is_drained_and_only_its_tail_is_kept():
    code = """
        import sys
        sys.stderr.write("a" * 2_000_000)
        sys.stderr.write("THE-END")
        sys.stderr.flush()
    """
    result = run_tool(PY, py(code), max_stderr_bytes=1000, timeout_seconds=60)
    assert len(result.stderr_tail) == 1000
    assert result.stderr_tail.endswith(b"THE-END")
    assert "<1000 bytes>" in repr(result)
    assert "aaaa" not in repr(result)


def test_stdout_over_the_cap_kills_the_process():
    code = """
        import sys
        while True:
            sys.stdout.buffer.write(b"x" * 65536)
    """
    start = time.monotonic()
    with pytest.raises(ExternalToolError, match="output exceeded the limit of 100000 bytes"):
        run_tool(PY, py(code), max_stdout_bytes=100_000, timeout_seconds=60)
    assert time.monotonic() - start < 30


def test_timeout_kills_the_process_and_raises():
    start = time.monotonic()
    with pytest.raises(ExternalToolTimeoutError) as info:
        run_tool(PY, py("import time; time.sleep(60)"), tool="sleeper", timeout_seconds=1.0)
    assert time.monotonic() - start < 20
    assert info.value.timeout_seconds == 1.0
    assert info.value.returncode is None
    assert str(info.value) == "sleeper: timed out after 1 s and was killed"


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = wintypes.DWORD()
        try:
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        finally:
            kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_timeout_kills_the_whole_process_tree():
    code = """
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(60)"])
        print(child.pid, flush=True)
        time.sleep(60)
    """
    grandchild = None
    with (
        pytest.raises(ExternalToolTimeoutError),
        stream_tool(PY, py(code), timeout_seconds=5.0) as proc,
    ):
        line = b""
        while not line.endswith(b"\n"):
            chunk = proc.read(1)
            if not chunk:
                break
            line += chunk
        grandchild = int(line)
        assert _pid_alive(grandchild)
        proc.read_all()  # blocks until the timeout kills the tree
    assert grandchild is not None
    deadline = time.monotonic() + 10
    while _pid_alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _pid_alive(grandchild)


def test_a_failing_stdin_source_propagates_its_own_error():
    class Broken(io.RawIOBase):
        def readable(self) -> bool:
            return True

        def read(self, size: int = -1) -> bytes:
            raise OSError("source went away")

    with pytest.raises(OSError, match="source went away"):
        run_tool(PY, ECHO, stdin=Broken(), timeout_seconds=60)  # type: ignore[arg-type]


def test_env_and_cwd_are_passed_to_the_child(tmp_path):
    code = "import os; print(os.getcwd()); print(os.environ.get('PK_TEST_VALUE'))"
    env = dict(os.environ, PK_TEST_VALUE="hello")
    result = run_tool(PY, py(code), env=env, cwd=tmp_path, timeout_seconds=60)
    lines = result.stdout.decode().splitlines()
    assert Path(lines[0]).resolve() == tmp_path.resolve()
    assert lines[1] == "hello"


def test_failure_details_are_logged_at_debug_only(caplog):
    code = "import sys; sys.stderr.write('detail-for-debugging'); sys.exit(1)"
    with caplog.at_level(logging.DEBUG, logger="provenance_kit"), pytest.raises(ExternalToolError):
        run_tool(PY, py(code), tool="fake", timeout_seconds=60)
    debug = [r for r in caplog.records if "detail-for-debugging" in r.getMessage()]
    assert debug
    assert all(r.levelno == logging.DEBUG for r in debug)


# ------------------------------------------------------------------ stream_tool


RECORDS = py(
    """
    import sys
    for i in range(10):
        sys.stdout.buffer.write(bytes([i]) * 100)
    """
)


def test_stream_reads_fixed_size_records_until_the_end():
    records = []
    with stream_tool(PY, RECORDS, timeout_seconds=60) as proc:
        while (record := proc.read_exact(100)) is not None:
            records.append(record)
    assert [r[0] for r in records] == list(range(10))
    assert all(len(r) == 100 for r in records)
    assert proc.returncode == 0


def test_stream_detects_a_truncated_record():
    with (
        pytest.raises(ExternalToolError, match="middle of a 300-byte record"),
        stream_tool(PY, RECORDS, timeout_seconds=60) as proc,
    ):
        while proc.read_exact(300) is not None:
            pass


def test_stream_checks_the_exit_status_after_a_full_read():
    code = "import sys; sys.stdout.write('partial'); sys.stdout.flush(); sys.exit(2)"
    with (
        pytest.raises(ExternalToolError) as info,
        stream_tool(PY, py(code), timeout_seconds=60) as proc,
    ):
        assert proc.read_all() == b"partial"
    assert info.value.returncode == 2


def test_stopping_early_kills_the_process_without_error():
    code = """
        import sys
        while True:
            sys.stdout.buffer.write(b"y" * 65536)
    """
    with stream_tool(PY, py(code), timeout_seconds=60) as proc:
        assert proc.read(10)
    assert proc.returncode is not None


def test_explicit_stop_is_not_an_error():
    with stream_tool(PY, py("import time; time.sleep(60)"), timeout_seconds=60) as proc:
        proc.stop()
    assert proc.returncode is not None


def test_an_exception_in_the_block_propagates_and_kills_the_process():
    with (
        pytest.raises(RuntimeError, match="consumer failed"),
        stream_tool(PY, py("import time; time.sleep(60)"), timeout_seconds=60) as proc,
    ):
        raise RuntimeError("consumer failed")
    assert proc.returncode is not None


def test_stream_total_output_cap():
    code = """
        import sys
        for _ in range(100):
            sys.stdout.buffer.write(b"z" * 10000)
    """
    with (
        pytest.raises(ExternalToolError, match="output exceeded"),
        stream_tool(PY, py(code), max_stdout_bytes=50_000, timeout_seconds=60) as proc,
    ):
        for _ in proc.iter_chunks(4096):
            pass


def test_stream_feeds_stdin_while_the_caller_reads(rng):
    data = rng.integers(0, 256, size=2_000_000, dtype="uint8").tobytes()
    with stream_tool(PY, ECHO, stdin=data, timeout_seconds=60) as proc:
        received = b"".join(proc.iter_chunks())
    assert received == data


# ------------------------------------------------------------------ validation


def test_a_single_string_is_refused_as_args():
    with pytest.raises(TypeError, match="not a single string"):
        run_tool(PY, "-c pass", timeout_seconds=60)  # type: ignore[arg-type]


def test_non_text_arguments_are_refused():
    with pytest.raises(TypeError):
        run_tool(PY, ["-c", 5], timeout_seconds=60)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        run_tool(PY, [b"-c"], timeout_seconds=60)  # type: ignore[list-item]


def test_nul_characters_are_refused():
    with pytest.raises(ValueError, match="NUL"):
        run_tool(PY, ["-c", "pass\x00"], timeout_seconds=60)


def test_relative_executables_are_refused():
    with pytest.raises(ValueError, match="absolute path"):
        run_tool("python", ["-c", "pass"], timeout_seconds=60)


@pytest.mark.skipif(sys.platform != "win32", reason="batch files are a Windows concern")
def test_batch_files_are_refused_on_windows(tmp_path):
    script = tmp_path / "tool.cmd"
    script.write_text("@echo off\n")
    with pytest.raises(ValueError, match="batch files"):
        run_tool(script, [], timeout_seconds=60)


def test_missing_executable_is_backend_unavailable(tmp_path):
    with pytest.raises(BackendUnavailableError) as info:
        run_tool(tmp_path / "does-not-exist", [], tool="ffmpeg", timeout_seconds=60)
    assert info.value.binary == "ffmpeg"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeouts_are_refused(timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        run_tool(PY, py("pass"), timeout_seconds=timeout)


def test_invalid_caps_are_refused():
    with pytest.raises(ValueError):
        run_tool(PY, py("pass"), max_stdout_bytes=0)
    with pytest.raises(TypeError):
        run_tool(PY, py("pass"), max_stderr_bytes=True)


def test_read_sizes_are_validated():
    with stream_tool(PY, py("pass"), timeout_seconds=60) as proc:
        with pytest.raises(ValueError):
            proc.read(0)
        with pytest.raises(ValueError):
            proc.read_exact(-1)
        proc.read_all()


# ------------------------------------------------------------------ ExternalTools


def _fake_executable(directory: Path, name: str) -> Path:
    path = directory / (name + (".exe" if sys.platform == "win32" else ""))
    path.write_bytes(b"")
    path.chmod(0o755)
    return path


def test_resolve_explicit_path(tmp_path):
    exe = _fake_executable(tmp_path, "ffmpeg")
    tools = ExternalTools(ffmpeg=exe, search_path=False)
    assert tools.resolve(Tool.FFMPEG) == exe.absolute()
    assert tools.resolve("ffmpeg") == exe.absolute()
    assert tools.available(Tool.FFMPEG)


def test_resolve_bare_name_on_path(tmp_path, monkeypatch):
    exe = _fake_executable(tmp_path, "my-ffmpeg")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert ExternalTools(ffmpeg="my-ffmpeg").resolve(Tool.FFMPEG) == exe


def test_resolve_searches_path_by_default(tmp_path, monkeypatch):
    exe = _fake_executable(tmp_path, "fpcalc")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert ExternalTools().resolve(Tool.FPCALC) == exe
    assert not ExternalTools(search_path=False).available(Tool.FPCALC)


def test_path_lookup_ignores_the_current_and_relative_directories(tmp_path, monkeypatch):
    _fake_executable(tmp_path, "ffprobe")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", os.pathsep.join([".", "relative", ""]))
    with pytest.raises(BackendUnavailableError) as info:
        ExternalTools().resolve(Tool.FFPROBE)
    assert info.value.binary == "ffprobe"
    assert f"{ENV_PREFIX}FFPROBE" in str(info.value)


def test_configured_path_that_is_not_a_file(tmp_path):
    tools = ExternalTools(ffmpeg=tmp_path / "missing" / "ffmpeg")
    with pytest.raises(BackendUnavailableError, match="not an executable file") as info:
        tools.resolve(Tool.FFMPEG)
    assert str(tmp_path) not in str(info.value)


def test_unknown_tool_name():
    with pytest.raises(ValueError):
        ExternalTools().resolve("convert")


def test_from_env_reads_prefixed_variables(tmp_path):
    exe = _fake_executable(tmp_path, "ffmpeg")
    tools = ExternalTools.from_env({f"{ENV_PREFIX}FFMPEG": str(exe), f"{ENV_PREFIX}FPCALC": " "})
    assert tools.ffmpeg == str(exe)
    assert tools.fpcalc is None
    assert tools.ffprobe is None


def test_external_tools_run_resolves_and_runs(monkeypatch):
    tools = ExternalTools(ffmpeg=PY, search_path=False)
    result = tools.run(Tool.FFMPEG, py("print('ok')"), timeout_seconds=60)
    assert result.tool == "ffmpeg"
    assert result.stdout.strip() == b"ok"
    with tools.stream("ffmpeg", py("print('streamed')"), timeout_seconds=60) as proc:
        assert proc.read_all().strip() == b"streamed"


def test_a_text_stdin_source_is_reported_as_a_type_error():
    with pytest.raises(TypeError, match="must return bytes"):
        run_tool(PY, ECHO, stdin=io.StringIO("text"), timeout_seconds=60)  # type: ignore[arg-type]
