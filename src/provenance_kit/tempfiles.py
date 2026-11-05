"""Private temporary files and directories, always deleted.

Media should travel through pipes (see :mod:`provenance_kit.tools`). When a
library or tool needs a real file, use these helpers:

* The directory is chosen by the caller (``directory=``), so an application can
  point it at a RAM-backed filesystem; ``None`` uses the platform default.
  Every public API that may create temporary files takes a
  ``temp_dir: StrPath | None = None`` argument and passes it through here.
* Files are created exclusively (``O_EXCL``) with owner-only permissions
  (``0o600`` for files and ``0o700`` for directories on POSIX). On Windows the
  permissions come from the directory's ACL; the default per-user temporary
  directory is private, a custom one must be secured by the caller.
* The file or directory is deleted when the ``with`` block exits, including on
  errors. A failed deletion is retried briefly and then logged as a warning
  (without the path above DEBUG level); it never masks the original error.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from provenance_kit._logging import get_logger
from provenance_kit._typing import BytesLike, StrPath

__all__ = ["private_temp_dir", "private_temp_file"]

_log = get_logger(__name__)

_DEFAULT_PREFIX = "pk-"
_DELETE_RETRIES = (0.05, 0.1, 0.25, 0.5)


@contextlib.contextmanager
def private_temp_file(
    *,
    suffix: str = "",
    prefix: str = _DEFAULT_PREFIX,
    directory: StrPath | None = None,
    data: BytesLike | None = None,
) -> Iterator[Path]:
    """Creates a private temporary file and deletes it on exit.

    Args:
        suffix: The file name suffix, for example ``".mp4"`` when a tool infers
            the format from the extension. Must not contain path separators.
        prefix: The file name prefix. Must not contain path separators.
        directory: Where to create the file. ``None`` uses the platform
            default temporary directory.
        data: Bytes to write before yielding. The file is closed either way,
            so another process can open it (required on Windows).

    Yields:
        The absolute path of the closed file.

    Raises:
        ValueError: If ``suffix`` or ``prefix`` contains a path separator or a
            NUL character, or ``directory`` is not an existing directory.
    """
    _check_name_part("suffix", suffix)
    _check_name_part("prefix", prefix)
    dir_text = _check_directory(directory)
    fd, name = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir_text)
    path = Path(name).absolute()
    try:
        try:
            if data is not None:
                view = memoryview(data).cast("B")
                while view:
                    written = os.write(fd, view)
                    view = view[written:]
        finally:
            os.close(fd)
        yield path
    finally:
        _delete(path, is_dir=False)


@contextlib.contextmanager
def private_temp_dir(
    *,
    prefix: str = _DEFAULT_PREFIX,
    directory: StrPath | None = None,
) -> Iterator[Path]:
    """Creates a private temporary directory and deletes it with its contents.

    Args:
        prefix: The directory name prefix. Must not contain path separators.
        directory: Where to create it. ``None`` uses the platform default.

    Yields:
        The absolute path of the directory.

    Raises:
        ValueError: If ``prefix`` contains a path separator or a NUL character,
            or ``directory`` is not an existing directory.
    """
    _check_name_part("prefix", prefix)
    dir_text = _check_directory(directory)
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=dir_text)).absolute()
    try:
        yield path
    finally:
        _delete(path, is_dir=True)


def _check_name_part(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str")
    separators = {"/", "\\", os.sep, "\x00"}
    if os.altsep:
        separators.add(os.altsep)
    if any(sep in value for sep in separators):
        raise ValueError(f"{name} must not contain path separators or NUL characters")


def _check_directory(directory: StrPath | None) -> str | None:
    if directory is None:
        return None
    text = os.fspath(directory)
    if not isinstance(text, str):
        raise TypeError("directory must be a text path")
    if not os.path.isdir(text):
        raise ValueError("the temporary directory does not exist")
    return text


def _delete(path: Path, *, is_dir: bool) -> None:
    for delay in (0.0, *_DELETE_RETRIES):
        if delay:
            # On Windows a file can stay locked for a moment after the process
            # that used it has exited (antivirus scanners, indexers).
            time.sleep(delay)
        try:
            if is_dir:
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            continue
        return
    _log.warning("could not delete a temporary %s", "directory" if is_dir else "file")
    _log.debug("undeleted temporary path: %s", path)
