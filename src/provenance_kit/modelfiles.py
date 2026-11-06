"""Pinned model files: where they live, how they are verified, how they are fetched.

Every model-backed adapter (TrustMark, VideoSeal, AudioSeal) loads its
weights through this module, so the rules of ARCHITECTURE.md (section 6,
"Model weights", and decision 5) are implemented once:

* **One directory convention.** A backend reads its files from the
  caller's ``model_dir``. Without one, it uses
  :func:`default_model_dir`: ``$PROVENANCE_KIT_MODEL_DIR/<backend>`` when the
  variable is set, else ``provenance-kit/models/<backend>`` in the per-user
  cache directory (``%LOCALAPPDATA%`` on Windows, ``~/Library/Caches`` on
  macOS, ``$XDG_CACHE_HOME`` or ``~/.cache`` elsewhere). A directory inside
  ``site-packages`` is refused.
* **Pinned files.** A :class:`ModelFile` records a file's official HTTPS
  URL, size and SHA-256. A file is used only after both match; a mismatch
  raises :class:`~provenance_kit.errors.ModelLoadError`.
* **Opt-in downloads.** Nothing here touches the network unless a download
  function is called. Downloads use HTTPS only (redirects included), a
  timeout per read, a total deadline (see below), a cap at the pinned size,
  and an atomic write: the bytes are hashed while they stream into a private
  temporary file in the target directory, which is renamed into place only
  after the size and hash matched. Nothing is left behind on failure.
* **A total deadline per file.** A server that sends one byte just inside
  every read timeout would otherwise keep a download alive forever. Each
  file must arrive within ``max(10 x timeout_seconds, size /
  DOWNLOAD_MIN_BYTES_PER_SECOND)`` seconds (64 KiB/s: about 27 minutes for
  100 MiB); a watchdog shuts the connection down when the deadline passes.
  The deadline is checked again once a connection is open, because name
  resolution cannot be interrupted. On a slower link, raise
  ``timeout_seconds`` or provision the files yourself.
* **Quiet and safe to log.** Progress goes to logging only. Messages name the
  file, never its directory.

Example::

    from provenance_kit.modelfiles import ModelFile, default_model_dir, ensure_model_files

    WEIGHTS = ModelFile(
        name="model.pt",
        url="https://example.org/model.pt",
        sha256="...",
        size=123_456,
    )
    ensure_model_files([WEIGHTS], default_model_dir("example"))  # deployment, network
    data = read_model_file(WEIGHTS, default_model_dir("example"))  # worker, offline

Thread safety: the functions are thread-safe. Concurrent downloads of the
same file write separate temporary files; the last rename wins, and every
candidate has passed the hash check.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import math
import os
import re
import site
import socket
import ssl
import sys
import sysconfig
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from provenance_kit._logging import get_logger
from provenance_kit._typing import StrPath
from provenance_kit.errors import ModelLoadError

__all__ = [
    "DEFAULT_DOWNLOAD_TIMEOUT_SECONDS",
    "DOWNLOAD_MIN_BYTES_PER_SECOND",
    "MODEL_DIR_ENV",
    "ModelFile",
    "default_model_dir",
    "download_model_file",
    "ensure_model_files",
    "find_model_file",
    "model_directory",
    "read_model_file",
    "verify_model_file",
]

_log = get_logger(__name__)

#: Environment variable naming the root directory of the model files; each
#: backend uses its own subdirectory (``trustmark``, ``videoseal``,
#: ``audioseal``).
MODEL_DIR_ENV = "PROVENANCE_KIT_MODEL_DIR"

#: Default network timeout of each read during a download, in seconds.
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60.0

#: The slowest average rate a download may have, in bytes per second. With
#: the read timeout, it sets each file's total deadline:
#: ``max(10 * timeout_seconds, size / DOWNLOAD_MIN_BYTES_PER_SECOND)``.
DOWNLOAD_MIN_BYTES_PER_SECOND = 64 * 1024

# The deadline is never shorter than this many read timeouts.
_DEADLINE_TIMEOUTS = 10

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BACKEND_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_CHUNK = 1024 * 1024
_USER_AGENT = "provenance-kit"


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelFile:
    """A pinned model file.

    Attributes:
        name: The file name inside the model directory.
        url: The official HTTPS download URL.
        sha256: The expected SHA-256, 64 lowercase hex digits.
        size: The expected size in bytes (also caps a download).

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the URL is not HTTPS, the name is not a plain file
            name, the digest is malformed or the size is not positive.
    """

    name: str
    url: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        """Validates the fields."""
        for field_name in ("name", "url", "sha256"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a str")
        if not self.url.startswith("https://"):
            raise ValueError("model URLs must use HTTPS")
        if not _SHA256_RE.match(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hex digits")
        if (
            any(sep in self.name for sep in ("/", "\\", "\x00"))
            or self.name in ("", ".", "..")
            or self.name != self.name.strip()
        ):
            raise ValueError("name must be a plain file name")
        if isinstance(self.size, bool) or not isinstance(self.size, int):
            raise TypeError("size must be an int")
        if self.size < 1:
            raise ValueError("size must be positive")


# --------------------------------------------------------------------------
# Directories
# --------------------------------------------------------------------------


def default_model_dir(backend: str, *, environ: Mapping[str, str] | None = None) -> Path:
    """Returns the default directory of a backend's model files.

    ``$PROVENANCE_KIT_MODEL_DIR/<backend>`` when the variable is set (and not
    empty); otherwise ``provenance-kit/models/<backend>`` in the per-user
    cache directory: ``%LOCALAPPDATA%`` on Windows, ``~/Library/Caches`` on
    macOS, ``$XDG_CACHE_HOME`` (default ``~/.cache``) elsewhere. The directory
    is not created.

    Args:
        backend: The backend's subdirectory name, for example
            ``"videoseal"`` (lowercase letters, digits, ``.``, ``_``, ``-``).
        environ: The environment to read. Defaults to ``os.environ``.

    Returns:
        The directory path (it may not exist yet).

    Raises:
        ValueError: If ``backend`` is not a valid subdirectory name.
    """
    name: object = backend
    if not isinstance(name, str) or not _BACKEND_RE.match(name):
        raise ValueError("backend must be a lowercase directory name such as 'videoseal'")
    env = os.environ if environ is None else environ
    configured = env.get(MODEL_DIR_ENV, "").strip()
    if configured:
        return Path(configured) / backend
    if sys.platform == "win32":
        local = env.get("LOCALAPPDATA", "").strip()
        base = Path(local) if local else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        xdg = env.get("XDG_CACHE_HOME", "").strip()
        base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "provenance-kit" / "models" / backend


def model_directory(model_dir: StrPath | None, backend: str) -> Path:
    """Returns the absolute model directory a backend should use.

    Args:
        model_dir: The caller's directory, or ``None`` for
            :func:`default_model_dir`.
        backend: The backend's subdirectory name (used for the default).

    Returns:
        The absolute, resolved directory (it may not exist yet).

    Raises:
        TypeError: If ``model_dir`` is not a path.
        ValueError: If the directory is inside ``site-packages`` (packages
            such as ``trustmark`` download there by default; model files
            belong to the deployment, not to an installed package).
    """
    if model_dir is None:
        directory = default_model_dir(backend)
    else:
        if not isinstance(model_dir, (str, os.PathLike)):
            raise TypeError("model_dir must be a path")
        directory = Path(os.fspath(model_dir))
    resolved = directory.resolve()
    for root in _site_roots():
        if resolved == root or root in resolved.parents:
            raise ValueError("the model directory must not be inside site-packages")
    return resolved


def _site_roots() -> set[Path]:
    candidates: list[str] = []
    paths = sysconfig.get_paths()
    candidates += [paths.get("purelib", ""), paths.get("platlib", "")]
    getsitepackages = getattr(site, "getsitepackages", None)
    if getsitepackages is not None:
        candidates += list(getsitepackages())
    getusersitepackages = getattr(site, "getusersitepackages", None)
    if getusersitepackages is not None:
        candidates.append(getusersitepackages())
    return {Path(c).resolve() for c in candidates if c}


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify_model_file(path: StrPath, spec: ModelFile, *, sha256: str | None = None) -> Path:
    """Checks a file against its pinned size and SHA-256.

    Args:
        path: The file to check.
        spec: The pinned file.
        sha256: A different expected digest, for weights you trained or
            converted yourself; the size is then not checked. ``None`` uses
            the pinned size and digest.

    Returns:
        The path, as a :class:`~pathlib.Path`.

    Raises:
        ValueError: If ``sha256`` is malformed.
        ModelLoadError: If the file is missing, cannot be read, or fails the
            size or hash check. The message names ``spec.name`` only.
    """
    expected, check_size = _expected_digest(spec, sha256)
    file = Path(os.fspath(path))
    try:
        if not file.is_file():
            raise ModelLoadError(f"model file {spec.name} was not found")
        if check_size and file.stat().st_size != spec.size:
            raise ModelLoadError(f"model file {spec.name} failed its SHA-256 check (wrong size)")
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            while chunk := handle.read(_CHUNK):
                digest.update(chunk)
    except OSError as exc:
        raise ModelLoadError(f"model file {spec.name} could not be read") from exc
    if digest.hexdigest() != expected:
        raise ModelLoadError(f"model file {spec.name} failed its SHA-256 check")
    _log.debug("verified model file %s at %s", spec.name, file)
    return file


def read_model_file(spec: ModelFile, directory: StrPath) -> bytes:
    """Reads a pinned file into memory and verifies the bytes read.

    Loading a model from the returned bytes (rather than from the path)
    guarantees that what is loaded is what was verified, even if the file is
    replaced in the meantime. At most ``spec.size + 1`` bytes are read.

    Args:
        spec: The pinned file.
        directory: The model directory.

    Returns:
        The verified content.

    Raises:
        ModelLoadError: If the file is missing, cannot be read or fails its
            size or hash check.
    """
    path = Path(os.fspath(directory)) / spec.name
    try:
        with path.open("rb") as handle:
            data = handle.read(spec.size + 1)
    except FileNotFoundError:
        raise ModelLoadError(_missing_message(spec)) from None
    except OSError as exc:
        raise ModelLoadError(f"model file {spec.name} could not be read") from exc
    if len(data) != spec.size or hashlib.sha256(data).hexdigest() != spec.sha256:
        raise ModelLoadError(f"model file {spec.name} failed its SHA-256 check")
    return data


def find_model_file(
    spec: ModelFile,
    *,
    backend: str,
    model_dir: StrPath | None = None,
    model_path: StrPath | None = None,
    sha256: str | None = None,
    download: bool = False,
    timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:
    """Locates a model file, downloads it if allowed, and verifies it.

    Args:
        spec: The pinned file.
        backend: The backend's subdirectory name (for the default directory).
        model_dir: The model directory; ``None`` uses
            :func:`default_model_dir`.
        model_path: An explicit file to use instead of
            ``model_dir / spec.name``.
        sha256: A different expected digest (see :func:`verify_model_file`).
        download: Whether a missing file may be fetched from ``spec.url``.
        timeout_seconds: The network timeout of each read; the download also
            has a total deadline (see :func:`download_model_file`).

    Returns:
        The path of the verified file.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``sha256`` is malformed, ``model_path`` is combined
            with ``download=True``, or the directory is inside
            ``site-packages``.
        ModelLoadError: If the file is missing (and may not be downloaded),
            cannot be read or downloaded, or fails its check.
    """
    _expected_digest(spec, sha256)
    if model_path is not None:
        if download:
            raise ValueError("download=True cannot be combined with an explicit model_path")
        if not isinstance(model_path, (str, os.PathLike)):
            raise TypeError("model_path must be a path")
        return verify_model_file(model_path, spec, sha256=sha256)
    directory = model_directory(model_dir, backend)
    path = directory / spec.name
    if not path.is_file():
        if not download:
            raise ModelLoadError(_missing_message(spec))
        if sha256 is not None and sha256.strip().lower() != spec.sha256:
            raise ModelLoadError(
                f"model file {spec.name}: a custom sha256 cannot be used with download=True"
            )
        return download_model_file(spec, directory, timeout_seconds=timeout_seconds)
    return verify_model_file(path, spec, sha256=sha256)


def _expected_digest(spec: ModelFile, sha256: str | None) -> tuple[str, bool]:
    if not isinstance(spec, ModelFile):
        raise TypeError("spec must be a ModelFile")
    if sha256 is None:
        return spec.sha256, True
    if not isinstance(sha256, str):
        raise TypeError("sha256 must be a str")
    expected = sha256.strip().lower()
    if not _SHA256_RE.match(expected):
        raise ValueError("sha256 must be 64 hex digits")
    return expected, expected == spec.sha256


def _missing_message(spec: ModelFile) -> str:
    return (
        f"model file {spec.name} is not in the model directory: provision it, or download it "
        "explicitly (download=True or the backend's download function)"
    )


# --------------------------------------------------------------------------
# Downloads (opt-in)
# --------------------------------------------------------------------------


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(  # noqa: PLR0917 - signature inherited from urllib
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        if not newurl.lower().startswith("https://"):
            raise urllib.error.URLError("refusing a redirect away from HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)  # type: ignore[arg-type]


def download_model_file(
    spec: ModelFile,
    directory: StrPath,
    *,
    timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:
    """Downloads a pinned file into ``directory`` and verifies it.

    The file is fetched from ``spec.url`` over HTTPS (redirects must stay on
    HTTPS), hashed while it streams into a private temporary file in
    ``directory``, and renamed to ``spec.name`` only when its size and
    SHA-256 match. An existing file of that name is replaced. Besides the
    timeout of each read, the whole download has a deadline of
    ``max(10 * timeout_seconds, spec.size / DOWNLOAD_MIN_BYTES_PER_SECOND)``
    seconds.

    Args:
        spec: The pinned file.
        directory: The target directory (created if missing). It is used as
            given; resolve it with :func:`model_directory` first.
        timeout_seconds: The network timeout of each read; ten times this is
            also the shortest total deadline.

    Returns:
        The path of the verified file, ``directory / spec.name``.

    Raises:
        TypeError: If ``timeout_seconds`` is not a number.
        ValueError: If ``timeout_seconds`` is not positive and finite.
        ModelLoadError: If the download fails, misses its deadline, exceeds
            the pinned size or fails the hash check. Nothing is left behind
            on failure.
    """
    timeout = _check_timeout(timeout_seconds)
    target_dir = Path(os.fspath(directory))
    target = target_dir / spec.name
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelLoadError(f"the model directory for {spec.name} could not be created") from exc
    _log.info("downloading model file %s (%d bytes)", spec.name, spec.size)
    deadline = _Deadline(_download_deadline(spec.size, timeout))
    opener = urllib.request.build_opener(
        _HttpsOnlyRedirects(), _DeadlineHTTPSHandler(deadline, _tls_context())
    )
    request = urllib.request.Request(spec.url, headers={"User-Agent": _USER_AGENT})  # noqa: S310 - HTTPS checked in ModelFile
    try:
        fd, temp_name = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=target_dir)
    except OSError as exc:
        raise ModelLoadError(f"model file {spec.name} could not be written") from exc
    temp_path = Path(temp_name)
    try:
        deadline.start()
        with os.fdopen(fd, "wb") as out, opener.open(request, timeout=timeout) as response:
            received, digest = _stream(response, out, spec, deadline)
        if received != spec.size or digest != spec.sha256:
            deadline.check()  # cut short by the watchdog: a timeout, not a bad file
            raise ModelLoadError(f"downloaded model file {spec.name} failed its SHA-256 check")
        os.replace(temp_path, target)
    except ModelLoadError:
        _discard(temp_path)
        raise
    except (OSError, http.client.HTTPException, ValueError) as exc:
        # URLError, HTTPError, socket timeouts and the deadline are OSErrors.
        _discard(temp_path)
        if deadline.expired:
            raise ModelLoadError(
                f"model file {spec.name} could not be downloaded in time "
                f"(deadline {deadline.seconds:g} s)"
            ) from exc
        raise ModelLoadError(
            f"model file {spec.name} could not be downloaded ({type(exc).__name__})"
        ) from exc
    except BaseException:
        _discard(temp_path)
        raise
    finally:
        deadline.cancel()
    _log.info("downloaded and verified model file %s", spec.name)
    return target


def _stream(response: Any, out: IO[bytes], spec: ModelFile, deadline: _Deadline) -> tuple[int, str]:
    """Copies the body to ``out``, at most ``spec.size + 1`` bytes; returns size and SHA-256."""
    digest = hashlib.sha256()
    received = 0
    while True:
        deadline.check()
        # read1 returns what has arrived, so the deadline is checked between
        # short reads too; the watchdog covers a read that blocks.
        chunk = response.read1(min(_CHUNK, spec.size + 1 - received))
        if not chunk:
            return received, digest.hexdigest()
        received += len(chunk)
        if received > spec.size:
            raise ModelLoadError(f"model file {spec.name} is larger than expected")
        digest.update(chunk)
        out.write(chunk)


def _download_deadline(size: int, timeout: float) -> float:
    """Returns the total deadline of a download, in seconds."""
    return max(_DEADLINE_TIMEOUTS * timeout, size / DOWNLOAD_MIN_BYTES_PER_SECOND)


def _tls_context() -> ssl.SSLContext:
    """Returns the TLS context of downloads: the system trust store, hostnames checked."""
    return ssl.create_default_context()


class _Deadline:
    """The total deadline of one download; a watchdog shuts its sockets down."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._end = time.monotonic() + seconds
        self._lock = threading.Lock()
        self._sockets: list[socket.socket] = []
        self._fired = threading.Event()
        self._timer = threading.Timer(seconds, self._fire)
        self._timer.daemon = True

    @property
    def expired(self) -> bool:
        """Whether the deadline has passed."""
        return self._fired.is_set() or time.monotonic() >= self._end

    def start(self) -> None:
        """Starts the watchdog."""
        self._timer.start()

    def cancel(self) -> None:
        """Stops the watchdog (the download is over)."""
        self._timer.cancel()

    def check(self) -> None:
        """Raises :class:`TimeoutError` once the deadline has passed."""
        if self.expired:
            raise TimeoutError("the download deadline passed")

    def watch(self, sock: socket.socket) -> None:
        """Registers a connected socket, or refuses it after the deadline."""
        with self._lock:
            if self._fired.is_set():
                raise TimeoutError("the download deadline passed")
            self._sockets.append(sock)

    def _fire(self) -> None:
        with self._lock:
            self._fired.set()
            sockets = list(self._sockets)
        for sock in sockets:
            # socket.socket.shutdown, not SSLSocket.shutdown: the TLS object
            # must stay intact while the downloading thread reads through it.
            with contextlib.suppress(OSError):
                socket.socket.shutdown(sock, socket.SHUT_RDWR)


class _DeadlineHTTPSConnection(http.client.HTTPSConnection):
    """An HTTPS connection whose socket the download's watchdog can shut down."""

    def __init__(self, host: str, *, deadline: _Deadline, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self._deadline = deadline

    def connect(self) -> None:
        """Connects (name lookup, TCP, proxy tunnel, TLS), then joins the deadline."""
        self._deadline.check()
        super().connect()
        self._deadline.watch(self.sock)


class _DeadlineHTTPSHandler(urllib.request.HTTPSHandler):
    """Opens HTTPS connections that belong to one download's deadline."""

    def __init__(self, deadline: _Deadline, context: ssl.SSLContext) -> None:
        super().__init__(context=context)
        self._deadline = deadline
        self._tls = context

    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        deadline = self._deadline

        def connection(host: str, **kwargs: Any) -> http.client.HTTPConnection:
            return _DeadlineHTTPSConnection(host, deadline=deadline, **kwargs)

        return self.do_open(connection, req, context=self._tls)


def ensure_model_files(
    specs: Iterable[ModelFile],
    directory: StrPath,
    *,
    timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> tuple[Path, ...]:
    """Makes sure every file is present and valid, downloading the others.

    Files already present with the pinned size and SHA-256 are kept (no
    network); missing or invalid ones are downloaded with
    :func:`download_model_file`, which replaces an invalid file only after
    the new one verified.

    Args:
        specs: The pinned files.
        directory: The target directory (created if needed). It is used as
            given; resolve it with :func:`model_directory` first.
        timeout_seconds: The network timeout of each read; each file also
            has a total deadline (see :func:`download_model_file`).

    Returns:
        The path of each verified file, in the order of ``specs``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``timeout_seconds`` is not positive and finite.
        ModelLoadError: If a download fails or a file fails its check.
    """
    _check_timeout(timeout_seconds)
    target_dir = Path(os.fspath(directory))
    paths: list[Path] = []
    for spec in specs:
        if not isinstance(spec, ModelFile):
            raise TypeError("specs must contain ModelFile items")
        path = target_dir / spec.name
        try:
            paths.append(verify_model_file(path, spec))
            continue
        except ModelLoadError:
            pass
        paths.append(download_model_file(spec, target_dir, timeout_seconds=timeout_seconds))
    return tuple(paths)


def _check_timeout(timeout_seconds: object) -> float:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise TypeError("timeout_seconds must be a number")
    try:
        finite = math.isfinite(timeout_seconds)
    except OverflowError:  # an int beyond the float range
        finite = False
    if not finite or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    return float(timeout_seconds)


def _discard(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()
