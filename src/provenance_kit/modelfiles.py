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
  timeout per read, a cap at the pinned size, and an atomic write: the bytes
  are hashed while they stream into a private temporary file in the target
  directory, which is renamed into place only after the size and hash
  matched. Nothing is left behind on failure.
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
import sys
import sysconfig
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from provenance_kit._logging import get_logger
from provenance_kit._typing import StrPath
from provenance_kit.errors import ModelLoadError

__all__ = [
    "DEFAULT_DOWNLOAD_TIMEOUT_SECONDS",
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
        timeout_seconds: The network timeout of each read.

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
    SHA-256 match. An existing file of that name is replaced.

    Args:
        spec: The pinned file.
        directory: The target directory (created if missing). It is used as
            given; resolve it with :func:`model_directory` first.
        timeout_seconds: The network timeout of each read.

    Returns:
        The path of the verified file, ``directory / spec.name``.

    Raises:
        TypeError: If ``timeout_seconds`` is not a number.
        ValueError: If ``timeout_seconds`` is not positive and finite.
        ModelLoadError: If the download fails, exceeds the pinned size or
            fails the hash check. Nothing is left behind on failure.
    """
    timeout = _check_timeout(timeout_seconds)
    target_dir = Path(os.fspath(directory))
    target = target_dir / spec.name
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelLoadError(f"the model directory for {spec.name} could not be created") from exc
    _log.info("downloading model file %s (%d bytes)", spec.name, spec.size)
    opener = urllib.request.build_opener(_HttpsOnlyRedirects())
    request = urllib.request.Request(spec.url, headers={"User-Agent": _USER_AGENT})  # noqa: S310 - HTTPS checked in ModelFile
    try:
        fd, temp_name = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=target_dir)
    except OSError as exc:
        raise ModelLoadError(f"model file {spec.name} could not be written") from exc
    temp_path = Path(temp_name)
    try:
        digest = hashlib.sha256()
        received = 0
        with os.fdopen(fd, "wb") as out, opener.open(request, timeout=timeout) as response:
            while chunk := response.read(min(_CHUNK, spec.size + 1 - received)):
                received += len(chunk)
                if received > spec.size:
                    raise ModelLoadError(f"model file {spec.name} is larger than expected")
                digest.update(chunk)
                out.write(chunk)
        if received != spec.size or digest.hexdigest() != spec.sha256:
            raise ModelLoadError(f"downloaded model file {spec.name} failed its SHA-256 check")
        os.replace(temp_path, target)
    except ModelLoadError:
        _discard(temp_path)
        raise
    except (OSError, http.client.HTTPException, ValueError) as exc:
        # URLError, HTTPError and socket timeouts are OSErrors.
        _discard(temp_path)
        raise ModelLoadError(
            f"model file {spec.name} could not be downloaded ({type(exc).__name__})"
        ) from exc
    except BaseException:
        _discard(temp_path)
        raise
    _log.info("downloaded and verified model file %s", spec.name)
    return target


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
        timeout_seconds: The network timeout of each read.

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
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    return float(timeout_seconds)


def _discard(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()
