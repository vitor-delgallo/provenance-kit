"""Model weight files: pinned SHA-256, offline loading, opt-in download (internal).

Rules (ARCHITECTURE.md, section 6, and decision 5):

* every file has a pinned SHA-256 that is verified before the file is loaded;
  a mismatch raises :class:`~provenance_kit.errors.ModelLoadError`;
* the directory is the caller's (``model_dir``); the default is a per-user
  cache, never ``site-packages``;
* nothing is downloaded unless the caller asks for it (``download=True``);
  downloads use HTTPS only (redirects included), are capped at the expected
  size, are hashed while streaming into a private temporary file in the
  target directory, and are moved into place only after the hash matched;
* no progress output: logging only (file names, never paths above DEBUG).

(The video package has the same private helper; subpackages do not import
each other, see ARCHITECTURE.md, section 1.)
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import os
import re
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from provenance_kit._logging import get_logger
from provenance_kit._typing import StrPath
from provenance_kit.errors import ModelLoadError

__all__ = [
    "MODEL_DIR_ENV",
    "ModelFile",
    "default_model_dir",
    "download_model_file",
    "resolve_model_file",
    "sha256_file",
]

_log = get_logger(__name__)

#: Environment variable that overrides the default model directory.
MODEL_DIR_ENV = "PROVENANCE_KIT_MODEL_DIR"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1024 * 1024
_DEFAULT_TIMEOUT = 60.0


@dataclass(frozen=True, slots=True)
class ModelFile:
    """A pinned model file.

    Attributes:
        name: The file name inside the model directory.
        url: The official HTTPS download URL.
        sha256: The expected SHA-256, lowercase hex.
        size: The expected size in bytes (caps the download).
    """

    name: str
    url: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        """Validates the fields."""
        if not self.url.startswith("https://"):
            raise ValueError("model URLs must use HTTPS")
        if not _SHA256_RE.match(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hex digits")
        if "/" in self.name or "\\" in self.name or self.name in ("", ".", ".."):
            raise ValueError("name must be a plain file name")
        if self.size < 1:
            raise ValueError("size must be positive")


def default_model_dir(subdir: str) -> Path:
    """Returns the default directory for a backend's model files.

    ``$PROVENANCE_KIT_MODEL_DIR/<subdir>`` when the variable is set, else
    ``~/.cache/provenance-kit/<subdir>``. The directory is not created here.

    Args:
        subdir: The backend's subdirectory, for example ``"audioseal"``.

    Returns:
        The directory path.
    """
    root = os.environ.get(MODEL_DIR_ENV, "").strip()
    base = Path(root) if root else Path.home() / ".cache" / "provenance-kit"
    return base / subdir


def sha256_file(path: Path) -> str:
    """Computes a file's SHA-256.

    Args:
        path: The file.

    Returns:
        The digest as lowercase hex.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_file(
    spec: ModelFile,
    *,
    model_path: StrPath | None,
    model_dir: StrPath | None,
    default_subdir: str,
    sha256: str | None,
    download: bool,
    timeout_seconds: float = _DEFAULT_TIMEOUT,
) -> Path:
    """Finds (or, if allowed, downloads) a model file and verifies its hash.

    Args:
        spec: The pinned file.
        model_path: An explicit file to use instead of ``model_dir / spec.name``.
        model_dir: The directory holding the file. Defaults to
            :func:`default_model_dir` with ``default_subdir``.
        default_subdir: The subdirectory used when ``model_dir`` is ``None``.
        sha256: The expected digest, or ``None`` for the pinned one. Pass
            another value only for weights you trained or converted yourself.
        download: Whether a missing file may be downloaded from ``spec.url``.
        timeout_seconds: The network timeout of each read.

    Returns:
        The path of the verified file.

    Raises:
        ValueError: If ``sha256`` is malformed, or ``model_path`` and
            ``download`` are both given.
        ModelLoadError: If the file is missing (and may not be downloaded),
            cannot be read or downloaded, or fails the hash check.
    """
    expected = spec.sha256 if sha256 is None else sha256.strip().lower()
    if not _SHA256_RE.match(expected):
        raise ValueError("sha256 must be 64 hex digits")
    if model_path is not None:
        if download:
            raise ValueError("download=True cannot be combined with an explicit model_path")
        path = Path(os.fspath(model_path))
    else:
        directory = (
            Path(os.fspath(model_dir))
            if model_dir is not None
            else default_model_dir(default_subdir)
        )
        path = directory / spec.name
        if not path.is_file():
            if not download:
                raise ModelLoadError(
                    f"model file {spec.name} was not found in the model directory; provision it "
                    "or pass download=True to fetch it from the official source"
                )
            if expected != spec.sha256:
                raise ModelLoadError(
                    f"model file {spec.name}: a custom sha256 cannot be used with download=True"
                )
            return download_model_file(spec, directory, timeout_seconds=timeout_seconds)
    if not path.is_file():
        raise ModelLoadError(f"model file {spec.name} was not found")
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise ModelLoadError(f"model file {spec.name} could not be read") from exc
    if actual != expected:
        raise ModelLoadError(f"model file {spec.name} failed its SHA-256 check")
    _log.debug("verified model file %s at %s", spec.name, path)
    return path


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
        if not newurl.startswith("https://"):
            raise urllib.error.URLError("refusing a redirect to a non-HTTPS URL")
        return super().redirect_request(req, fp, code, msg, headers, newurl)  # type: ignore[arg-type]


def download_model_file(
    spec: ModelFile,
    directory: Path,
    *,
    timeout_seconds: float = _DEFAULT_TIMEOUT,
) -> Path:
    """Downloads a pinned file into ``directory`` and verifies it.

    Args:
        spec: The pinned file.
        directory: The target directory (created if missing).
        timeout_seconds: The network timeout of each read.

    Returns:
        The path of the verified file, ``directory / spec.name``.

    Raises:
        ModelLoadError: If the download fails, exceeds the expected size or
            fails the hash check. Nothing is left behind on failure.
    """
    target = directory / spec.name
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelLoadError(f"model directory for {spec.name} could not be created") from exc
    _log.info("downloading model file %s", spec.name)
    opener = urllib.request.build_opener(_HttpsOnlyRedirects())
    request = urllib.request.Request(spec.url, headers={"User-Agent": "provenance-kit"})  # noqa: S310 - HTTPS checked
    fd, temp_name = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=directory)
    temp_path = Path(temp_name)
    try:
        digest = hashlib.sha256()
        received = 0
        with os.fdopen(fd, "wb") as out, opener.open(request, timeout=timeout_seconds) as response:
            while chunk := response.read(_CHUNK):
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
        _discard(temp_path)
        raise ModelLoadError(f"model file {spec.name} could not be downloaded") from exc
    except BaseException:
        _discard(temp_path)
        raise
    _log.info("downloaded and verified model file %s", spec.name)
    return target


def _discard(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()
