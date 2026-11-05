"""Helpers shared by the test suite.

* :func:`require_module` and :func:`require_tool` skip a test, with a reason
  that says what to install, when an optional dependency or binary is missing.
  Integration tests call them instead of failing.
* :func:`find_tools` locates ffmpeg, ffprobe and fpcalc for tests: the
  ``PROVENANCE_KIT_<TOOL>`` environment variables first, then the repository's
  ``tools/`` directory (see docs/DEVELOPMENT.md), then ``PATH``.
* :func:`model_dir` is where the integration tests keep each backend's model
  files: ``$PROVENANCE_KIT_MODEL_DIR/<backend>`` when the variable is set
  (the library's own convention), else the repository's git-ignored
  ``.cache/models/<backend>``.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from provenance_kit.errors import BackendUnavailableError
from provenance_kit.modelfiles import MODEL_DIR_ENV, default_model_dir
from provenance_kit.tools import ENV_PREFIX, ExternalTools, Tool

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"

#: The repository's model root (git-ignored), used when
#: ``PROVENANCE_KIT_MODEL_DIR`` is not set. Same layout: one subdirectory per
#: backend (``trustmark``, ``videoseal``, ``audioseal``).
MODEL_ROOT = REPO_ROOT / ".cache" / "models"

#: Where each binary lives under ``tools/`` (see docs/DEVELOPMENT.md).
_TOOL_SUBDIRS = {Tool.FFMPEG: "ffmpeg", Tool.FFPROBE: "ffmpeg", Tool.FPCALC: "fpcalc"}

#: A fixed seed for every random generator in the tests.
SEED = 20261105


def _repo_tool(tool: Tool) -> Path | None:
    name = tool.value + (".exe" if sys.platform == "win32" else "")
    subdir = TOOLS_DIR / _TOOL_SUBDIRS[tool]
    for candidate in (subdir / name, subdir / "bin" / name):
        if candidate.is_file():
            return candidate
    return None


def find_tools() -> ExternalTools:
    """Returns an ExternalTools configuration for the test run."""
    paths: dict[str, str | None] = {}
    for tool in Tool:
        from_env = os.environ.get(ENV_PREFIX + tool.name, "").strip()
        repo = _repo_tool(tool)
        paths[tool.value] = from_env or (str(repo) if repo else None)
    return ExternalTools(
        ffmpeg=paths["ffmpeg"],
        ffprobe=paths["ffprobe"],
        fpcalc=paths["fpcalc"],
        search_path=True,
    )


def require_tool(tool: Tool) -> Path:
    """Returns the tool's path or skips the test with a clear reason."""
    try:
        return find_tools().resolve(tool)
    except BackendUnavailableError:
        pytest.skip(
            f"needs the {tool.value} binary: set {ENV_PREFIX}{tool.name}, put it under "
            f"tools/{_TOOL_SUBDIRS[tool]}/, or install it on PATH"
        )


def model_dir(backend: str) -> Path:
    """Returns the directory of a backend's model files for the test run."""
    if os.environ.get(MODEL_DIR_ENV, "").strip():
        return default_model_dir(backend)
    return MODEL_ROOT / backend


#: Modules that importing provenance-kit must never load as a side effect.
HEAVY_MODULES = (
    "torch",
    "torchvision",
    "trustmark",
    "videoseal",
    "audioseal",
    "imagehash",
    "pdqhash",
    "c2pa",
    "opentimestamps",
    "scipy",
    "cryptography",
    "bitcoin",
)


def assert_imports_lightly(
    modules: list[str],
    forbidden: tuple[str, ...] = HEAVY_MODULES,
) -> None:
    """Imports ``modules`` in a fresh interpreter and fails if a heavy module loads.

    Every module agent calls this for its public modules, so that importing
    ``provenance_kit.<area>.<module>`` stays cheap until a backend is used.
    """
    code = (
        "import importlib, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        f"print(','.join(m for m in {list(forbidden)!r} if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    loaded = result.stdout.strip()
    assert not loaded, f"importing {modules} loaded heavy modules: {loaded}"


def require_module(name: str, *, extra: str) -> ModuleType:
    """Imports an optional module or skips the test with a clear reason.

    Only a missing ``name`` (or a missing parent package of it) skips. Every
    other import failure propagates and fails the test: a missing transitive
    dependency, a broken native library, or an error inside the module, and
    above all a broken ``provenance_kit`` import, which must never be
    mistaken for "extra not installed".
    """
    if name == "provenance_kit" or name.startswith("provenance_kit."):
        raise ValueError("provenance_kit modules are not optional: import them directly")
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (name == missing or name.startswith(missing + ".")):
            pytest.skip(f'needs {name}: pip install "provenance-kit[{extra}]"')
        raise
