"""Helpers shared by the test suite.

* :func:`require_module` and :func:`require_tool` skip a test, with a reason
  that says what to install, when an optional dependency or binary is missing.
  Integration tests call them instead of failing. Setting
  ``PROVENANCE_KIT_TEST_REQUIRE`` to a comma-separated list of extras (and
  tool names) turns those skips into failures for the listed ones, so that
  an environment that is meant to have them (CI) cannot pass by skipping.
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
from typing import NoReturn

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

#: Comma-separated extras (or tool names) whose absence fails the test run
#: instead of skipping, for example ``c2pa,timestamps,image-hash,audio,video``.
REQUIRE_ENV = "PROVENANCE_KIT_TEST_REQUIRE"

#: The extras of pyproject.toml (``all`` and ``dev`` aside).
EXTRAS = frozenset(
    {
        "image-watermark",
        "image-hash",
        "pdq-native",
        "video",
        "video-watermark",
        "audio",
        "audio-watermark",
        "c2pa",
        "timestamps",
    }
)

#: The extras whose features run each binary: listing one requires the binary.
_TOOL_EXTRAS = {
    Tool.FFMPEG: ("video", "audio"),
    Tool.FFPROBE: ("video", "audio"),
    Tool.FPCALC: ("audio",),
}


def _required() -> frozenset[str]:
    """Returns the names listed in ``PROVENANCE_KIT_TEST_REQUIRE``, validated."""
    names = frozenset(
        name.strip() for name in os.environ.get(REQUIRE_ENV, "").split(",") if name.strip()
    )
    unknown = names - EXTRAS - {tool.value for tool in Tool}
    if unknown:
        raise ValueError(
            f"{REQUIRE_ENV} names unknown extras or tools: {', '.join(sorted(unknown))}"
        )
    return names


def _skip_or_fail(reason: str, *, required: bool) -> NoReturn:
    if required:
        pytest.fail(f"{reason} ({REQUIRE_ENV} requires it)", pytrace=False)
    pytest.skip(reason, allow_module_level=True)


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
    """Returns the tool's path or skips the test with a clear reason.

    The skip is a failure instead when ``PROVENANCE_KIT_TEST_REQUIRE`` lists
    the tool or an extra that needs it (``video``: ffmpeg and ffprobe;
    ``audio``: those and fpcalc).
    """
    required = _required()
    try:
        return find_tools().resolve(tool)
    except BackendUnavailableError:
        _skip_or_fail(
            f"needs the {tool.value} binary: set {ENV_PREFIX}{tool.name}, put it under "
            f"tools/{_TOOL_SUBDIRS[tool]}/, or install it on PATH",
            required=tool.value in required or not required.isdisjoint(_TOOL_EXTRAS[tool]),
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
    mistaken for "extra not installed". The skip is a failure instead when
    ``PROVENANCE_KIT_TEST_REQUIRE`` lists ``extra``.

    It may be called at module level (an integration file's import guard):
    the skip, or the failure, then applies to the whole module.
    """
    if name == "provenance_kit" or name.startswith("provenance_kit."):
        raise ValueError("provenance_kit modules are not optional: import them directly")
    if extra not in EXTRAS:
        raise ValueError(f"unknown extra {extra!r}")
    required = _required()
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (name == missing or name.startswith(missing + ".")):
            _skip_or_fail(
                f'needs {name}: pip install "provenance-kit[{extra}]"', required=extra in required
            )
        raise
