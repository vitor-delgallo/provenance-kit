"""The test-suite helpers themselves: skipping must never hide a real failure."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib

import pytest

from provenance_kit.tools import ExternalTools, Tool
from tests import support
from tests.support import REQUIRE_ENV, require_module, require_tool


@pytest.fixture(autouse=True)
def _no_required_extras(monkeypatch):
    """These tests set PROVENANCE_KIT_TEST_REQUIRE themselves (CI sets it for the run)."""
    monkeypatch.delenv(REQUIRE_ENV, raising=False)


def test_require_module_returns_installed_modules():
    assert require_module("json", extra="c2pa") is sys.modules["json"]


def test_require_module_skips_when_the_module_is_missing():
    with pytest.raises(pytest.skip.Exception, match=r"provenance-kit\[image-hash\]"):
        require_module("pk_no_such_module", extra="image-hash")
    with pytest.raises(pytest.skip.Exception):
        require_module("pk_no_such_module.submodule", extra="image-hash")  # missing parent


def test_require_module_fails_on_a_broken_module(tmp_path, monkeypatch):
    (tmp_path / "pk_broken_dependency.py").write_text("import pk_missing_transitive\n")
    (tmp_path / "pk_broken_code.py").write_text("raise RuntimeError('bug at import')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="pk_missing_transitive"):
        require_module("pk_broken_dependency", extra="c2pa")
    with pytest.raises(RuntimeError, match="bug at import"):
        require_module("pk_broken_code", extra="c2pa")


def test_require_module_refuses_provenance_kit_modules():
    with pytest.raises(ValueError, match="not optional"):
        require_module("provenance_kit.image.watermark", extra="image-watermark")


# ------------------------------------------------------------------ required extras


def test_require_module_refuses_unknown_extras():
    with pytest.raises(ValueError, match="unknown extra"):
        require_module("json", extra="c2pa-python")


def test_skips_allow_module_level_guards():
    with pytest.raises(pytest.skip.Exception) as info:
        require_module("pk_no_such_module", extra="c2pa")
    assert info.value.allow_module_level


def test_listed_extras_turn_missing_module_skips_into_failures(monkeypatch):
    monkeypatch.setenv(REQUIRE_ENV, " c2pa , timestamps ")
    with pytest.raises(pytest.fail.Exception, match=rf"provenance-kit\[c2pa\].*{REQUIRE_ENV}"):
        require_module("pk_no_such_module", extra="c2pa")
    with pytest.raises(pytest.skip.Exception):
        require_module("pk_no_such_module", extra="image-hash")  # not listed
    assert require_module("json", extra="c2pa") is sys.modules["json"]


def test_listed_extras_turn_missing_tool_skips_into_failures(monkeypatch):
    monkeypatch.setattr(
        support, "find_tools", lambda: ExternalTools(ffmpeg=None, ffprobe=None, search_path=False)
    )
    monkeypatch.delenv(REQUIRE_ENV, raising=False)
    for tool in Tool:
        with pytest.raises(pytest.skip.Exception):
            require_tool(tool)
    monkeypatch.setenv(REQUIRE_ENV, "video")  # ffmpeg and ffprobe, not fpcalc
    with pytest.raises(pytest.fail.Exception, match="ffmpeg"):
        require_tool(Tool.FFMPEG)
    with pytest.raises(pytest.fail.Exception, match="ffprobe"):
        require_tool(Tool.FFPROBE)
    with pytest.raises(pytest.skip.Exception):
        require_tool(Tool.FPCALC)
    for value in ("audio", "fpcalc"):
        monkeypatch.setenv(REQUIRE_ENV, value)
        with pytest.raises(pytest.fail.Exception, match="fpcalc"):
            require_tool(Tool.FPCALC)


def test_unknown_required_names_are_an_error(monkeypatch):
    monkeypatch.setenv(REQUIRE_ENV, "c2pa,timestamp")  # a typo must not disable the check
    with pytest.raises(ValueError, match="timestamp"):
        require_module("json", extra="c2pa")
    with pytest.raises(ValueError, match="timestamp"):
        require_tool(Tool.FFMPEG)


def test_known_extras_match_pyproject():
    with (support.REPO_ROOT / "pyproject.toml").open("rb") as handle:
        extras = set(tomllib.load(handle)["project"]["optional-dependencies"])
    assert extras - {"all", "dev"} == support.EXTRAS


def test_a_required_extra_fails_a_module_level_guard(tmp_path):
    """End to end: the module-level guard of an integration file fails the run."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="ascii")
    (tmp_path / "test_guarded.py").write_text(
        "from tests.support import require_module\n"
        "require_module('pk_no_such_module', extra='c2pa')\n"
        "def test_never_runs():\n"
        "    pass\n",
        encoding="ascii",
    )
    (tmp_path / "test_other.py").write_text("def test_runs():\n    pass\n", encoding="ascii")
    env = dict(os.environ, PYTHONPATH=str(support.REPO_ROOT), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)]
    env.pop(REQUIRE_ENV, None)
    skipped = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False
    )
    assert skipped.returncode == 0, skipped.stdout[-2000:]
    assert "1 passed, 1 skipped" in skipped.stdout
    env[REQUIRE_ENV] = "c2pa"
    failed = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False
    )
    assert failed.returncode != 0
    assert "1 error" in failed.stdout, failed.stdout[-2000:]
