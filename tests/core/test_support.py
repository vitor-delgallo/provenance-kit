"""The test-suite helpers themselves: skipping must never hide a real failure."""

from __future__ import annotations

import sys

import pytest

from tests.support import require_module


def test_require_module_returns_installed_modules():
    assert require_module("json", extra="x") is sys.modules["json"]


def test_require_module_skips_when_the_module_is_missing():
    with pytest.raises(pytest.skip.Exception, match=r"provenance-kit\[demo\]"):
        require_module("pk_no_such_module", extra="demo")
    with pytest.raises(pytest.skip.Exception):
        require_module("pk_no_such_module.submodule", extra="demo")  # missing parent


def test_require_module_fails_on_a_broken_module(tmp_path, monkeypatch):
    (tmp_path / "pk_broken_dependency.py").write_text("import pk_missing_transitive\n")
    (tmp_path / "pk_broken_code.py").write_text("raise RuntimeError('bug at import')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="pk_missing_transitive"):
        require_module("pk_broken_dependency", extra="demo")
    with pytest.raises(RuntimeError, match="bug at import"):
        require_module("pk_broken_code", extra="demo")


def test_require_module_refuses_provenance_kit_modules():
    with pytest.raises(ValueError, match="not optional"):
        require_module("provenance_kit.image.watermark", extra="image-watermark")
