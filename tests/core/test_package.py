"""Package-level contracts: version, import cost, logging, errors, lazy imports."""

from __future__ import annotations

import importlib.metadata
import logging
import sys
import types

import pytest

import provenance_kit
from provenance_kit import errors
from provenance_kit._logging import LOGGER_NAME, get_logger
from provenance_kit._optional import import_optional
from tests.support import HEAVY_MODULES, assert_imports_lightly


def test_version_is_single_sourced():
    assert provenance_kit.__version__ == "0.1.0"
    assert importlib.metadata.version("provenance-kit") == provenance_kit.__version__


CORE_MODULES = [
    "provenance_kit",
    "provenance_kit.image",
    "provenance_kit.video",
    "provenance_kit.audio",
    "provenance_kit.provenance",
    *(
        f"provenance_kit.{name}"
        for name in (
            "_checks",
            "_ffmpeg",
            "bits",
            "detection",
            "device",
            "errors",
            "limits",
            "matching",
            "mediatype",
            "modelfiles",
            "payload",
            "tempfiles",
            "tools",
        )
    ),
]


def test_importing_the_core_does_not_import_heavy_dependencies():
    # The core does not even need the light core dependencies at import time.
    assert_imports_lightly(CORE_MODULES, forbidden=(*HEAVY_MODULES, "numpy", "PIL"))


def test_library_logger_has_a_null_handler_and_no_level():
    logger = logging.getLogger(LOGGER_NAME)
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
    assert logger.level == logging.NOTSET


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (None, "provenance_kit"),
        ("provenance_kit", "provenance_kit"),
        ("video.frames", "provenance_kit.video.frames"),
        ("provenance_kit.tools", "provenance_kit.tools"),
    ],
)
def test_get_logger_names(name, expected):
    assert get_logger(name).name == expected


def test_error_hierarchy():
    base = errors.ProvenanceKitError
    for cls in (
        errors.BackendUnavailableError,
        errors.MediaError,
        errors.ExternalToolError,
        errors.ModelLoadError,
        errors.PayloadError,
        errors.WatermarkBackendError,
    ):
        assert issubclass(cls, base)
    assert not issubclass(errors.WatermarkBackendError, (errors.MediaError, errors.ModelLoadError))
    assert set(errors.__all__) <= set(provenance_kit.__all__)
    for cls in (
        errors.MediaDecodeError,
        errors.UnsupportedMediaError,
        errors.MediaLimitExceededError,
    ):
        assert issubclass(cls, errors.MediaError)
    assert issubclass(errors.ExternalToolTimeoutError, errors.ExternalToolError)
    assert not issubclass(base, (ValueError, TypeError))


def test_backend_unavailable_messages_name_the_fix():
    missing = errors.BackendUnavailableError.missing_extra(
        "TrustMark watermarking", "image-watermark"
    )
    assert missing.extra == "image-watermark"
    assert 'pip install "provenance-kit[image-watermark]"' in str(missing)
    binary = errors.BackendUnavailableError.missing_binary("fpcalc", feature="Chromaprint")
    assert binary.binary == "fpcalc"
    assert "Chromaprint requires the fpcalc binary" in str(binary)
    assert "PROVENANCE_KIT_FPCALC" in str(binary)


def test_media_limit_error_without_actual_value():
    err = errors.MediaLimitExceededError("max_frames", 10)
    assert str(err) == "media exceeds max_frames=10"
    assert err.actual is None


def test_import_optional_returns_installed_modules():
    assert import_optional("json", extra="x", feature="JSON") is importlib.import_module("json")


def test_import_optional_names_the_extra_when_missing():
    with pytest.raises(errors.BackendUnavailableError) as info:
        import_optional("provenance_kit_no_such_module", extra="c2pa", feature="C2PA signing")
    assert info.value.extra == "c2pa"
    assert 'pip install "provenance-kit[c2pa]"' in str(info.value)
    assert isinstance(info.value.__cause__, ImportError)


def test_import_optional_wraps_broken_native_modules(monkeypatch):
    finder_calls = []

    class BrokenLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module: types.ModuleType) -> None:
            finder_calls.append(module.__name__)
            raise OSError("DLL load failed")

    class BrokenFinder:
        def find_spec(self, name, path=None, target=None):
            if name != "pk_broken_native":
                return None
            from importlib.machinery import ModuleSpec

            return ModuleSpec(name, BrokenLoader())

    monkeypatch.setattr(sys, "meta_path", [BrokenFinder(), *sys.meta_path])
    with pytest.raises(errors.BackendUnavailableError, match="failed to import"):
        import_optional("pk_broken_native", extra="audio-watermark", feature="AudioSeal")
    assert finder_calls == ["pk_broken_native"]
