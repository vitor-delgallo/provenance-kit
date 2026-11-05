"""Lazy import of optional dependencies (internal, shared by every backend).

Heavy dependencies (PyTorch, TrustMark, c2pa-python, ...) are never imported at
module import time. A backend calls :func:`import_optional` when it is first
used, and a missing or broken dependency becomes a
:class:`~provenance_kit.errors.BackendUnavailableError` that names the extra to
install.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from provenance_kit.errors import BackendUnavailableError

__all__ = ["import_optional"]


def import_optional(module: str, *, extra: str, feature: str) -> ModuleType:
    """Imports an optional dependency or raises ``BackendUnavailableError``.

    Args:
        module: The module to import, for example ``"trustmark"``.
        extra: The provenance-kit extra that installs it, for example
            ``"image-watermark"``.
        feature: What needs it, used in the message, for example
            ``"TrustMark watermarking"``.

    Returns:
        The imported module.

    Raises:
        BackendUnavailableError: If the module is not installed, or if
            importing it fails for any other reason (a broken native library,
            an ABI mismatch). The original exception is chained as
            ``__cause__``.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise BackendUnavailableError.missing_extra(
            feature, extra, reason=f"module {module!r} is not installed"
        ) from exc
    except Exception as exc:
        # Native extensions can fail with OSError (a DLL that does not load)
        # or RuntimeError (an ABI mismatch). The backend is unavailable either
        # way; the cause stays chained for debugging.
        raise BackendUnavailableError.missing_extra(
            feature,
            extra,
            reason=f"module {module!r} failed to import ({type(exc).__name__})",
        ) from exc
