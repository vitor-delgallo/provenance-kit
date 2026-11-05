"""Compute device selection for the PyTorch backends.

Every model-backed backend takes ``device: str = "cpu"`` in its constructor.
Accepted values are ``"cpu"``, ``"cuda"``, ``"cuda:<index>"`` and ``"auto"``
(CUDA when available, else CPU). The default is ``"cpu"`` so that behavior
does not change silently with the hardware; applications opt into GPUs.

:func:`parse_device` validates the string without importing PyTorch;
:func:`resolve_device` checks it against the installed PyTorch.
"""

from __future__ import annotations

import re
from typing import Any

from provenance_kit.errors import BackendUnavailableError

__all__ = ["parse_device", "resolve_device"]

_DEVICE_RE = re.compile(r"^(cpu|auto|cuda)(?::(\d{1,3}))?$")


def parse_device(device: str) -> str:
    """Validates and normalizes a device string.

    Args:
        device: ``"cpu"``, ``"cuda"``, ``"cuda:<index>"`` or ``"auto"``
            (case-insensitive, surrounding spaces ignored).

    Returns:
        The normalized string, for example ``"cuda:0"`` stays ``"cuda:0"`` and
        ``" CPU "`` becomes ``"cpu"``.

    Raises:
        TypeError: If ``device`` is not a ``str``.
        ValueError: If it is not one of the accepted forms.
    """
    if not isinstance(device, str):
        raise TypeError(f"device must be a str, got {type(device).__name__}")
    normalized = device.strip().lower()
    match = _DEVICE_RE.match(normalized)
    if match is None or (match.group(2) is not None and match.group(1) != "cuda"):
        raise ValueError('device must be "cpu", "cuda", "cuda:<index>" or "auto"')
    return normalized


def resolve_device(device: str, torch: Any) -> str:
    """Resolves a device string against the installed PyTorch.

    Args:
        device: A device string accepted by :func:`parse_device`.
        torch: The ``torch`` module, already imported by the backend through
            :func:`provenance_kit._optional.import_optional` (passing it keeps
            this module free of heavy imports and lets tests use a fake).

    Returns:
        A concrete device: ``"cpu"``, ``"cuda"`` or ``"cuda:<index>"``.

    Raises:
        TypeError: If ``device`` is not a ``str``.
        ValueError: If ``device`` is malformed.
        BackendUnavailableError: If CUDA was requested explicitly but is not
            available, or the requested GPU index does not exist.
    """
    normalized = parse_device(device)
    if normalized == "cpu":
        return "cpu"
    available = bool(torch.cuda.is_available())
    if normalized == "auto":
        return "cuda" if available else "cpu"
    if not available:
        raise BackendUnavailableError(
            "CUDA was requested but is not available: install a CUDA build of PyTorch "
            'or use device="cpu"'
        )
    _, _, index = normalized.partition(":")
    if index and int(index) >= int(torch.cuda.device_count()):
        raise BackendUnavailableError(f"CUDA device {index} does not exist")
    return normalized
