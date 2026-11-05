from __future__ import annotations

from types import SimpleNamespace

import pytest

from provenance_kit.device import parse_device, resolve_device
from provenance_kit.errors import BackendUnavailableError


def fake_torch(available: bool, count: int = 0) -> SimpleNamespace:
    cuda = SimpleNamespace(is_available=lambda: available, device_count=lambda: count)
    return SimpleNamespace(cuda=cuda)


@pytest.mark.parametrize(
    ("given", "expected"),
    [("cpu", "cpu"), (" CPU ", "cpu"), ("cuda", "cuda"), ("cuda:1", "cuda:1"), ("AUTO", "auto")],
)
def test_parse_device(given, expected):
    assert parse_device(given) == expected


@pytest.mark.parametrize("bad", ["gpu", "cuda:", "cpu:0", "auto:1", "cuda:-1", "cuda:1234", ""])
def test_parse_device_rejects_unknown_forms(bad):
    with pytest.raises(ValueError):
        parse_device(bad)


def test_parse_device_rejects_non_strings():
    with pytest.raises(TypeError):
        parse_device(0)  # type: ignore[arg-type]


def test_cpu_never_touches_torch():
    assert resolve_device("cpu", None) == "cpu"


def test_auto_follows_availability():
    assert resolve_device("auto", fake_torch(True, 1)) == "cuda"
    assert resolve_device("auto", fake_torch(False)) == "cpu"


def test_explicit_cuda_requires_cuda():
    assert resolve_device("cuda:1", fake_torch(True, 2)) == "cuda:1"
    with pytest.raises(BackendUnavailableError, match="not available"):
        resolve_device("cuda", fake_torch(False))
    with pytest.raises(BackendUnavailableError, match="does not exist"):
        resolve_device("cuda:2", fake_torch(True, 2))
