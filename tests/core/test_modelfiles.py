"""Pinned model files: directory convention, verification, opt-in downloads."""

from __future__ import annotations

import hashlib
import io
import sys
import sysconfig
import urllib.error
import urllib.request

import pytest

from provenance_kit import modelfiles
from provenance_kit.errors import ModelLoadError
from provenance_kit.modelfiles import (
    MODEL_DIR_ENV,
    ModelFile,
    default_model_dir,
    download_model_file,
    ensure_model_files,
    find_model_file,
    model_directory,
    read_model_file,
    verify_model_file,
)

DATA = b"model weights" * 100
SPEC = ModelFile(
    name="model.bin",
    url="https://example.invalid/model.bin",
    sha256=hashlib.sha256(DATA).hexdigest(),
    size=len(DATA),
)
OTHER_DATA = b"other weights"
OTHER = ModelFile(
    name="other.bin",
    url="https://example.invalid/other.bin",
    sha256=hashlib.sha256(OTHER_DATA).hexdigest(),
    size=len(OTHER_DATA),
)


# ------------------------------------------------------------ ModelFile


def test_model_file_validation():
    good = {"name": "a", "url": "https://example.invalid/a", "sha256": "0" * 64, "size": 1}
    ModelFile(**good)
    for change in (
        {"url": "http://example.invalid/a"},
        {"name": "../a"},
        {"name": "dir/a"},
        {"name": "dir\\a"},
        {"name": ""},
        {"name": " a"},
        {"sha256": "0" * 63},
        {"sha256": "A" * 64},
        {"size": 0},
    ):
        with pytest.raises(ValueError):
            ModelFile(**{**good, **change})
    with pytest.raises(TypeError):
        ModelFile(**{**good, "size": 1.5})
    with pytest.raises(TypeError):
        ModelFile(**{**good, "size": True})
    with pytest.raises(TypeError):
        ModelFile("a", "https://example.invalid/a", "0" * 64, 1)  # type: ignore[misc]


# ------------------------------------------------------------ directories


def test_default_model_dir_honors_the_environment(tmp_path):
    env = {MODEL_DIR_ENV: str(tmp_path)}
    assert default_model_dir("videoseal", environ=env) == tmp_path / "videoseal"
    assert default_model_dir("trustmark", environ={MODEL_DIR_ENV: "  "}).parts[-3:] == (
        "provenance-kit",
        "models",
        "trustmark",
    )


def test_default_model_dir_reads_os_environ(monkeypatch, tmp_path):
    monkeypatch.setenv(MODEL_DIR_ENV, str(tmp_path))
    assert default_model_dir("audioseal") == tmp_path / "audioseal"
    monkeypatch.delenv(MODEL_DIR_ENV)
    assert default_model_dir("audioseal").parts[-3:] == ("provenance-kit", "models", "audioseal")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows cache location")
def test_default_model_dir_on_windows_uses_localappdata(tmp_path):
    env = {"LOCALAPPDATA": str(tmp_path)}
    assert default_model_dir("x", environ=env) == tmp_path / "provenance-kit" / "models" / "x"


@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="XDG cache location")
def test_default_model_dir_elsewhere_uses_xdg_cache_home(tmp_path):
    env = {"XDG_CACHE_HOME": str(tmp_path)}
    assert default_model_dir("x", environ=env) == tmp_path / "provenance-kit" / "models" / "x"


@pytest.mark.parametrize("backend", ["", "Video", "a/b", "../x", ".hidden", 7])
def test_default_model_dir_refuses_odd_backend_names(backend):
    with pytest.raises(ValueError):
        default_model_dir(backend)  # type: ignore[arg-type]


def test_model_directory_resolves_and_refuses_site_packages(tmp_path):
    assert model_directory(tmp_path / "x" / ".." / "m", "test") == (tmp_path / "m").resolve()
    inside = sysconfig.get_paths()["purelib"] + "/trustmark/models"
    with pytest.raises(ValueError, match="site-packages"):
        model_directory(inside, "trustmark")
    with pytest.raises(TypeError):
        model_directory(3, "trustmark")  # type: ignore[arg-type]


# ------------------------------------------------------------ verification


def test_verify_accepts_the_pinned_file(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(DATA)
    assert verify_model_file(path, SPEC) == path
    assert verify_model_file(str(path), SPEC) == path


def test_verify_refuses_missing_resized_and_tampered_files(tmp_path):
    path = tmp_path / "model.bin"
    with pytest.raises(ModelLoadError, match=r"model\.bin") as missing:
        verify_model_file(path, SPEC)
    assert str(tmp_path) not in str(missing.value)
    path.write_bytes(DATA + b"!")
    with pytest.raises(ModelLoadError, match="SHA-256"):
        verify_model_file(path, SPEC)
    path.write_bytes(DATA[:-1] + b"!")
    with pytest.raises(ModelLoadError, match="SHA-256") as tampered:
        verify_model_file(path, SPEC)
    assert str(tmp_path) not in str(tampered.value)


def test_verify_with_a_custom_digest_skips_the_size(tmp_path):
    path = tmp_path / "custom.bin"
    path.write_bytes(b"my own weights")
    digest = hashlib.sha256(b"my own weights").hexdigest()
    assert verify_model_file(path, SPEC, sha256=digest.upper()) == path
    with pytest.raises(ModelLoadError):
        verify_model_file(path, SPEC)
    with pytest.raises(ValueError):
        verify_model_file(path, SPEC, sha256="xyz")


def test_read_model_file_returns_the_verified_bytes(tmp_path):
    with pytest.raises(ModelLoadError, match=r"model\.bin") as missing:
        read_model_file(SPEC, tmp_path)
    assert str(tmp_path) not in str(missing.value)
    (tmp_path / "model.bin").write_bytes(b"evil" * 325)
    with pytest.raises(ModelLoadError, match="SHA-256"):
        read_model_file(SPEC, tmp_path)
    (tmp_path / "model.bin").write_bytes(DATA + b"longer")
    with pytest.raises(ModelLoadError):
        read_model_file(SPEC, tmp_path)
    (tmp_path / "model.bin").write_bytes(DATA)
    assert read_model_file(SPEC, tmp_path) == DATA


def find(**kwargs):
    return find_model_file(SPEC, backend="test", **kwargs)


def test_find_verifies_pre_provisioned_files(tmp_path):
    (tmp_path / "model.bin").write_bytes(DATA)
    assert find(model_dir=tmp_path) == (tmp_path / "model.bin").resolve()
    assert find(model_path=tmp_path / "model.bin") == tmp_path / "model.bin"


def test_find_uses_the_default_directory(tmp_path, monkeypatch):
    monkeypatch.setenv(MODEL_DIR_ENV, str(tmp_path))
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "model.bin").write_bytes(DATA)
    assert find() == (tmp_path / "test" / "model.bin").resolve()


def test_find_refuses_tampered_files(tmp_path):
    (tmp_path / "model.bin").write_bytes(DATA + b"!")
    with pytest.raises(ModelLoadError, match="SHA-256"):
        find(model_dir=tmp_path)


def test_find_never_downloads_unless_asked(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network used")

    monkeypatch.setattr(modelfiles.urllib.request, "build_opener", no_network)
    with pytest.raises(ModelLoadError, match="download=True"):
        find(model_dir=tmp_path)
    with pytest.raises(ModelLoadError):
        find(model_path=tmp_path / "missing.bin")


def test_find_validates_arguments(tmp_path):
    with pytest.raises(ValueError):
        find(model_dir=tmp_path, sha256="xyz")
    with pytest.raises(ValueError):
        find(model_path=tmp_path / "model.bin", download=True)
    with pytest.raises(ModelLoadError, match="custom sha256"):
        find(model_dir=tmp_path, sha256="1" * 64, download=True)
    with pytest.raises(TypeError):
        find_model_file("model.bin", backend="test")  # type: ignore[arg-type]


# ------------------------------------------------------------ downloads


class FakeOpener:
    """Serves payloads by URL; an exception payload is raised instead."""

    def __init__(self, payloads: dict[str, bytes | Exception]) -> None:
        self.payloads = payloads
        self.requests: list[str] = []
        self.timeouts: list[float] = []

    def open(self, request, timeout):
        self.requests.append(request.full_url)
        self.timeouts.append(timeout)
        payload = self.payloads[request.full_url]
        if isinstance(payload, Exception):
            raise payload
        return io.BytesIO(payload)


def patch_opener(monkeypatch, payloads) -> FakeOpener:
    opener = FakeOpener(payloads)
    monkeypatch.setattr(modelfiles.urllib.request, "build_opener", lambda *handlers: opener)
    return opener


def test_download_writes_only_verified_files(tmp_path, monkeypatch):
    opener = patch_opener(monkeypatch, {SPEC.url: DATA})
    path = find(model_dir=tmp_path / "new", download=True, timeout_seconds=7)
    assert path.read_bytes() == DATA
    assert opener.requests == [SPEC.url]
    assert opener.timeouts == [7.0]
    assert sorted(p.name for p in path.parent.iterdir()) == ["model.bin"]


@pytest.mark.parametrize(
    "payload",
    [
        DATA + b"x",
        DATA[:-1],
        b"y" * len(DATA),
        urllib.error.URLError("down"),
        OSError(),
    ],
)
def test_failed_downloads_leave_nothing_behind(tmp_path, monkeypatch, payload):
    patch_opener(monkeypatch, {SPEC.url: payload})
    with pytest.raises(ModelLoadError) as info:
        download_model_file(SPEC, tmp_path)
    assert str(tmp_path) not in str(info.value)
    assert list(tmp_path.iterdir()) == []


def test_download_validates_the_timeout(tmp_path):
    with pytest.raises(ValueError):
        download_model_file(SPEC, tmp_path, timeout_seconds=0)
    with pytest.raises(TypeError):
        download_model_file(SPEC, tmp_path, timeout_seconds="5")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ensure_model_files([SPEC], tmp_path, timeout_seconds=float("inf"))


def test_ensure_keeps_valid_files_and_replaces_the_others(tmp_path, monkeypatch):
    opener = patch_opener(monkeypatch, {SPEC.url: DATA, OTHER.url: OTHER_DATA})
    paths = ensure_model_files([SPEC, OTHER], tmp_path / "models")
    assert [p.name for p in paths] == ["model.bin", "other.bin"]
    assert opener.requests == [SPEC.url, OTHER.url]
    ensure_model_files([SPEC, OTHER], tmp_path / "models")
    assert opener.requests == [SPEC.url, OTHER.url]  # nothing fetched the second time
    (tmp_path / "models" / "other.bin").write_bytes(b"corrupted")
    ensure_model_files([SPEC, OTHER], tmp_path / "models")
    assert opener.requests == [SPEC.url, OTHER.url, OTHER.url]
    assert (tmp_path / "models" / "other.bin").read_bytes() == OTHER_DATA


def test_ensure_keeps_an_invalid_file_when_the_download_fails(tmp_path, monkeypatch):
    patch_opener(monkeypatch, {SPEC.url: b"tampered"})
    (tmp_path / "model.bin").write_bytes(b"old")
    with pytest.raises(ModelLoadError, match="failed its SHA-256 check"):
        ensure_model_files([SPEC], tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.bin"]


def test_redirects_must_stay_on_https():
    handler = modelfiles._HttpsOnlyRedirects()
    request = urllib.request.Request("https://example.invalid/a")
    with pytest.raises(urllib.error.URLError):
        handler.redirect_request(
            request, io.BytesIO(), 302, "Found", {}, "http://example.invalid/b"
        )
