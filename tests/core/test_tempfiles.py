from __future__ import annotations

import os
import stat
import sys

import pytest

from provenance_kit.tempfiles import private_temp_dir, private_temp_file


def test_file_is_created_written_and_deleted(tmp_path):
    with private_temp_file(suffix=".mp4", directory=tmp_path, data=b"payload") as path:
        assert path.parent == tmp_path.absolute()
        assert path.name.startswith("pk-")
        assert path.suffix == ".mp4"
        assert path.read_bytes() == b"payload"
    assert not path.exists()


def test_file_is_deleted_when_the_block_raises(tmp_path):
    with pytest.raises(RuntimeError), private_temp_file(directory=tmp_path) as path:
        raise RuntimeError("boom")
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_file_already_removed_by_the_caller_is_fine(tmp_path):
    with private_temp_file(directory=tmp_path) as path:
        path.unlink()


def test_memoryview_data(tmp_path):
    with private_temp_file(directory=tmp_path, data=memoryview(b"abc")) as path:
        assert path.read_bytes() == b"abc"


def test_default_directory_is_the_platform_temp_dir():
    with private_temp_file() as path:
        assert path.is_file()
    assert not path.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_posix_permissions_are_private(tmp_path):
    with private_temp_file(directory=tmp_path) as path:
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    with private_temp_dir(directory=tmp_path) as directory:
        assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700


def test_directory_and_its_contents_are_deleted(tmp_path):
    with private_temp_dir(directory=tmp_path) as directory:
        (directory / "frame-0001.png").write_bytes(b"x")
        (directory / "nested").mkdir()
        (directory / "nested" / "a.bin").write_bytes(b"y")
    assert not directory.exists()


def test_directory_is_deleted_when_the_block_raises(tmp_path):
    with (
        pytest.raises(ValueError, match="inner"),
        private_temp_dir(directory=tmp_path) as directory,
    ):
        (directory / "f").write_bytes(b"x")
        raise ValueError("inner")
    assert not directory.exists()


@pytest.mark.parametrize("bad", ["../x", "a/b", "a\\b", "a\x00b"])
def test_names_with_separators_are_refused(tmp_path, bad):
    with pytest.raises(ValueError), private_temp_file(suffix=bad, directory=tmp_path):
        pass
    with pytest.raises(ValueError), private_temp_dir(prefix=bad, directory=tmp_path):
        pass


def test_missing_directory_is_refused(tmp_path):
    with (
        pytest.raises(ValueError, match="does not exist"),
        private_temp_file(directory=tmp_path / "missing"),
    ):
        pass


def test_undeletable_file_is_logged_without_its_path(tmp_path, monkeypatch, caplog):
    import pathlib

    def refuse(self: pathlib.Path, missing_ok: bool = False) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr("provenance_kit.tempfiles._DELETE_RETRIES", (0.0,))
    with private_temp_file(directory=tmp_path) as path:
        monkeypatch.setattr(pathlib.Path, "unlink", refuse)
    monkeypatch.undo()
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings
    assert all(str(path) not in r.getMessage() for r in warnings)
    path.unlink()
