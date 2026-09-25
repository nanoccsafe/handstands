"""Tests for :mod:`handstand.paths`: env-var overrides and defaults."""

from __future__ import annotations

import os
import pathlib

import pytest

from handstand import paths

ALL_ENV_VARS = (paths.DATA_ENV_VAR, paths.VIDEOS_ENV_VAR)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with no override set, whatever the outer environment."""
    for name in ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_data_dir_default() -> None:
    assert paths.data_dir() == pathlib.Path("/mnt/sharedOs/handstand-workspace/data")


def test_videos_dir_default() -> None:
    assert paths.videos_dir() == pathlib.Path("/mnt/sharedOs/handstand-workspace/videos")


def test_data_dir_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.DATA_ENV_VAR, "/tmp/handstand-data")
    assert paths.data_dir() == pathlib.Path("/tmp/handstand-data")


def test_videos_dir_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.VIDEOS_ENV_VAR, "/tmp/handstand-videos")
    assert paths.videos_dir() == pathlib.Path("/tmp/handstand-videos")


def test_overrides_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.DATA_ENV_VAR, "/tmp/only-data")
    assert paths.data_dir() == pathlib.Path("/tmp/only-data")
    assert paths.videos_dir() == paths.VIDEOS_DIR

    monkeypatch.delenv(paths.DATA_ENV_VAR)
    monkeypatch.setenv(paths.VIDEOS_ENV_VAR, "/tmp/only-videos")
    assert paths.data_dir() == paths.DATA_DIR
    assert paths.videos_dir() == pathlib.Path("/tmp/only-videos")


def test_empty_override_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.DATA_ENV_VAR, "")
    monkeypatch.setenv(paths.VIDEOS_ENV_VAR, "")
    assert paths.data_dir() == paths.DATA_DIR
    assert paths.videos_dir() == paths.VIDEOS_DIR


def test_override_is_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.DATA_ENV_VAR, "~/handstand-data")
    monkeypatch.setenv(paths.VIDEOS_ENV_VAR, "~/handstand-videos")
    assert paths.data_dir() == pathlib.Path.home() / "handstand-data"
    assert paths.videos_dir() == pathlib.Path.home() / "handstand-videos"


def test_defaults_match_env_var_names() -> None:
    assert paths.DATA_ENV_VAR == "HANDSTAND_DATA"
    assert paths.VIDEOS_ENV_VAR == "HANDSTAND_VIDEOS"


def test_dir_functions_return_paths_not_strings() -> None:
    assert isinstance(paths.data_dir(), pathlib.Path)
    assert isinstance(paths.videos_dir(), pathlib.Path)


def test_dir_functions_do_not_create_directories(monkeypatch: pytest.MonkeyPatch) -> None:
    target = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "handstand-paths-absent"
    monkeypatch.setenv(paths.DATA_ENV_VAR, str(target))
    monkeypatch.setenv(paths.VIDEOS_ENV_VAR, str(target))
    assert not target.exists()
    paths.data_dir()
    paths.videos_dir()
    assert not target.exists()
