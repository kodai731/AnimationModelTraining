from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import anim_ml.paths as paths_module
from anim_ml.paths import (
    PROJECT_ROOT,
    clear_cache,
    get_exports_dir,
    get_processed_data_dir,
    get_raw_data_dir,
    get_runs_dir,
    get_shared_data_dir,
    resolve_data_path,
)


@pytest.fixture(autouse=True)
def _reset_cache() -> None:  # pyright: ignore[reportUnusedFunction]
    clear_cache()


@pytest.mark.unit
class TestPathsTomlExists:
    def test_paths_toml_present(self) -> None:
        assert (PROJECT_ROOT / "paths.toml").exists()

    def test_paths_toml_has_shared_data_dir(self) -> None:
        import tomllib

        with open(PROJECT_ROOT / "paths.toml", "rb") as f:
            config = tomllib.load(f)

        assert "paths" in config
        assert "shared_data_dir" in config["paths"]
        assert isinstance(config["paths"]["shared_data_dir"], str)


@pytest.mark.unit
class TestGetSharedDataDir:
    def test_returns_absolute_path(self) -> None:
        result = get_shared_data_dir()
        assert result.is_absolute()

    def test_relative_path_resolved_from_project_root(self, tmp_path: Path) -> None:
        toml_content = textwrap.dedent("""\
            [paths]
            shared_data_dir = "../SharedData"
        """)
        fake_toml = tmp_path / "paths.toml"
        fake_toml.write_text(toml_content)

        original_config = paths_module.CONFIG_PATH
        original_root = paths_module.PROJECT_ROOT
        try:
            paths_module.CONFIG_PATH = fake_toml
            paths_module.PROJECT_ROOT = tmp_path

            result = get_shared_data_dir()
            expected = (tmp_path / ".." / "SharedData").resolve()
            assert result == expected
        finally:
            paths_module.CONFIG_PATH = original_config
            paths_module.PROJECT_ROOT = original_root

    def test_absolute_path_used_as_is(self, tmp_path: Path) -> None:
        abs_path = tmp_path / "MyData"
        toml_content = f'[paths]\nshared_data_dir = "{abs_path.as_posix()}"'
        fake_toml = tmp_path / "paths.toml"
        fake_toml.write_text(toml_content)

        original = paths_module.CONFIG_PATH
        try:
            paths_module.CONFIG_PATH = fake_toml
            result = get_shared_data_dir()
            assert result == abs_path
        finally:
            paths_module.CONFIG_PATH = original


@pytest.mark.unit
class TestSubdirectoryPaths:
    def test_raw_data_dir(self) -> None:
        result = get_raw_data_dir()
        assert result == get_shared_data_dir() / "data" / "raw"

    def test_processed_data_dir(self) -> None:
        result = get_processed_data_dir()
        assert result == get_shared_data_dir() / "data" / "processed"

    def test_runs_dir(self) -> None:
        result = get_runs_dir()
        assert result == get_shared_data_dir() / "runs"

    def test_exports_dir(self) -> None:
        result = get_exports_dir()
        assert result == get_shared_data_dir() / "exports"

    def test_all_paths_are_absolute(self) -> None:
        assert get_raw_data_dir().is_absolute()
        assert get_processed_data_dir().is_absolute()
        assert get_runs_dir().is_absolute()
        assert get_exports_dir().is_absolute()


@pytest.mark.unit
class TestResolveDataPath:
    def test_relative_path_resolved_under_shared_data(self) -> None:
        result = resolve_data_path("data/processed/all_curves.h5")
        expected = get_shared_data_dir() / "data" / "processed" / "all_curves.h5"
        assert result == expected

    def test_absolute_path_returned_unchanged(self, tmp_path: Path) -> None:
        abs_path = tmp_path / "some_file.h5"
        result = resolve_data_path(abs_path)
        assert result == abs_path

    def test_path_object_input(self) -> None:
        result = resolve_data_path(Path("runs/model/best.pt"))
        expected = get_shared_data_dir() / "runs" / "model" / "best.pt"
        assert result == expected

    def test_result_is_always_absolute(self) -> None:
        result = resolve_data_path("some/relative/path")
        assert result.is_absolute()


@pytest.mark.unit
class TestCaching:
    def test_repeated_calls_return_same_object(self) -> None:
        first = get_shared_data_dir()
        second = get_shared_data_dir()
        assert first is second

    def test_clear_cache_reloads(self, tmp_path: Path) -> None:
        first = get_shared_data_dir()

        clear_cache()

        second = get_shared_data_dir()
        assert first == second
        assert first is not second
