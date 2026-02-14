from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "paths.toml"

_cached_shared_data_dir: Path | None = None


def clear_cache() -> None:
    global _cached_shared_data_dir
    _cached_shared_data_dir = None


def _load_shared_data_dir() -> Path:
    global _cached_shared_data_dir
    if _cached_shared_data_dir is not None:
        return _cached_shared_data_dir

    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)

    raw = config["paths"]["shared_data_dir"]
    path = Path(raw)

    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()

    _cached_shared_data_dir = path
    return path


def get_shared_data_dir() -> Path:
    return _load_shared_data_dir()


def get_raw_data_dir() -> Path:
    return get_shared_data_dir() / "data" / "raw"


def get_processed_data_dir() -> Path:
    return get_shared_data_dir() / "data" / "processed"


def get_runs_dir() -> Path:
    return get_shared_data_dir() / "runs"


def get_exports_dir() -> Path:
    return get_shared_data_dir() / "exports"


def resolve_data_path(relative_path: str | Path) -> Path:
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return get_shared_data_dir() / p
