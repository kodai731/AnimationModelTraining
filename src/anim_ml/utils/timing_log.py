from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from anim_ml.paths import PROJECT_ROOT

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

LOG_DIR = PROJECT_ROOT / "log"


class TimingLog:
    def __init__(self, model_name: str) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        self._path = LOG_DIR / f"{model_name}_{timestamp}.csv"
        self._header_written = False

    @property
    def path(self) -> Path:
        return self._path

    def write_epoch(self, epoch: int, timings: dict[str, float]) -> None:
        row = {"epoch": epoch, **timings}

        if not self._header_written:
            with open(self._path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            self._header_written = True
        else:
            with open(self._path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)

    @staticmethod
    @contextmanager
    def measure() -> Iterator[dict[str, float]]:
        result: dict[str, float] = {}
        start = time.perf_counter()
        yield result
        result["elapsed"] = time.perf_counter() - start
