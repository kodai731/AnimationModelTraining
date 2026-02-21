from __future__ import annotations

import csv
import statistics
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from anim_ml.paths import PROJECT_ROOT, get_shared_data_dir

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


class BatchTimingLog:
    _STEP_FIELDS = ["epoch", "step", "data_wait_ms", "compute_ms"]
    _SUMMARY_FIELDS = [
        "epoch", "num_steps",
        "data_wait_mean_ms", "data_wait_p50_ms",
        "data_wait_p95_ms", "data_wait_max_ms",
        "compute_mean_ms", "compute_p50_ms",
        "compute_p95_ms", "compute_max_ms",
        "data_wait_ratio",
    ]

    def __init__(self, model_name: str) -> None:
        log_dir = get_shared_data_dir() / "log" / "Training"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        self._step_path = log_dir / f"{model_name}_batch_{timestamp}.csv"
        self._summary_path = log_dir / f"{model_name}_batch_summary_{timestamp}.csv"
        self._step_header_written = False
        self._summary_header_written = False

        self._epoch_data_waits: list[float] = []
        self._epoch_computes: list[float] = []
        self._current_epoch = 0
        self._data_start = 0.0

    @property
    def step_path(self) -> Path:
        return self._step_path

    @property
    def summary_path(self) -> Path:
        return self._summary_path

    def mark_data_start(self) -> None:
        self._data_start = time.perf_counter()

    def begin_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch
        self._epoch_data_waits.clear()
        self._epoch_computes.clear()
        self.mark_data_start()

    def record_step(
        self, step: int, data_wait_sec: float, compute_sec: float,
    ) -> None:
        data_ms = data_wait_sec * 1000
        compute_ms = compute_sec * 1000
        self._epoch_data_waits.append(data_ms)
        self._epoch_computes.append(compute_ms)

        row = {
            "epoch": self._current_epoch,
            "step": step,
            "data_wait_ms": round(data_ms, 2),
            "compute_ms": round(compute_ms, 2),
        }
        self._write_step_row(row)

    def end_epoch(self) -> None:
        if not self._epoch_data_waits:
            return

        waits = sorted(self._epoch_data_waits)
        comps = sorted(self._epoch_computes)
        total_wait = sum(waits)
        total_comp = sum(comps)

        summary = {
            "epoch": self._current_epoch,
            "num_steps": len(waits),
            "data_wait_mean_ms": round(statistics.mean(waits), 2),
            "data_wait_p50_ms": round(_percentile(waits, 50), 2),
            "data_wait_p95_ms": round(_percentile(waits, 95), 2),
            "data_wait_max_ms": round(waits[-1], 2),
            "compute_mean_ms": round(statistics.mean(comps), 2),
            "compute_p50_ms": round(_percentile(comps, 50), 2),
            "compute_p95_ms": round(_percentile(comps, 95), 2),
            "compute_max_ms": round(comps[-1], 2),
            "data_wait_ratio": round(
                total_wait / max(total_wait + total_comp, 1e-9), 4,
            ),
        }
        self._write_summary_row(summary)

    def _write_step_row(self, row: dict[str, object]) -> None:
        if not self._step_header_written:
            with open(self._step_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._STEP_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            self._step_header_written = True
        else:
            with open(self._step_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._STEP_FIELDS)
                writer.writerow(row)

    def _write_summary_row(self, row: dict[str, object]) -> None:
        if not self._summary_header_written:
            with open(self._summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._SUMMARY_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            self._summary_header_written = True
        else:
            with open(self._summary_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._SUMMARY_FIELDS)
                writer.writerow(row)


def _percentile(sorted_data: list[float], pct: float) -> float:
    if not sorted_data:
        return 0.0
    idx = (len(sorted_data) - 1) * pct / 100
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])
