from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from anim_ml.utils.timing_log import TimingLog


@pytest.mark.unit
class TestTimingLog:
    def test_creates_csv_file(self, tmp_path: Path) -> None:
        with patch("anim_ml.utils.timing_log.LOG_DIR", tmp_path):
            log = TimingLog("test_model")
            log.write_epoch(1, {"train_sec": 1.5, "val_sec": 0.5})

        assert log.path.exists()
        assert log.path.suffix == ".csv"
        assert "test_model" in log.path.name

    def test_csv_header_and_rows(self, tmp_path: Path) -> None:
        with patch("anim_ml.utils.timing_log.LOG_DIR", tmp_path):
            log = TimingLog("test_model")
            log.write_epoch(1, {"train_sec": 1.5, "val_sec": 0.5})
            log.write_epoch(2, {"train_sec": 1.3, "val_sec": 0.4})

        with open(log.path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["epoch"] == "1"
        assert rows[0]["train_sec"] == "1.5"
        assert rows[0]["val_sec"] == "0.5"
        assert rows[1]["epoch"] == "2"

    def test_file_named_with_timestamp(self, tmp_path: Path) -> None:
        with patch("anim_ml.utils.timing_log.LOG_DIR", tmp_path):
            log = TimingLog("curve_copilot")

        assert log.path.name.startswith("curve_copilot_")
        assert log.path.name.endswith(".csv")

    def test_multiple_columns(self, tmp_path: Path) -> None:
        with patch("anim_ml.utils.timing_log.LOG_DIR", tmp_path):
            log = TimingLog("test")
            log.write_epoch(1, {
                "train_sec": 10.0,
                "val_sec": 2.0,
                "checkpoint_sec": 0.5,
                "total_sec": 12.5,
                "num_steps": 100,
            })

        with open(log.path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["checkpoint_sec"] == "0.5"
        assert rows[0]["num_steps"] == "100"


@pytest.mark.unit
class TestTimingLogMeasure:
    def test_measure_records_elapsed(self) -> None:
        with TimingLog.measure() as result:
            total = sum(range(1000))
            _ = total

        assert "elapsed" in result
        assert result["elapsed"] >= 0.0

    def test_measure_result_is_writable_before_exit(self) -> None:
        with TimingLog.measure() as result:
            pass

        assert isinstance(result["elapsed"], float)
