from __future__ import annotations

from unittest.mock import patch

import torch

from anim_ml.utils.batch_budget import (
    BatchBudget,
    _snap_to_tier,
    resolve_batch_size,
)

_CPU_TIERS = (32, 64, 128, 256, 512, 1024, 2048)
_GPU_TIERS = (128, 256, 512, 1024, 2048, 4096)
PARAM_COUNT_4M = 4_000_000


class TestSnapToTier:
    def test_exact_tier_match(self) -> None:
        assert _snap_to_tier(256, _CPU_TIERS, 2048) == 256

    def test_between_tiers_rounds_down(self) -> None:
        assert _snap_to_tier(300, _CPU_TIERS, 2048) == 256

    def test_below_smallest_tier(self) -> None:
        assert _snap_to_tier(16, _CPU_TIERS, 2048) == 32

    def test_capped_by_config(self) -> None:
        assert _snap_to_tier(4096, _CPU_TIERS, 512) == 512

    def test_above_all_tiers(self) -> None:
        assert _snap_to_tier(10000, _CPU_TIERS, 10000) == 2048

    def test_gpu_tiers(self) -> None:
        assert _snap_to_tier(3000, _GPU_TIERS, 4096) == 2048


class TestResolveBatchSizeCPU:
    @patch("anim_ml.utils.batch_budget.get_available_memory_mb", return_value=8000)
    @patch("anim_ml.utils.batch_budget.os.cpu_count", return_value=2)
    def test_2_core_cpu(self, _cpu: object, _mem: object) -> None:
        result = resolve_batch_size(torch.device("cpu"), 2048, PARAM_COUNT_4M)
        assert result.batch_size == 128

    @patch("anim_ml.utils.batch_budget.get_available_memory_mb", return_value=8000)
    @patch("anim_ml.utils.batch_budget.os.cpu_count", return_value=8)
    def test_8_core_cpu(self, _cpu: object, _mem: object) -> None:
        result = resolve_batch_size(torch.device("cpu"), 2048, PARAM_COUNT_4M)
        assert result.batch_size == 512

    @patch("anim_ml.utils.batch_budget.get_available_memory_mb", return_value=8000)
    @patch("anim_ml.utils.batch_budget.os.cpu_count", return_value=32)
    def test_config_caps_cpu(self, _cpu: object, _mem: object) -> None:
        result = resolve_batch_size(torch.device("cpu"), 1024, PARAM_COUNT_4M)
        assert result.batch_size <= 1024

    @patch("anim_ml.utils.batch_budget.get_available_memory_mb", return_value=1024)
    @patch("anim_ml.utils.batch_budget.os.cpu_count", return_value=16)
    def test_low_ram_caps_batch(self, _cpu: object, _mem: object) -> None:
        result = resolve_batch_size(torch.device("cpu"), 2048, PARAM_COUNT_4M)
        assert result.batch_size < 1024

    @patch("anim_ml.utils.batch_budget.get_available_memory_mb", return_value=8000)
    @patch("anim_ml.utils.batch_budget.os.cpu_count", return_value=None)
    def test_cpu_count_none_fallback(self, _cpu: object, _mem: object) -> None:
        result = resolve_batch_size(torch.device("cpu"), 2048, PARAM_COUNT_4M)
        assert result.batch_size == 64


class TestResolveBatchSizeCUDA:
    @patch("torch.cuda.mem_get_info", return_value=(8 * 1024**3, 12 * 1024**3))
    def test_cuda_with_vram(self, _mem: object) -> None:
        result = resolve_batch_size(torch.device("cuda"), 2048, PARAM_COUNT_4M)
        assert result.batch_size in _GPU_TIERS
        assert result.batch_size <= 2048

    @patch("torch.cuda.mem_get_info", return_value=(24 * 1024**3, 24 * 1024**3))
    def test_cuda_large_vram(self, _mem: object) -> None:
        result = resolve_batch_size(torch.device("cuda"), 4096, PARAM_COUNT_4M)
        assert result.batch_size <= 4096


class TestResolveBatchSizeDirectML:
    def test_directml_uses_config(self) -> None:
        result = resolve_batch_size(torch.device("privateuseone"), 2048, PARAM_COUNT_4M)
        assert result.batch_size == 2048
        assert "config default" in result.reason


class TestBatchBudgetDataclass:
    def test_frozen(self) -> None:
        budget = BatchBudget(batch_size=256, reason="test")
        assert budget.batch_size == 256
        assert budget.reason == "test"
