from __future__ import annotations

from unittest.mock import patch

from anim_ml.utils.memory_budget import MemoryBudget, select_memory_tier_mb

MB = 1024 * 1024


class TestSelectMemoryTierMb:
    def test_large_available_capped_by_config(self) -> None:
        assert select_memory_tier_mb(available_mb=10000, max_budget_mb=2048) == 2048

    def test_small_available_selects_lower_tier(self) -> None:
        assert select_memory_tier_mb(available_mb=4000, max_budget_mb=8192) == 2048

    def test_exact_tier_boundary(self) -> None:
        assert select_memory_tier_mb(available_mb=1500 + 1024, max_budget_mb=8192) == 1024

    def test_just_above_tier(self) -> None:
        assert select_memory_tier_mb(available_mb=1500 + 1025, max_budget_mb=8192) == 1024

    def test_just_below_tier(self) -> None:
        assert select_memory_tier_mb(available_mb=1500 + 1023, max_budget_mb=8192) == 500

    def test_very_low_memory_returns_zero(self) -> None:
        assert select_memory_tier_mb(available_mb=1500, max_budget_mb=8192) == 0

    def test_below_overhead_returns_zero(self) -> None:
        assert select_memory_tier_mb(available_mb=500, max_budget_mb=8192) == 0

    def test_highest_tier(self) -> None:
        assert select_memory_tier_mb(available_mb=50000, max_budget_mb=32768) == 32768

    def test_config_limits_to_smallest_tier(self) -> None:
        assert select_memory_tier_mb(available_mb=10000, max_budget_mb=500) == 500


class TestMemoryBudgetRefresh:
    @patch("anim_ml.utils.memory_budget.get_available_memory_mb", return_value=6000)
    @patch("anim_ml.utils.memory_budget.get_configured_budget_mb", return_value=8192)
    def test_refresh_updates_total_bytes(self, _cfg: object, _avail: object) -> None:
        budget = MemoryBudget(total_bytes=1024 * MB)
        budget.refresh()
        assert budget.total_bytes == 4096 * MB

    @patch("anim_ml.utils.memory_budget.get_available_memory_mb", return_value=6000)
    @patch("anim_ml.utils.memory_budget.get_configured_budget_mb", return_value=8192)
    def test_refresh_clears_allocations(self, _cfg: object, _avail: object) -> None:
        budget = MemoryBudget(total_bytes=4096 * MB)
        budget.request("train", 2048 * MB)
        budget.request("val", 1024 * MB)
        budget.refresh()
        assert budget.available_bytes == budget.total_bytes

    @patch("anim_ml.utils.memory_budget.get_available_memory_mb", return_value=6000)
    @patch("anim_ml.utils.memory_budget.get_configured_budget_mb", return_value=8192)
    def test_refresh_returns_change_when_tier_changes(
        self, _cfg: object, _avail: object,
    ) -> None:
        budget = MemoryBudget(total_bytes=1024 * MB)
        result = budget.refresh()
        assert result is not None
        assert result == (1024 * MB, 4096 * MB)

    @patch("anim_ml.utils.memory_budget.get_available_memory_mb", return_value=6000)
    @patch("anim_ml.utils.memory_budget.get_configured_budget_mb", return_value=8192)
    def test_refresh_returns_none_when_tier_unchanged(
        self, _cfg: object, _avail: object,
    ) -> None:
        budget = MemoryBudget(total_bytes=4096 * MB)
        result = budget.refresh()
        assert result is None
