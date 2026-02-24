from __future__ import annotations

import ctypes
import platform
import tomllib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "paths.toml"

_MEMORY_TIERS_MB = (500, 1024, 2048, 4096, 8192, 16384, 24576, 32768)
_OVERHEAD_MB = 1500
_MB_TO_BYTES = 1024 * 1024


def get_configured_budget_mb() -> int:
    with open(_CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)
    return int(config.get("paths", {}).get("cache_budget_mb", 2048))


def get_available_memory_mb() -> int:
    try:
        if platform.system() == "Windows":
            return _get_available_memory_windows()
        return _get_available_memory_linux()
    except Exception:
        return 4096


def _get_available_memory_windows() -> int:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return int(stat.ullAvailPhys // (1024 * 1024))


def _get_available_memory_linux() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 4096


def select_memory_tier_mb(available_mb: int, max_budget_mb: int) -> int:
    data_budget = available_mb - _OVERHEAD_MB
    effective = min(data_budget, max_budget_mb)

    selected = 0
    for tier in _MEMORY_TIERS_MB:
        if tier <= effective:
            selected = tier
        else:
            break

    return selected


def resolve_cache_budget_bytes() -> int:
    configured = get_configured_budget_mb()
    available = get_available_memory_mb()
    capped = min(configured, int(available * 0.75))
    return max(capped, 0) * _MB_TO_BYTES


class MemoryBudget:
    def __init__(self, total_bytes: int) -> None:
        self._total_bytes = total_bytes
        self._allocations: dict[str, int] = {}

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def available_bytes(self) -> int:
        return max(self._total_bytes - sum(self._allocations.values()), 0)

    def request(self, name: str, desired_bytes: int) -> int:
        granted = min(desired_bytes, self.available_bytes)
        if granted > 0:
            self._allocations[name] = granted
        return granted

    def release(self, name: str) -> None:
        self._allocations.pop(name, None)

    def refresh(self) -> tuple[int, int] | None:
        available = get_available_memory_mb()
        configured = get_configured_budget_mb()
        new_tier_mb = select_memory_tier_mb(available, configured)

        old_bytes = self._total_bytes
        new_bytes = new_tier_mb * _MB_TO_BYTES

        self._total_bytes = new_bytes
        self._allocations.clear()

        if old_bytes != new_bytes:
            return (old_bytes, new_bytes)
        return None


def create_memory_budget() -> MemoryBudget:
    available = get_available_memory_mb()
    configured = get_configured_budget_mb()
    tier_mb = select_memory_tier_mb(available, configured)
    return MemoryBudget(total_bytes=tier_mb * _MB_TO_BYTES)
