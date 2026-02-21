from __future__ import annotations

import ctypes
import platform
import tomllib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "paths.toml"


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


def resolve_cache_budget_bytes() -> int:
    configured = get_configured_budget_mb()
    available = get_available_memory_mb()
    capped = min(configured, int(available * 0.75))
    return max(capped, 0) * 1024 * 1024
