from __future__ import annotations

import json
import os
import platform
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from anim_ml.paths import get_shared_data_dir

if TYPE_CHECKING:
    from pathlib import Path

    import torch


def _get_log_dir() -> Path:
    return get_shared_data_dir() / "log" / "Preparation"


def _get_process_memory_mb() -> float:
    try:
        import resource
        rusage: Any = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
        return rusage.ru_maxrss / 1024.0  # type: ignore[reportUnknownVariableType]
    except ImportError:
        pass

    try:
        import psutil  # type: ignore[import-untyped]
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    if platform.system() == "Linux":
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            pass

    return -1.0


def _get_shm_usage_mb() -> dict[str, float]:
    if platform.system() != "Linux":
        return {}
    try:
        stat: Any = os.statvfs("/dev/shm")  # type: ignore[attr-defined]
        total_mb = float(stat.f_blocks * stat.f_frsize) / (1024 * 1024)  # type: ignore[reportUnknownArgumentType]
        free_mb = float(stat.f_bavail * stat.f_frsize) / (1024 * 1024)  # type: ignore[reportUnknownArgumentType]
        return {"shm_total_mb": round(total_mb, 1), "shm_used_mb": round(total_mb - free_mb, 1)}
    except OSError:
        return {}


def _get_system_memory_mb() -> dict[str, float]:
    if platform.system() != "Linux":
        return {}
    try:
        with open("/proc/meminfo") as f:
            info: dict[str, int] = {}
            for line in f:
                parts = line.split()
                if parts[0].rstrip(":") in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                    info[parts[0].rstrip(":")] = int(parts[1])

        return {
            "mem_total_mb": round(info.get("MemTotal", 0) / 1024, 1),
            "mem_available_mb": round(info.get("MemAvailable", 0) / 1024, 1),
            "swap_total_mb": round(info.get("SwapTotal", 0) / 1024, 1),
            "swap_used_mb": round((info.get("SwapTotal", 0) - info.get("SwapFree", 0)) / 1024, 1),
        }
    except OSError:
        return {}


def _tensor_size_mb(t: torch.Tensor) -> float:
    return t.nelement() * t.element_size() / (1024 * 1024)


class PreparationLog:
    def __init__(self, model_name: str) -> None:
        log_dir = _get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        self._path = log_dir / f"{model_name}_{timestamp}.jsonl"
        self._start_time = time.monotonic()

    @property
    def path(self) -> Path:
        return self._path

    def log(self, event: str, **kwargs: Any) -> None:
        record: dict[str, Any] = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "elapsed_sec": round(time.monotonic() - self._start_time, 3),
            "event": event,
            "rss_mb": round(_get_process_memory_mb(), 1),
        }
        record.update(_get_shm_usage_mb())
        record.update(_get_system_memory_mb())
        record.update(kwargs)

        line = json.dumps(record, default=str)
        with open(self._path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

        parts = [f"[prep] {event} | rss={record['rss_mb']}MB"]
        if "shm_used_mb" in record:
            parts.append(f" shm_used={record['shm_used_mb']}MB")
        if "mem_available_mb" in record:
            parts.append(f" mem_avail={record['mem_available_mb']}MB")
        if kwargs:
            parts.append(f" | {kwargs}")
        print("".join(parts), flush=True)

    def log_tensor_info(self, event: str, tensors: dict[str, torch.Tensor]) -> None:
        sizes = {name: round(_tensor_size_mb(t), 2) for name, t in tensors.items()}
        total_mb = sum(sizes.values())
        self.log(event, tensor_sizes_mb=sizes, total_tensor_mb=round(total_mb, 2))
