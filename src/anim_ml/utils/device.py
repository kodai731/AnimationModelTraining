from __future__ import annotations

import os

import torch


def detect_training_device(override: str | None = None) -> torch.device:
    if override is not None:
        if override == "dml":
            import torch_directml  # type: ignore[import-untyped]
            return torch_directml.device()  # type: ignore[reportReturnType]
        return torch.device(override)

    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml  # type: ignore[import-untyped]
        return torch_directml.device()  # type: ignore[reportReturnType]
    except (ImportError, OSError):
        pass

    return torch.device("cpu")


def supports_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"


def resolve_num_workers(requested: int) -> int:
    cpu_count = os.cpu_count() or 1
    max_workers = max(cpu_count - 1, 1)
    return min(requested, max_workers)
