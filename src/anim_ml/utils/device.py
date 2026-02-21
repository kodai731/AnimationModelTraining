from __future__ import annotations

import torch


def detect_training_device(override: str | None = None) -> torch.device:
    if override is not None:
        if override == "dml":
            import torch_directml
            return torch_directml.device()
        return torch.device(override)

    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml
        return torch_directml.device()
    except (ImportError, OSError):
        pass

    return torch.device("cpu")


def supports_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"
