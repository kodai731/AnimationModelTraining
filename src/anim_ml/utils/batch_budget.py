from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from anim_ml.utils.memory_budget import get_available_memory_mb

_CPU_TIERS = (32, 64, 128, 256, 512, 1024, 2048)
_GPU_TIERS = (128, 256, 512, 1024, 2048, 4096)

_BYTES_PER_PARAM = 4
_MODEL_MEMORY_MULTIPLIER = 4
_PER_SAMPLE_ACTIVATION_KB = 512
_RAM_RESERVE_MB = 512


@dataclass(frozen=True)
class BatchBudget:
    batch_size: int
    reason: str


def _snap_to_tier(value: int, tiers: tuple[int, ...], cap: int) -> int:
    selected = tiers[0]
    for tier in tiers:
        if tier <= min(value, cap):
            selected = tier
        else:
            break
    return selected


def _estimate_model_fixed_mb(param_count: int) -> float:
    return param_count * _BYTES_PER_PARAM * _MODEL_MEMORY_MULTIPLIER / (1024 * 1024)


def _estimate_max_batch_for_ram(param_count: int, available_mb: int) -> int:
    model_mb = _estimate_model_fixed_mb(param_count)
    remaining_mb = available_mb - _RAM_RESERVE_MB - model_mb
    if remaining_mb <= 0:
        return 32
    return int(remaining_mb * 1024 / _PER_SAMPLE_ACTIVATION_KB)


def _resolve_cpu(config_batch_size: int, param_count: int) -> BatchBudget:
    cores = os.cpu_count() or 1
    core_based = cores * 64

    available_mb = get_available_memory_mb()

    ram_max = config_batch_size
    if param_count > 0:
        ram_max = _estimate_max_batch_for_ram(param_count, available_mb)

    candidate = min(core_based, ram_max)
    batch_size = _snap_to_tier(candidate, _CPU_TIERS, config_batch_size)

    return BatchBudget(
        batch_size=batch_size,
        reason=f"cpu: {cores} cores, {available_mb}MB available",
    )


def _resolve_cuda(config_batch_size: int, param_count: int) -> BatchBudget:
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    free_mb = int(free_bytes / (1024 * 1024))

    if param_count > 0 and free_mb > 0:
        usable_mb = free_mb * 0.8
        model_mb = _estimate_model_fixed_mb(param_count)
        remaining_mb = usable_mb - model_mb

        if remaining_mb > 0:
            vram_max = int(remaining_mb * 1024 / _PER_SAMPLE_ACTIVATION_KB)
            batch_size = _snap_to_tier(vram_max, _GPU_TIERS, config_batch_size)
            return BatchBudget(
                batch_size=batch_size,
                reason=f"cuda: {free_mb}MB free VRAM",
            )

    return BatchBudget(
        batch_size=config_batch_size,
        reason="cuda: using config default",
    )


def resolve_batch_size(
    device: torch.device,
    config_batch_size: int,
    param_count: int,
) -> BatchBudget:
    if device.type == "cuda":
        return _resolve_cuda(config_batch_size, param_count)

    if device.type == "cpu":
        return _resolve_cpu(config_batch_size, param_count)

    return BatchBudget(
        batch_size=config_batch_size,
        reason=f"{device.type}: using config default",
    )
