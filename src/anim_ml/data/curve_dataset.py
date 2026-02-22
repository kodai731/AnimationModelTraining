from __future__ import annotations

import math
import os
import random
from typing import TYPE_CHECKING, cast

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py  # type: ignore[import-untyped]
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    from anim_ml.utils.memory_budget import MemoryBudget
    from anim_ml.utils.preparation_log import PreparationLog

UNK_TOKEN = 1
PAE_WINDOW_SIZE = 64

_FIELD_DTYPES: dict[str, torch.dtype] = {
    "context_keyframes": torch.float32,
    "target": torch.float32,
    "property_type": torch.int64,
    "topology_features": torch.float32,
    "bone_name_tokens": torch.int64,
    "query_time": torch.float32,
    "curve_window": torch.float32,
}

_OPTIONAL_FALLBACK_SHAPES: dict[str, tuple[int, ...]] = {
    "curve_window": (PAE_WINDOW_SIZE,),
}

_DTYPE_BYTE_SIZES: dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.int64: 8,
}


class CurveCopilotDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        hdf5_paths: list[Path],
        split: str = "train",
        unk_rate: float = 0.25,
        cache_budget_bytes: int = 0,
        prep_log: PreparationLog | None = None,
        memory_budget: MemoryBudget | None = None,
        budget_name: str = "",
    ) -> None:
        self._split = split
        self._unk_rate = unk_rate
        self._paths: list[str] = []
        self._offsets: list[int] = []
        self._file_counts: list[int] = []
        self._memory_budget = memory_budget
        self._budget_name = budget_name

        if prep_log:
            prep_log.log("curve_dataset_init_start", split=split, num_files=len(hdf5_paths))

        per_sample_bytes = 0
        offset = 0
        for i, path in enumerate(hdf5_paths):
            with h5py.File(path, "r") as f:
                if split not in f:
                    if prep_log:
                        prep_log.log("curve_hdf5_split_missing", file_index=i)
                    continue

                grp = cast("h5py.Group", f[split])
                if "context_keyframes" not in grp:
                    if prep_log:
                        prep_log.log("curve_hdf5_split_missing", file_index=i)
                    continue

                n = len(cast("h5py.Dataset", grp["context_keyframes"]))

                if per_sample_bytes == 0:
                    per_sample_bytes = _compute_per_sample_bytes(grp)

            self._paths.append(str(path))
            self._offsets.append(offset)
            self._file_counts.append(n)
            offset += n

            if prep_log:
                prep_log.log("curve_hdf5_indexed", file_index=i, samples=n)

        self._total_count = offset
        self._per_sample_bytes = per_sample_bytes

        effective_budget = self._resolve_effective_budget(cache_budget_bytes)
        self._apply_chunk_size(effective_budget)

        self._chunk_index = 0
        self._epoch_offset: int = 0
        self._cache: dict[str, torch.Tensor] = {}
        self._loaded_count = 0

        self._load_current_chunk()

        if prep_log:
            prep_log.log(
                "curve_dataset_init_done",
                split=split,
                total_samples=self._total_count,
                loaded_samples=self._loaded_count,
                fully_loaded=self.is_fully_loaded,
            )

    @property
    def is_fully_loaded(self) -> bool:
        return self._loaded_count >= self._total_count

    @property
    def num_chunks(self) -> int:
        return self._num_chunks

    @property
    def total_count(self) -> int:
        return self._total_count

    def _resolve_effective_budget(self, cache_budget_bytes: int) -> int:
        if self._memory_budget is not None and self._budget_name:
            total_needed = self._per_sample_bytes * self._total_count
            return self._memory_budget.request(self._budget_name, total_needed)
        return cache_budget_bytes

    def _apply_chunk_size(self, effective_budget: int) -> None:
        if self._memory_budget is not None and effective_budget <= 0:
            self._chunk_size = 0
            self._num_chunks = 1
            return

        chunk_size = self._total_count
        if effective_budget > 0 and self._per_sample_bytes > 0 and self._total_count > 0:
            budget_samples = effective_budget // self._per_sample_bytes
            chunk_size = max(min(budget_samples, self._total_count), 1)

        self._chunk_size = chunk_size
        if self._chunk_size > 0:
            self._num_chunks = math.ceil(self._total_count / self._chunk_size)
        else:
            self._num_chunks = 1

    def begin_epoch(self, epoch: int) -> None:
        if self._chunk_size < self._total_count:
            self._epoch_offset = random.Random(epoch).randint(0, self._chunk_size - 1)
        else:
            self._epoch_offset = 0

    def _load_current_chunk(self) -> None:
        if self._total_count == 0 or self._chunk_size == 0:
            self._loaded_count = 0
            return

        samples_before = self._chunk_index * self._chunk_size
        chunk_count = min(self._chunk_size, self._total_count - samples_before)
        chunk_start = (self._epoch_offset + samples_before) % self._total_count

        self._cache.clear()

        tail = self._total_count - chunk_start
        if chunk_count <= tail:
            file_slices = self._collect_file_slices(chunk_start, chunk_count)
        else:
            file_slices = self._collect_file_slices(chunk_start, tail)
            file_slices += self._collect_file_slices(0, chunk_count - tail)

        for key, dtype in _FIELD_DTYPES.items():
            parts: list[torch.Tensor] = []
            for file_idx, local_start, local_end in file_slices:
                with h5py.File(self._paths[file_idx], "r") as f:
                    split_grp = cast("h5py.Group", f[self._split])
                    if key not in split_grp and key in _OPTIONAL_FALLBACK_SHAPES:
                        count = local_end - local_start
                        fallback_shape = (count,) + _OPTIONAL_FALLBACK_SHAPES[key]
                        parts.append(torch.zeros(fallback_shape, dtype=dtype))
                    else:
                        ds = cast("h5py.Dataset", split_grp[key])
                        parts.append(torch.as_tensor(ds[local_start:local_end], dtype=dtype))
            self._cache[key] = torch.cat(parts)
            del parts

        if self._cache["query_time"].ndim == 1:
            self._cache["query_time"] = self._cache["query_time"].unsqueeze(-1)

        self._loaded_count = chunk_count

    def _collect_file_slices(
        self, chunk_start: int, chunk_count: int,
    ) -> list[tuple[int, int, int]]:
        slices: list[tuple[int, int, int]] = []
        remaining = chunk_count
        pos = chunk_start

        while remaining > 0:
            file_idx, local_idx = self._locate(pos)
            available = self._file_counts[file_idx] - local_idx
            to_read = min(available, remaining)
            slices.append((file_idx, local_idx, local_idx + to_read))
            remaining -= to_read
            pos += to_read

        return slices

    def evict_cache(self) -> None:
        self._cache.clear()
        self._loaded_count = 0
        if self._memory_budget and self._budget_name:
            self._memory_budget.release(self._budget_name)

    def reload_cache(self) -> None:
        if self._memory_budget and self._budget_name and self._per_sample_bytes > 0:
            total_needed = self._per_sample_bytes * self._total_count
            effective = self._memory_budget.request(self._budget_name, total_needed)
            self._apply_chunk_size(effective)
        self._chunk_index = 0
        self._load_current_chunk()

    def _locate(self, index: int) -> tuple[int, int]:
        lo, hi = 0, len(self._offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._offsets[mid] <= index:
                lo = mid
            else:
                hi = mid - 1
        return lo, index - self._offsets[lo]

    def __len__(self) -> int:
        return self._loaded_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self._loaded_count:
            msg = f"Index {index} out of range [0, {self._loaded_count})"
            raise IndexError(msg)

        tokens = self._cache["bone_name_tokens"][index].clone()
        if self._split == "train" and self._unk_rate > 0:
            tokens = _augment_bone_name_tokens(tokens, self._unk_rate)

        return {
            "context_keyframes": self._cache["context_keyframes"][index],
            "target": self._cache["target"][index],
            "property_type": self._cache["property_type"][index],
            "topology_features": self._cache["topology_features"][index],
            "bone_name_tokens": tokens,
            "query_time": self._cache["query_time"][index],
            "curve_window": self._cache["curve_window"][index],
        }

    def reload_chunk(self, chunk_index: int | None = None) -> None:
        if self.is_fully_loaded:
            return
        if chunk_index is not None:
            self._chunk_index = chunk_index % self._num_chunks
        else:
            self._chunk_index = (self._chunk_index + 1) % self._num_chunks
        self._load_current_chunk()

    def close(self) -> None:
        self._cache.clear()
        self._loaded_count = 0
        self._total_count = 0


def _compute_per_sample_bytes(grp: h5py.Group) -> int:
    total = 0
    for key, dtype in _FIELD_DTYPES.items():
        if key not in grp and key in _OPTIONAL_FALLBACK_SHAPES:
            sample_elements = math.prod(_OPTIONAL_FALLBACK_SHAPES[key])
        else:
            ds = cast("h5py.Dataset", grp[key])
            shape = cast("tuple[int, ...]", ds.shape)  # pyright: ignore[reportUnknownMemberType]
            sample_elements = math.prod(shape[1:]) if len(shape) > 1 else 1
        total += sample_elements * _DTYPE_BYTE_SIZES[dtype]
    return total


def _augment_bone_name_tokens(tokens: torch.Tensor, unk_rate: float) -> torch.Tensor:
    mask = torch.rand(tokens.shape) < unk_rate
    non_pad = tokens != 0
    replace_mask = mask & non_pad
    tokens[replace_mask] = UNK_TOKEN
    return tokens
