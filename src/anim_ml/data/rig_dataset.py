from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py  # type: ignore[import-untyped]
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    from anim_ml.utils.preparation_log import PreparationLog

_FIELD_DTYPES: dict[str, torch.dtype] = {
    "joint_features": torch.float32,
    "topology_features": torch.float32,
    "bone_name_tokens": torch.int64,
    "joint_mask": torch.float32,
    "target_deltas": torch.float32,
    "confidence_targets": torch.float32,
    "source_indices": torch.int64,
    "target_indices": torch.int64,
    "edge_direction": torch.int64,
    "edge_mask": torch.float32,
}

_DTYPE_BYTE_SIZES: dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.int64: 8,
}


class RigPropagationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        hdf5_paths: list[Path],
        split: str = "train",
        cache_budget_bytes: int = 0,
        prep_log: PreparationLog | None = None,
    ) -> None:
        self._split = split
        self._paths: list[str] = []
        self._offsets: list[int] = []
        self._file_counts: list[int] = []

        if prep_log:
            prep_log.log("rig_dataset_init_start", split=split, num_files=len(hdf5_paths))

        per_sample_bytes = 0
        offset = 0
        for i, path in enumerate(hdf5_paths):
            with h5py.File(path, "r") as f:
                if split not in f or "joint_features" not in f[split]:
                    if prep_log:
                        prep_log.log("rig_hdf5_split_missing", file_index=i)
                    continue

                grp = f[split]
                n = len(grp["joint_features"])

                if per_sample_bytes == 0:
                    per_sample_bytes = _compute_per_sample_bytes(grp)

            self._paths.append(str(path))
            self._offsets.append(offset)
            self._file_counts.append(n)
            offset += n

            if prep_log:
                prep_log.log("rig_hdf5_indexed", file_index=i, samples=n)

        self._total_count = offset
        self._per_sample_bytes = per_sample_bytes

        chunk_size = self._total_count
        if cache_budget_bytes > 0 and per_sample_bytes > 0 and self._total_count > 0:
            budget_samples = cache_budget_bytes // per_sample_bytes
            chunk_size = max(min(budget_samples, self._total_count), 1)

        self._chunk_size = chunk_size
        if self._chunk_size > 0:
            self._num_chunks = math.ceil(self._total_count / self._chunk_size)
        else:
            self._num_chunks = 1
        self._chunk_index = 0
        self._cache: dict[str, torch.Tensor] = {}
        self._loaded_count = 0

        self._load_current_chunk()

        if prep_log:
            prep_log.log(
                "rig_dataset_init_done",
                split=split,
                total_samples=self._total_count,
                loaded_samples=self._loaded_count,
                fully_loaded=self.is_fully_loaded,
            )

    @property
    def is_fully_loaded(self) -> bool:
        return self._loaded_count >= self._total_count

    @property
    def total_count(self) -> int:
        return self._total_count

    def _load_current_chunk(self) -> None:
        if self._total_count == 0 or self._chunk_size == 0:
            self._loaded_count = 0
            return

        chunk_start = self._chunk_index * self._chunk_size
        chunk_count = min(self._chunk_size, self._total_count - chunk_start)

        buffers: dict[str, list[torch.Tensor]] = {key: [] for key in _FIELD_DTYPES}
        remaining = chunk_count
        pos = chunk_start

        while remaining > 0:
            file_idx, local_idx = self._locate(pos)
            available = self._file_counts[file_idx] - local_idx
            to_read = min(available, remaining)
            end = local_idx + to_read

            with h5py.File(self._paths[file_idx], "r") as f:
                grp = f[self._split]
                for key, dtype in _FIELD_DTYPES.items():
                    buffers[key].append(torch.as_tensor(grp[key][local_idx:end], dtype=dtype))

            remaining -= to_read
            pos += to_read

        self._cache = {key: torch.cat(parts) for key, parts in buffers.items()}
        self._loaded_count = chunk_count

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

        return {key: self._cache[key][index] for key in _FIELD_DTYPES}

    def reload_chunk(self) -> None:
        if self.is_fully_loaded:
            return
        self._chunk_index = (self._chunk_index + 1) % self._num_chunks
        self._load_current_chunk()

    def close(self) -> None:
        self._cache.clear()
        self._loaded_count = 0
        self._total_count = 0


def _compute_per_sample_bytes(grp: h5py.Group) -> int:
    total = 0
    for key, dtype in _FIELD_DTYPES.items():
        ds = grp[key]
        sample_elements = math.prod(ds.shape[1:]) if len(ds.shape) > 1 else 1
        total += sample_elements * _DTYPE_BYTE_SIZES[dtype]
    return total
