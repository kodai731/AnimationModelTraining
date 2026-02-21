from __future__ import annotations

import os
from typing import TYPE_CHECKING

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    from anim_ml.utils.preparation_log import PreparationLog


class RigPropagationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        hdf5_paths: list[Path],
        split: str = "train",
        use_shared_memory: bool = True,
        prep_log: PreparationLog | None = None,
    ) -> None:
        self._split = split
        self._paths: list[str] = []
        self._offsets: list[int] = []
        self._handles: list[h5py.File | None] = []

        if prep_log:
            prep_log.log("rig_dataset_init_start", split=split, num_files=len(hdf5_paths))

        offset = 0
        for i, path in enumerate(hdf5_paths):
            with h5py.File(path, "r") as f:
                if split not in f or "joint_features" not in f[split]:
                    if prep_log:
                        prep_log.log("rig_hdf5_split_missing", file_index=i)
                    continue
                n = len(f[split]["joint_features"])

            self._paths.append(str(path))
            self._offsets.append(offset)
            self._handles.append(None)
            offset += n

            if prep_log:
                prep_log.log("rig_hdf5_indexed", file_index=i, samples=n)

        self._total = offset

        if prep_log:
            prep_log.log("rig_dataset_init_done", split=split, total_samples=self._total)

    def _ensure_open(self, file_idx: int) -> h5py.File:
        handle = self._handles[file_idx]
        if handle is None:
            handle = h5py.File(self._paths[file_idx], "r")
            self._handles[file_idx] = handle
        return handle

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
        return self._total

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self._total:
            msg = f"Index {index} out of range [0, {self._total})"
            raise IndexError(msg)

        file_idx, local_idx = self._locate(index)
        f = self._ensure_open(file_idx)
        grp = f[self._split]

        return {
            "joint_features": torch.from_numpy(np.array(grp["joint_features"][local_idx], dtype=np.float32)),
            "topology_features": torch.from_numpy(np.array(grp["topology_features"][local_idx], dtype=np.float32)),
            "bone_name_tokens": torch.from_numpy(np.array(grp["bone_name_tokens"][local_idx], dtype=np.int64)),
            "joint_mask": torch.from_numpy(np.array(grp["joint_mask"][local_idx], dtype=np.float32)),
            "target_deltas": torch.from_numpy(np.array(grp["target_deltas"][local_idx], dtype=np.float32)),
            "confidence_targets": torch.from_numpy(np.array(grp["confidence_targets"][local_idx], dtype=np.float32)),
            "source_indices": torch.from_numpy(np.array(grp["source_indices"][local_idx], dtype=np.int64)),
            "target_indices": torch.from_numpy(np.array(grp["target_indices"][local_idx], dtype=np.int64)),
            "edge_direction": torch.from_numpy(np.array(grp["edge_direction"][local_idx], dtype=np.int64)),
            "edge_mask": torch.from_numpy(np.array(grp["edge_mask"][local_idx], dtype=np.float32)),
        }

    def close(self) -> None:
        for i, handle in enumerate(self._handles):
            if handle is not None:
                handle.close()
                self._handles[i] = None
        self._total = 0
