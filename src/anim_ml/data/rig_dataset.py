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


class RigPropagationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, hdf5_paths: list[Path], split: str = "train") -> None:
        self._adjacency_np: np.ndarray | None = None

        features_chunks: list[torch.Tensor] = []
        types_chunks: list[torch.Tensor] = []
        deltas_chunks: list[torch.Tensor] = []
        confidence_chunks: list[torch.Tensor] = []

        for path in hdf5_paths:
            with h5py.File(path, "r") as f:
                if split not in f:
                    continue

                if self._adjacency_np is None and "adjacency" in f:
                    self._adjacency_np = np.array(f["adjacency"])

                grp = f[split]
                features_chunks.append(torch.from_numpy(np.array(grp["joint_features"], dtype=np.float32)))
                types_chunks.append(torch.from_numpy(np.array(grp["joint_types"], dtype=np.int64)))
                deltas_chunks.append(torch.from_numpy(np.array(grp["target_deltas"], dtype=np.float32)))
                confidence_chunks.append(torch.from_numpy(np.array(grp["confidence_targets"], dtype=np.float32)))

        if features_chunks:
            self._joint_features = torch.cat(features_chunks)
            self._joint_types = torch.cat(types_chunks)
            self._target_deltas = torch.cat(deltas_chunks)
            self._confidence_targets = torch.cat(confidence_chunks)
            self._move_to_shared_memory()
        else:
            self._joint_features = torch.empty(0)
            self._joint_types = torch.empty(0, dtype=torch.int64)
            self._target_deltas = torch.empty(0)
            self._confidence_targets = torch.empty(0)

    def _move_to_shared_memory(self) -> None:
        if len(self._joint_features) == 0:
            return
        self._joint_features.share_memory_()
        self._joint_types.share_memory_()
        self._target_deltas.share_memory_()
        self._confidence_targets.share_memory_()

    @property
    def adjacency(self) -> np.ndarray:
        if self._adjacency_np is None:
            msg = "No adjacency data found in HDF5 files"
            raise ValueError(msg)
        return self._adjacency_np

    def __len__(self) -> int:
        return len(self._joint_features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self._joint_features):
            msg = f"Index {index} out of range [0, {len(self._joint_features)})"
            raise IndexError(msg)

        return {
            "joint_features": self._joint_features[index],
            "joint_types": self._joint_types[index],
            "target_deltas": self._target_deltas[index],
            "confidence_targets": self._confidence_targets[index],
        }

    def close(self) -> None:
        self._joint_features = torch.empty(0)
        self._joint_types = torch.empty(0, dtype=torch.int64)
        self._target_deltas = torch.empty(0)
        self._confidence_targets = torch.empty(0)
