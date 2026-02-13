from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path


class RigPropagationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, hdf5_paths: list[Path], split: str = "train") -> None:
        self._files: list[h5py.File] = []
        self._lengths: list[int] = []
        self._cumulative: list[int] = []
        self._adjacency: np.ndarray | None = None

        cumsum = 0
        for path in hdf5_paths:
            f = h5py.File(path, "r")
            if split not in f:
                f.close()
                continue

            self._files.append(f)
            grp: Any = f[split]
            n: int = grp["joint_features"].shape[0]
            self._lengths.append(n)
            cumsum += n
            self._cumulative.append(cumsum)

            if self._adjacency is None and "adjacency" in f:
                self._adjacency = np.array(f["adjacency"])

        self._split = split
        self._total: int = cumsum

    @property
    def adjacency(self) -> np.ndarray:
        if self._adjacency is None:
            msg = "No adjacency data found in HDF5 files"
            raise ValueError(msg)
        return self._adjacency

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self._total:
            msg = f"Index {index} out of range [0, {self._total})"
            raise IndexError(msg)

        file_idx, local_idx = self._resolve_index(index)
        grp: Any = self._files[file_idx][self._split]

        joint_features: np.ndarray = np.array(grp["joint_features"][local_idx], dtype=np.float32)
        joint_types: np.ndarray = np.array(grp["joint_types"][local_idx], dtype=np.int64)
        target_deltas: np.ndarray = np.array(grp["target_deltas"][local_idx], dtype=np.float32)
        confidence_targets: np.ndarray = np.array(
            grp["confidence_targets"][local_idx], dtype=np.float32,
        )

        return {
            "joint_features": torch.from_numpy(joint_features),  # type: ignore[no-any-return]
            "joint_types": torch.from_numpy(joint_types),  # type: ignore[no-any-return]
            "target_deltas": torch.from_numpy(target_deltas),  # type: ignore[no-any-return]
            "confidence_targets": torch.from_numpy(confidence_targets),  # type: ignore[no-any-return]
        }

    def close(self) -> None:
        for f in self._files:
            f.close()
        self._files.clear()
        self._lengths.clear()
        self._cumulative.clear()
        self._total = 0

    def _resolve_index(self, index: int) -> tuple[int, int]:
        for file_idx, cum in enumerate(self._cumulative):
            if index < cum:
                prev = self._cumulative[file_idx - 1] if file_idx > 0 else 0
                return file_idx, index - prev
        msg = f"Index {index} out of range"
        raise IndexError(msg)
