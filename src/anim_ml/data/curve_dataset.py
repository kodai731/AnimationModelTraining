from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path


class CurveCopilotDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, hdf5_paths: list[Path], split: str = "train") -> None:
        self._files: list[h5py.File] = []
        self._lengths: list[int] = []
        self._cumulative: list[int] = []

        cumsum = 0
        for path in hdf5_paths:
            f = h5py.File(path, "r")
            if split not in f:
                f.close()
                continue

            self._files.append(f)
            grp: Any = f[split]
            n: int = grp["context_keyframes"].shape[0]
            self._lengths.append(n)
            cumsum += n
            self._cumulative.append(cumsum)

        self._split = split
        self._total: int = cumsum

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self._total:
            msg = f"Index {index} out of range [0, {self._total})"
            raise IndexError(msg)

        file_idx, local_idx = self._resolve_index(index)
        grp: Any = self._files[file_idx][self._split]

        context: np.ndarray = np.array(grp["context_keyframes"][local_idx], dtype=np.float32)
        target: np.ndarray = np.array(grp["target"][local_idx], dtype=np.float32)
        prop_type: int = int(grp["property_type"][local_idx])
        joint_cat: int = int(grp["joint_category"][local_idx])
        query_time: float = float(grp["query_time"][local_idx])

        context_tensor: torch.Tensor = torch.from_numpy(context)  # type: ignore[no-any-return]
        target_tensor: torch.Tensor = torch.from_numpy(target)  # type: ignore[no-any-return]

        return {
            "context_keyframes": context_tensor,
            "target": target_tensor,
            "property_type": torch.tensor(prop_type, dtype=torch.int64),
            "joint_category": torch.tensor(joint_cat, dtype=torch.int64),
            "query_time": torch.tensor([query_time], dtype=torch.float32),
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
