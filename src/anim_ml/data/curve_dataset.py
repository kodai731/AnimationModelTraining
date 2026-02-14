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


class CurveCopilotDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, hdf5_paths: list[Path], split: str = "train") -> None:
        context_chunks: list[torch.Tensor] = []
        target_chunks: list[torch.Tensor] = []
        prop_type_chunks: list[torch.Tensor] = []
        joint_cat_chunks: list[torch.Tensor] = []
        query_time_chunks: list[torch.Tensor] = []

        for path in hdf5_paths:
            with h5py.File(path, "r") as f:
                if split not in f:
                    continue
                grp = f[split]
                context_chunks.append(torch.from_numpy(np.array(grp["context_keyframes"], dtype=np.float32)))
                target_chunks.append(torch.from_numpy(np.array(grp["target"], dtype=np.float32)))
                prop_type_chunks.append(torch.from_numpy(np.array(grp["property_type"], dtype=np.int64)))
                joint_cat_chunks.append(torch.from_numpy(np.array(grp["joint_category"], dtype=np.int64)))
                query_time_chunks.append(torch.from_numpy(np.array(grp["query_time"], dtype=np.float32)))

        if context_chunks:
            self._context = torch.cat(context_chunks)
            self._target = torch.cat(target_chunks)
            self._prop_type = torch.cat(prop_type_chunks)
            self._joint_cat = torch.cat(joint_cat_chunks)
            self._query_time = torch.cat(query_time_chunks)
            self._move_to_shared_memory()
        else:
            self._context = torch.empty(0)
            self._target = torch.empty(0)
            self._prop_type = torch.empty(0, dtype=torch.int64)
            self._joint_cat = torch.empty(0, dtype=torch.int64)
            self._query_time = torch.empty(0)

    def _move_to_shared_memory(self) -> None:
        if len(self._context) == 0:
            return
        self._context.share_memory_()
        self._target.share_memory_()
        self._prop_type.share_memory_()
        self._joint_cat.share_memory_()
        self._query_time.share_memory_()

    def __len__(self) -> int:
        return len(self._context)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self._context):
            msg = f"Index {index} out of range [0, {len(self._context)})"
            raise IndexError(msg)

        return {
            "context_keyframes": self._context[index],
            "target": self._target[index],
            "property_type": self._prop_type[index],
            "joint_category": self._joint_cat[index],
            "query_time": self._query_time[index].unsqueeze(0),
        }

    def close(self) -> None:
        self._context = torch.empty(0)
        self._target = torch.empty(0)
        self._prop_type = torch.empty(0, dtype=torch.int64)
        self._joint_cat = torch.empty(0, dtype=torch.int64)
        self._query_time = torch.empty(0)
