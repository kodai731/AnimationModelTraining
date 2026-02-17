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

UNK_TOKEN = 1


class CurveCopilotDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        hdf5_paths: list[Path],
        split: str = "train",
        unk_rate: float = 0.25,
    ) -> None:
        self._split = split
        self._unk_rate = unk_rate

        context_chunks: list[torch.Tensor] = []
        target_chunks: list[torch.Tensor] = []
        prop_type_chunks: list[torch.Tensor] = []
        topo_chunks: list[torch.Tensor] = []
        name_token_chunks: list[torch.Tensor] = []
        query_time_chunks: list[torch.Tensor] = []

        for path in hdf5_paths:
            with h5py.File(path, "r") as f:
                if split not in f:
                    continue
                grp = f[split]
                context_chunks.append(
                    torch.from_numpy(np.array(grp["context_keyframes"], dtype=np.float32)),
                )
                target_chunks.append(
                    torch.from_numpy(np.array(grp["target"], dtype=np.float32)),
                )
                prop_type_chunks.append(
                    torch.from_numpy(np.array(grp["property_type"], dtype=np.int64)),
                )
                topo_chunks.append(
                    torch.from_numpy(np.array(grp["topology_features"], dtype=np.float32)),
                )
                name_token_chunks.append(
                    torch.from_numpy(np.array(grp["bone_name_tokens"], dtype=np.int64)),
                )
                query_time_chunks.append(
                    torch.from_numpy(np.array(grp["query_time"], dtype=np.float32)),
                )

        if context_chunks:
            self._context = torch.cat(context_chunks)
            self._target = torch.cat(target_chunks)
            self._prop_type = torch.cat(prop_type_chunks)
            self._topo = torch.cat(topo_chunks)
            self._name_tokens = torch.cat(name_token_chunks)
            self._query_time = torch.cat(query_time_chunks)
            self._move_to_shared_memory()
        else:
            self._context = torch.empty(0)
            self._target = torch.empty(0)
            self._prop_type = torch.empty(0, dtype=torch.int64)
            self._topo = torch.empty(0)
            self._name_tokens = torch.empty(0, dtype=torch.int64)
            self._query_time = torch.empty(0)

    def _move_to_shared_memory(self) -> None:
        if len(self._context) == 0:
            return
        self._context.share_memory_()
        self._target.share_memory_()
        self._prop_type.share_memory_()
        self._topo.share_memory_()
        self._name_tokens.share_memory_()
        self._query_time.share_memory_()

    def __len__(self) -> int:
        return len(self._context)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self._context):
            msg = f"Index {index} out of range [0, {len(self._context)})"
            raise IndexError(msg)

        tokens = self._name_tokens[index].clone()
        if self._split == "train" and self._unk_rate > 0:
            tokens = _augment_bone_name_tokens(tokens, self._unk_rate)

        return {
            "context_keyframes": self._context[index],
            "target": self._target[index],
            "property_type": self._prop_type[index],
            "topology_features": self._topo[index],
            "bone_name_tokens": tokens,
            "query_time": self._query_time[index].unsqueeze(0),
        }

    def close(self) -> None:
        self._context = torch.empty(0)
        self._target = torch.empty(0)
        self._prop_type = torch.empty(0, dtype=torch.int64)
        self._topo = torch.empty(0)
        self._name_tokens = torch.empty(0, dtype=torch.int64)
        self._query_time = torch.empty(0)


def _augment_bone_name_tokens(tokens: torch.Tensor, unk_rate: float) -> torch.Tensor:
    mask = torch.rand(tokens.shape) < unk_rate
    non_pad = tokens != 0
    replace_mask = mask & non_pad
    tokens[replace_mask] = UNK_TOKEN
    return tokens
