from __future__ import annotations

import gc
import os
from typing import TYPE_CHECKING

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path


def _load_tensor(
    grp: h5py.Group,  # type: ignore[type-arg]
    key: str,
    dtype: type[np.floating] | type[np.integer],  # type: ignore[type-arg]
) -> torch.Tensor:
    return torch.from_numpy(np.array(grp[key], dtype=dtype))  # type: ignore[no-any-return, reportUnknownMemberType]


class RigPropagationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        hdf5_paths: list[Path],
        split: str = "train",
        use_shared_memory: bool = True,
    ) -> None:
        features_chunks: list[torch.Tensor] = []
        topo_chunks: list[torch.Tensor] = []
        tokens_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        deltas_chunks: list[torch.Tensor] = []
        confidence_chunks: list[torch.Tensor] = []
        src_chunks: list[torch.Tensor] = []
        tgt_chunks: list[torch.Tensor] = []
        edge_dir_chunks: list[torch.Tensor] = []
        edge_mask_chunks: list[torch.Tensor] = []

        for path in hdf5_paths:
            with h5py.File(path, "r") as f:
                if split not in f:
                    continue

                grp: h5py.Group = f[split]  # type: ignore[assignment]
                features_chunks.append(
                    _load_tensor(grp, "joint_features", np.float32),
                )
                topo_chunks.append(
                    _load_tensor(grp, "topology_features", np.float32),
                )
                tokens_chunks.append(
                    _load_tensor(grp, "bone_name_tokens", np.int64),
                )
                mask_chunks.append(
                    _load_tensor(grp, "joint_mask", np.float32),
                )
                deltas_chunks.append(
                    _load_tensor(grp, "target_deltas", np.float32),
                )
                confidence_chunks.append(
                    _load_tensor(grp, "confidence_targets", np.float32),
                )
                src_chunks.append(
                    _load_tensor(grp, "source_indices", np.int64),
                )
                tgt_chunks.append(
                    _load_tensor(grp, "target_indices", np.int64),
                )
                edge_dir_chunks.append(
                    _load_tensor(grp, "edge_direction", np.int64),
                )
                edge_mask_chunks.append(
                    _load_tensor(grp, "edge_mask", np.float32),
                )

        if features_chunks:
            self._joint_features = torch.cat(features_chunks)
            self._topology_features = torch.cat(topo_chunks)
            self._bone_name_tokens = torch.cat(tokens_chunks)
            self._joint_mask = torch.cat(mask_chunks)
            self._target_deltas = torch.cat(deltas_chunks)
            self._confidence_targets = torch.cat(confidence_chunks)
            self._source_indices = torch.cat(src_chunks)
            self._target_indices = torch.cat(tgt_chunks)
            self._edge_direction = torch.cat(edge_dir_chunks)
            self._edge_mask = torch.cat(edge_mask_chunks)

            del features_chunks, topo_chunks, tokens_chunks, mask_chunks
            del deltas_chunks, confidence_chunks
            del src_chunks, tgt_chunks, edge_dir_chunks, edge_mask_chunks
            gc.collect()

            if use_shared_memory:
                self._move_to_shared_memory()
        else:
            self._joint_features = torch.empty(0)
            self._topology_features = torch.empty(0)
            self._bone_name_tokens = torch.empty(0, dtype=torch.int64)
            self._joint_mask = torch.empty(0)
            self._target_deltas = torch.empty(0)
            self._confidence_targets = torch.empty(0)
            self._source_indices = torch.empty(0, dtype=torch.int64)
            self._target_indices = torch.empty(0, dtype=torch.int64)
            self._edge_direction = torch.empty(0, dtype=torch.int64)
            self._edge_mask = torch.empty(0)

    def _move_to_shared_memory(self) -> None:
        if len(self._joint_features) == 0:
            return
        self._joint_features.share_memory_()
        self._topology_features.share_memory_()
        self._bone_name_tokens.share_memory_()
        self._joint_mask.share_memory_()
        self._target_deltas.share_memory_()
        self._confidence_targets.share_memory_()
        self._source_indices.share_memory_()
        self._target_indices.share_memory_()
        self._edge_direction.share_memory_()
        self._edge_mask.share_memory_()

    def __len__(self) -> int:
        return len(self._joint_features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self._joint_features):
            msg = f"Index {index} out of range [0, {len(self._joint_features)})"
            raise IndexError(msg)

        return {
            "joint_features": self._joint_features[index],
            "topology_features": self._topology_features[index],
            "bone_name_tokens": self._bone_name_tokens[index],
            "joint_mask": self._joint_mask[index],
            "target_deltas": self._target_deltas[index],
            "confidence_targets": self._confidence_targets[index],
            "source_indices": self._source_indices[index],
            "target_indices": self._target_indices[index],
            "edge_direction": self._edge_direction[index],
            "edge_mask": self._edge_mask[index],
        }

    def close(self) -> None:
        self._joint_features = torch.empty(0)
        self._topology_features = torch.empty(0)
        self._bone_name_tokens = torch.empty(0, dtype=torch.int64)
        self._joint_mask = torch.empty(0)
        self._target_deltas = torch.empty(0)
        self._confidence_targets = torch.empty(0)
        self._source_indices = torch.empty(0, dtype=torch.int64)
        self._target_indices = torch.empty(0, dtype=torch.int64)
        self._edge_direction = torch.empty(0, dtype=torch.int64)
        self._edge_mask = torch.empty(0)
