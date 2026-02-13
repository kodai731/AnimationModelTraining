from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from anim_ml.data.rig_dataset import RigPropagationDataset


def _create_test_hdf5(tmp_path: Path, num_samples: int = 20) -> Path:
    hdf5_path = tmp_path / "test_rig.h5"
    rng = np.random.default_rng(42)

    with h5py.File(hdf5_path, "w") as f:
        adj_src = [0, 1, 0, 2, 0, 3, 1, 4, 2, 5, 3, 6, 4, 7, 5, 8, 6, 9, 7, 10, 8, 11]
        adj_tgt = [1, 0, 2, 0, 3, 0, 4, 1, 5, 2, 6, 3, 7, 4, 8, 5, 9, 6, 10, 7, 11, 8]
        num_edges = len(adj_src)
        adjacency = np.array([adj_src[:42] + adj_src[:42 - num_edges],
                              adj_tgt[:42] + adj_tgt[:42 - num_edges]], dtype=np.int64)

        real_adj_src = []
        real_adj_tgt = []
        parent_indices = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        for child, parent in enumerate(parent_indices):
            if parent == -1:
                continue
            real_adj_src.extend([parent, child])
            real_adj_tgt.extend([child, parent])
        adjacency = np.array([real_adj_src, real_adj_tgt], dtype=np.int64)
        f.create_dataset("adjacency", data=adjacency)

        for split_name in ("train", "val"):
            grp = f.create_group(split_name)
            joint_features = rng.standard_normal((num_samples, 22, 10)).astype(np.float32)
            joint_types = rng.integers(0, 13, (num_samples, 22)).astype(np.int64)

            target_deltas = rng.standard_normal((num_samples, 22, 4)).astype(np.float32)
            norms = np.linalg.norm(target_deltas, axis=-1, keepdims=True)
            target_deltas /= np.maximum(norms, 1e-8)

            confidence_targets = rng.choice([0.0, 1.0], size=(num_samples, 22)).astype(np.float32)

            grp.create_dataset("joint_features", data=joint_features)
            grp.create_dataset("joint_types", data=joint_types)
            grp.create_dataset("target_deltas", data=target_deltas)
            grp.create_dataset("confidence_targets", data=confidence_targets)

    return hdf5_path


@pytest.mark.unit
class TestRigDataset:
    def test_length(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path, num_samples=20)
        dataset = RigPropagationDataset([hdf5_path], split="train")
        assert len(dataset) == 20
        dataset.close()

    def test_getitem_shapes(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = RigPropagationDataset([hdf5_path], split="train")
        item = dataset[0]

        assert item["joint_features"].shape == (22, 10)
        assert item["joint_types"].shape == (22,)
        assert item["target_deltas"].shape == (22, 4)
        assert item["confidence_targets"].shape == (22,)
        dataset.close()

    def test_getitem_dtypes(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = RigPropagationDataset([hdf5_path], split="train")
        item = dataset[0]

        assert item["joint_features"].dtype == torch.float32
        assert item["joint_types"].dtype == torch.int64
        assert item["target_deltas"].dtype == torch.float32
        assert item["confidence_targets"].dtype == torch.float32
        dataset.close()

    def test_adjacency_property(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = RigPropagationDataset([hdf5_path], split="train")
        adj = dataset.adjacency
        assert adj.shape[0] == 2
        assert adj.shape[1] == 42
        dataset.close()


@pytest.mark.unit
class TestDataLoaderCompatibility:
    def test_batching(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = RigPropagationDataset([hdf5_path], split="train")

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            dataset, batch_size=4, shuffle=False, num_workers=0,
        )
        batch = next(iter(loader))

        assert batch["joint_features"].shape == (4, 22, 10)
        assert batch["joint_types"].shape == (4, 22)
        assert batch["target_deltas"].shape == (4, 22, 4)
        assert batch["confidence_targets"].shape == (4, 22)
        dataset.close()

    def test_full_iteration(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path, num_samples=10)
        dataset = RigPropagationDataset([hdf5_path], split="train")

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            dataset, batch_size=4, shuffle=False, num_workers=0,
        )

        total = 0
        for batch in loader:
            total += batch["joint_features"].shape[0]

        assert total == 10
        dataset.close()
