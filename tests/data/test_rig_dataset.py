from __future__ import annotations

import gc
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from anim_ml.data.rig_data_generator import MAX_EDGES, MAX_JOINTS
from anim_ml.data.rig_dataset import RigPropagationDataset

NUM_TEST_JOINTS = 20
PARENT_INDICES = [
    -1, 0, 0, 0,
    1, 2, 3,
    4, 5, 6,
    9, 9, 9,
    10, 11, 12,
    14, 15,
    16, 17,
]


def _create_test_hdf5(tmp_path: Path, num_samples: int = 20) -> Path:
    hdf5_path = tmp_path / "test_rig.h5"
    rng = np.random.default_rng(42)

    src_list = []
    tgt_list = []
    for child, parent in enumerate(PARENT_INDICES):
        if parent == -1:
            continue
        src_list.extend([parent, child])
        tgt_list.extend([child, parent])
    num_real_edges = len(src_list)

    with h5py.File(hdf5_path, "w") as f:
        for split_name in ("train", "val"):
            grp = f.create_group(split_name)

            joint_features = np.zeros((num_samples, MAX_JOINTS, 9), dtype=np.float32)
            joint_features[:, :NUM_TEST_JOINTS, :] = rng.standard_normal(
                (num_samples, NUM_TEST_JOINTS, 9),
            ).astype(np.float32)

            topology_features = np.zeros((num_samples, MAX_JOINTS, 6), dtype=np.float32)
            topology_features[:, :NUM_TEST_JOINTS, :] = rng.standard_normal(
                (num_samples, NUM_TEST_JOINTS, 6),
            ).astype(np.float32)

            bone_name_tokens = np.zeros((num_samples, MAX_JOINTS, 32), dtype=np.int64)
            bone_name_tokens[:, :NUM_TEST_JOINTS, :] = rng.integers(
                0, 44, (num_samples, NUM_TEST_JOINTS, 32),
            ).astype(np.int64)

            joint_mask = np.zeros((num_samples, MAX_JOINTS), dtype=np.float32)
            joint_mask[:, :NUM_TEST_JOINTS] = 1.0

            target_deltas = np.zeros((num_samples, MAX_JOINTS, 4), dtype=np.float32)
            raw_deltas = rng.standard_normal((num_samples, NUM_TEST_JOINTS, 4)).astype(np.float32)
            norms = np.linalg.norm(raw_deltas, axis=-1, keepdims=True)
            target_deltas[:, :NUM_TEST_JOINTS, :] = raw_deltas / np.maximum(norms, 1e-8)

            confidence_targets = np.zeros((num_samples, MAX_JOINTS), dtype=np.float32)
            confidence_targets[:, :NUM_TEST_JOINTS] = rng.choice(
                [0.0, 1.0], size=(num_samples, NUM_TEST_JOINTS),
            ).astype(np.float32)

            source_indices = np.zeros((num_samples, MAX_EDGES), dtype=np.int64)
            target_indices = np.zeros((num_samples, MAX_EDGES), dtype=np.int64)
            edge_direction = np.zeros((num_samples, MAX_EDGES), dtype=np.int64)
            edge_mask_arr = np.zeros((num_samples, MAX_EDGES), dtype=np.int64)

            for i in range(num_samples):
                source_indices[i, :num_real_edges] = src_list
                target_indices[i, :num_real_edges] = tgt_list
                for e in range(num_real_edges):
                    s, t = src_list[e], tgt_list[e]
                    edge_direction[i, e] = 0 if PARENT_INDICES[t] == s else 1
                edge_mask_arr[i, :num_real_edges] = 1

            grp.create_dataset("joint_features", data=joint_features)
            grp.create_dataset("topology_features", data=topology_features)
            grp.create_dataset("bone_name_tokens", data=bone_name_tokens)
            grp.create_dataset("joint_mask", data=joint_mask)
            grp.create_dataset("target_deltas", data=target_deltas)
            grp.create_dataset("confidence_targets", data=confidence_targets)
            grp.create_dataset("source_indices", data=source_indices)
            grp.create_dataset("target_indices", data=target_indices)
            grp.create_dataset("edge_direction", data=edge_direction)
            grp.create_dataset("edge_mask", data=edge_mask_arr)

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

        assert item["joint_features"].shape == (MAX_JOINTS, 9)
        assert item["topology_features"].shape == (MAX_JOINTS, 6)
        assert item["bone_name_tokens"].shape == (MAX_JOINTS, 32)
        assert item["joint_mask"].shape == (MAX_JOINTS,)
        assert item["target_deltas"].shape == (MAX_JOINTS, 4)
        assert item["confidence_targets"].shape == (MAX_JOINTS,)
        assert item["source_indices"].shape == (MAX_EDGES,)
        assert item["target_indices"].shape == (MAX_EDGES,)
        assert item["edge_direction"].shape == (MAX_EDGES,)
        assert item["edge_mask"].shape == (MAX_EDGES,)
        dataset.close()

    def test_getitem_dtypes(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = RigPropagationDataset([hdf5_path], split="train")
        item = dataset[0]

        assert item["joint_features"].dtype == torch.float32
        assert item["topology_features"].dtype == torch.float32
        assert item["bone_name_tokens"].dtype == torch.int64
        assert item["joint_mask"].dtype == torch.float32
        assert item["target_deltas"].dtype == torch.float32
        assert item["confidence_targets"].dtype == torch.float32
        assert item["source_indices"].dtype == torch.int64
        assert item["target_indices"].dtype == torch.int64
        assert item["edge_direction"].dtype == torch.int64
        assert item["edge_mask"].dtype == torch.float32
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

        assert batch["joint_features"].shape == (4, MAX_JOINTS, 9)
        assert batch["topology_features"].shape == (4, MAX_JOINTS, 6)
        assert batch["bone_name_tokens"].shape == (4, MAX_JOINTS, 32)
        assert batch["joint_mask"].shape == (4, MAX_JOINTS)
        assert batch["target_deltas"].shape == (4, MAX_JOINTS, 4)
        assert batch["confidence_targets"].shape == (4, MAX_JOINTS)
        assert batch["source_indices"].shape == (4, MAX_EDGES)
        assert batch["target_indices"].shape == (4, MAX_EDGES)
        assert batch["edge_direction"].shape == (4, MAX_EDGES)
        assert batch["edge_mask"].shape == (4, MAX_EDGES)
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


@pytest.mark.unit
class TestMemoryOptimization:
    def test_all_samples_accessible_after_chunks_freed(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path, num_samples=20)
        dataset = RigPropagationDataset([hdf5_path], split="train", use_shared_memory=False)

        for i in range(len(dataset)):
            item = dataset[i]
            assert item["joint_features"].shape == (MAX_JOINTS, 9)

        dataset.close()

    def test_shared_and_non_shared_data_match(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path, num_samples=10)

        ds_shared = RigPropagationDataset([hdf5_path], split="train", use_shared_memory=True)
        ds_plain = RigPropagationDataset([hdf5_path], split="train", use_shared_memory=False)

        assert len(ds_shared) == len(ds_plain)
        for i in range(len(ds_shared)):
            shared_item = ds_shared[i]
            plain_item = ds_plain[i]
            assert torch.equal(shared_item["joint_features"], plain_item["joint_features"])
            assert torch.equal(shared_item["target_deltas"], plain_item["target_deltas"])
            assert torch.equal(shared_item["bone_name_tokens"], plain_item["bone_name_tokens"])

        ds_shared.close()
        ds_plain.close()

    def test_dataloader_works_after_gc(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path, num_samples=10)
        dataset = RigPropagationDataset([hdf5_path], split="train", use_shared_memory=False)

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            dataset, batch_size=4, shuffle=False, num_workers=0,
        )

        gc.collect()

        total = 0
        for batch in loader:
            total += batch["joint_features"].shape[0]
            assert batch["joint_features"].ndim == 3

        assert total == 10
        dataset.close()
