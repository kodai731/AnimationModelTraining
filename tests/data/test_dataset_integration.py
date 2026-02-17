from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from anim_ml.data.bvh_parser import parse_bvh
from anim_ml.data.curve_dataset import CurveCopilotDataset
from anim_ml.data.curve_extractor import extract_curve_samples

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _create_test_hdf5(tmp_path: Path) -> Path:
    motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
    samples = extract_curve_samples(motion)

    tmp_path.mkdir(parents=True, exist_ok=True)
    hdf5_path = tmp_path / "test_curves.h5"
    with h5py.File(hdf5_path, "w") as f:
        for split_name in ("train", "val"):
            grp = f.create_group(split_name)
            grp.create_dataset(
                "context_keyframes",
                data=np.stack([s.context_keyframes for s in samples]),
            )
            grp.create_dataset(
                "target",
                data=np.stack([s.target_keyframe for s in samples]),
            )
            grp.create_dataset(
                "property_type",
                data=np.array([s.property_type for s in samples], dtype=np.int32),
            )
            grp.create_dataset(
                "topology_features",
                data=np.stack([s.topology_features for s in samples]),
            )
            grp.create_dataset(
                "bone_name_tokens",
                data=np.stack([s.bone_name_tokens for s in samples]),
            )
            grp.create_dataset(
                "query_time",
                data=np.array([s.query_time for s in samples], dtype=np.float32),
            )

    return hdf5_path


@pytest.mark.unit
class TestCurveCopilotDataset:
    def test_length(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")
        assert len(dataset) > 0
        dataset.close()

    def test_getitem_types(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")

        item = dataset[0]
        assert isinstance(item["context_keyframes"], torch.Tensor)
        assert isinstance(item["target"], torch.Tensor)
        assert isinstance(item["property_type"], torch.Tensor)
        assert isinstance(item["topology_features"], torch.Tensor)
        assert isinstance(item["bone_name_tokens"], torch.Tensor)
        assert isinstance(item["query_time"], torch.Tensor)

        dataset.close()

    def test_getitem_shapes(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")

        item = dataset[0]
        assert item["context_keyframes"].shape == (8, 6)
        assert item["target"].shape == (6,)
        assert item["property_type"].shape == ()
        assert item["topology_features"].shape == (6,)
        assert item["bone_name_tokens"].shape == (32,)
        assert item["query_time"].shape == (1,)

        dataset.close()

    def test_getitem_dtypes(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")

        item = dataset[0]
        assert item["context_keyframes"].dtype == torch.float32
        assert item["target"].dtype == torch.float32
        assert item["property_type"].dtype == torch.int64
        assert item["topology_features"].dtype == torch.float32
        assert item["bone_name_tokens"].dtype == torch.int64
        assert item["query_time"].dtype == torch.float32

        dataset.close()

    def test_index_out_of_range(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")

        with pytest.raises(IndexError):
            dataset[len(dataset)]

        dataset.close()

    def test_close(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")
        dataset.close()
        assert len(dataset) == 0


@pytest.mark.unit
class TestMultipleHdf5Files:
    def test_combined_length(self, tmp_path: Path) -> None:
        path1 = _create_test_hdf5(tmp_path / "a")
        path2 = _create_test_hdf5(tmp_path / "b")

        ds_single = CurveCopilotDataset([path1], split="train")
        single_len = len(ds_single)
        ds_single.close()

        ds_combined = CurveCopilotDataset([path1, path2], split="train")
        assert len(ds_combined) == single_len * 2
        ds_combined.close()

    def test_access_across_files(self, tmp_path: Path) -> None:
        path1 = _create_test_hdf5(tmp_path / "a")
        path2 = _create_test_hdf5(tmp_path / "b")

        ds = CurveCopilotDataset([path1, path2], split="train")

        ds_single = CurveCopilotDataset([path1], split="train")
        boundary = len(ds_single)
        ds_single.close()

        item_before = ds[boundary - 1]
        item_after = ds[boundary]
        assert item_before["context_keyframes"].shape == (8, 6)
        assert item_after["context_keyframes"].shape == (8, 6)

        ds.close()


@pytest.mark.unit
class TestMissingSplit:
    def test_empty_for_missing_split(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="test")
        assert len(dataset) == 0
        dataset.close()
