from __future__ import annotations

import gc
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

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


@pytest.mark.unit
class TestMemoryOptimization:
    def test_full_preload(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")

        assert dataset.is_fully_loaded
        assert len(dataset) == dataset.total_count
        for i in range(len(dataset)):
            item = dataset[i]
            assert item["context_keyframes"].shape == (8, 6)

        dataset.close()

    def test_chunk_preload_with_small_budget(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train", cache_budget_bytes=1)

        assert len(dataset) == 1
        assert dataset.total_count > 1
        assert not dataset.is_fully_loaded

        item = dataset[0]
        assert item["context_keyframes"].shape == (8, 6)

        dataset.close()

    def test_reload_chunk_matches_full_load(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)

        full_ds = CurveCopilotDataset([hdf5_path], split="val")
        total = full_ds.total_count
        full_targets = [full_ds[i]["target"].clone() for i in range(total)]
        full_ds.close()

        chunked_ds = CurveCopilotDataset(
            [hdf5_path], split="val", cache_budget_bytes=1,
        )
        assert len(chunked_ds) == 1

        chunk_targets = []
        for i in range(total):
            chunk_targets.append(chunked_ds[0]["target"].clone())
            if i < total - 1:
                chunked_ds.reload_chunk()

        for i in range(total):
            assert torch.equal(full_targets[i], chunk_targets[i])

        chunked_ds.close()

    def test_dataloader_works_after_gc(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            dataset, batch_size=4, shuffle=False, num_workers=0,
        )

        gc.collect()

        total = 0
        for batch in loader:
            total += batch["context_keyframes"].shape[0]
            assert batch["context_keyframes"].ndim == 3

        assert total == len(dataset)
        dataset.close()


def _create_chunked_curve_dataset(
    tmp_path: Path, *, num_files: int = 3,
) -> tuple[CurveCopilotDataset, list[Path]]:
    paths = [_create_test_hdf5(tmp_path / str(i)) for i in range(num_files)]

    full_ds = CurveCopilotDataset(paths, split="val")
    per_sample = full_ds._per_sample_bytes
    total = full_ds.total_count
    full_ds.close()

    budget = per_sample * max(total // 3, 2)
    ds = CurveCopilotDataset(paths, split="val", cache_budget_bytes=budget)
    assert not ds.is_fully_loaded
    assert ds._chunk_size > 1
    return ds, paths


@pytest.mark.unit
class TestChunkRandomization:
    def test_begin_epoch_sets_offset_in_range(self, tmp_path: Path) -> None:
        dataset, _ = _create_chunked_curve_dataset(tmp_path)

        for epoch in range(10):
            dataset.begin_epoch(epoch)
            assert 0 <= dataset._epoch_offset < dataset._chunk_size

        dataset.close()

    def test_begin_epoch_no_offset_when_fully_loaded(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")
        assert dataset.is_fully_loaded

        dataset.begin_epoch(42)
        assert dataset._epoch_offset == 0
        dataset.close()

    def test_different_epochs_produce_different_offsets(self, tmp_path: Path) -> None:
        dataset, _ = _create_chunked_curve_dataset(tmp_path)

        offsets = set()
        for epoch in range(20):
            dataset.begin_epoch(epoch)
            offsets.add(dataset._epoch_offset)

        assert len(offsets) > 1
        dataset.close()

    def test_all_samples_covered_across_chunks(self, tmp_path: Path) -> None:
        paths = [_create_test_hdf5(tmp_path / d) for d in ("fa", "fb", "fc")]

        full_ds = CurveCopilotDataset(paths, split="val")
        total = full_ds.total_count
        expected = {full_ds[i]["target"].numpy().tobytes() for i in range(total)}
        full_ds.close()

        chunked_ds, _ = _create_chunked_curve_dataset(tmp_path / "chunked")

        chunked_ds.begin_epoch(7)
        collected: set[bytes] = set()
        for chunk_idx in range(chunked_ds.num_chunks):
            chunked_ds.reload_chunk(chunk_idx)
            for i in range(len(chunked_ds)):
                collected.add(chunked_ds[i]["target"].numpy().tobytes())

        assert collected == expected
        chunked_ds.close()

    def test_begin_epoch_is_reproducible(self, tmp_path: Path) -> None:
        dataset, _ = _create_chunked_curve_dataset(tmp_path)

        dataset.begin_epoch(3)
        offset_first = dataset._epoch_offset

        dataset.begin_epoch(3)
        offset_second = dataset._epoch_offset

        assert offset_first == offset_second
        dataset.close()
