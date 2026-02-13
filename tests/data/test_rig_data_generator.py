from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anim_ml.data.bvh_parser import MotionData
from anim_ml.data.rig_data_generator import (
    SMPL_MIRROR_PAIRS,
    RigJointType,
    build_adjacency,
    classify_rig_joint,
    compute_hierarchy_depth,
    generate_rig_samples_from_motion,
    save_rig_samples_hdf5,
)
from anim_ml.utils.skeleton import SMPL_22_JOINT_NAMES, SMPL_22_PARENT_INDICES


def _make_synthetic_motion(num_frames: int = 30) -> MotionData:
    rng = np.random.default_rng(123)
    rotations = rng.uniform(-30.0, 30.0, size=(num_frames, 22, 3))
    positions = np.zeros((num_frames, 22, 3))
    positions[:, 0, 1] = 1.0

    return MotionData(
        joint_names=list(SMPL_22_JOINT_NAMES),
        parent_indices=list(SMPL_22_PARENT_INDICES),
        frame_time=1.0 / 30.0,
        positions=positions,
        rotations=rotations,
    )


@pytest.mark.unit
class TestBuildAdjacency:
    def test_edge_count(self) -> None:
        adj = build_adjacency(SMPL_22_PARENT_INDICES)
        assert adj.shape == (2, 42)

    def test_bidirectional(self) -> None:
        adj = build_adjacency(SMPL_22_PARENT_INDICES)
        edge_set = {(adj[0, i], adj[1, i]) for i in range(adj.shape[1])}
        for src, tgt in list(edge_set):
            assert (tgt, src) in edge_set

    def test_no_self_loops(self) -> None:
        adj = build_adjacency(SMPL_22_PARENT_INDICES)
        for i in range(adj.shape[1]):
            assert adj[0, i] != adj[1, i]


@pytest.mark.unit
class TestClassifyRigJoint:
    def test_all_joints_classified(self) -> None:
        for name in SMPL_22_JOINT_NAMES:
            jtype = classify_rig_joint(name)
            assert isinstance(jtype, RigJointType)

    def test_pelvis_is_root(self) -> None:
        assert classify_rig_joint("pelvis") == RigJointType.ROOT

    def test_left_hip_is_upper_leg(self) -> None:
        assert classify_rig_joint("left_hip") == RigJointType.UPPER_LEG

    def test_head_is_head(self) -> None:
        assert classify_rig_joint("head") == RigJointType.HEAD


@pytest.mark.unit
class TestComputeHierarchyDepth:
    def test_root_is_zero(self) -> None:
        depths = compute_hierarchy_depth(SMPL_22_PARENT_INDICES)
        assert depths[0] == 0.0

    def test_normalized_range(self) -> None:
        depths = compute_hierarchy_depth(SMPL_22_PARENT_INDICES)
        assert depths.min() >= 0.0
        assert depths.max() <= 1.0

    def test_max_depth_is_one(self) -> None:
        depths = compute_hierarchy_depth(SMPL_22_PARENT_INDICES)
        assert np.isclose(depths.max(), 1.0)


@pytest.mark.unit
class TestGenerateRigSamples:
    def test_produces_samples(self) -> None:
        motion = _make_synthetic_motion()
        samples = generate_rig_samples_from_motion(
            motion, frame_skips=[1], augment_mirror=False, augment_noise=False,
            rng=np.random.default_rng(0),
        )
        assert len(samples) > 0

    def test_feature_shape(self) -> None:
        motion = _make_synthetic_motion()
        samples = generate_rig_samples_from_motion(
            motion, frame_skips=[1], augment_mirror=False, augment_noise=False,
            rng=np.random.default_rng(0),
        )
        sample = samples[0]
        assert sample.joint_features.shape == (22, 10)
        assert sample.joint_types.shape == (22,)
        assert sample.target_deltas.shape == (22, 4)
        assert sample.confidence_targets.shape == (22,)

    def test_edited_joints_have_identity_target(self) -> None:
        motion = _make_synthetic_motion()
        samples = generate_rig_samples_from_motion(
            motion, frame_skips=[2], num_edited_range=(1, 1),
            augment_mirror=False, augment_noise=False,
            rng=np.random.default_rng(42),
        )
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        for sample in samples[:5]:
            edited_mask = sample.joint_features[:, 8] == 1.0
            assert edited_mask.sum() >= 1
            for j in range(22):
                if edited_mask[j]:
                    np.testing.assert_allclose(
                        sample.target_deltas[j], identity, atol=1e-5,
                    )

    def test_base_quats_are_unit(self) -> None:
        motion = _make_synthetic_motion()
        samples = generate_rig_samples_from_motion(
            motion, frame_skips=[1], augment_mirror=False, augment_noise=False,
            rng=np.random.default_rng(0),
        )
        for sample in samples[:5]:
            norms = np.linalg.norm(sample.joint_features[:, :4], axis=-1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_confidence_is_binary(self) -> None:
        motion = _make_synthetic_motion()
        samples = generate_rig_samples_from_motion(
            motion, frame_skips=[1], augment_mirror=False, augment_noise=False,
            rng=np.random.default_rng(0),
        )
        for sample in samples[:5]:
            for val in sample.confidence_targets:
                assert val in (0.0, 1.0)

    def test_no_nan(self) -> None:
        motion = _make_synthetic_motion()
        samples = generate_rig_samples_from_motion(
            motion, frame_skips=[1, 2], augment_mirror=True, augment_noise=True,
            rng=np.random.default_rng(0),
        )
        for sample in samples:
            assert not np.isnan(sample.joint_features).any()
            assert not np.isnan(sample.target_deltas).any()
            assert not np.isnan(sample.confidence_targets).any()

    def test_num_edited_in_range(self) -> None:
        motion = _make_synthetic_motion()
        samples = generate_rig_samples_from_motion(
            motion, frame_skips=[1], num_edited_range=(1, 3),
            augment_mirror=False, augment_noise=False,
            rng=np.random.default_rng(0),
        )
        for sample in samples[:20]:
            num_edited = int(sample.joint_features[:, 8].sum())
            assert 1 <= num_edited <= 3


@pytest.mark.unit
class TestMirrorAugmentation:
    def test_mirror_produces_samples(self) -> None:
        motion = _make_synthetic_motion()
        samples_no_mirror = generate_rig_samples_from_motion(
            motion, frame_skips=[1], augment_mirror=False, augment_noise=False,
            rng=np.random.default_rng(0),
        )
        samples_with_mirror = generate_rig_samples_from_motion(
            motion, frame_skips=[1], augment_mirror=True, augment_noise=False,
            rng=np.random.default_rng(0),
        )
        assert len(samples_with_mirror) > len(samples_no_mirror)


@pytest.mark.unit
class TestSaveHdf5:
    def test_creates_file(self, tmp_path: Path) -> None:
        motion = _make_synthetic_motion()
        samples = generate_rig_samples_from_motion(
            motion, frame_skips=[1], augment_mirror=False, augment_noise=False,
            rng=np.random.default_rng(0),
        )
        adjacency = build_adjacency(SMPL_22_PARENT_INDICES)
        output_path = tmp_path / "test_rig.h5"
        save_rig_samples_hdf5(samples, output_path, adjacency)
        assert output_path.exists()

    def test_split_groups(self, tmp_path: Path) -> None:
        import h5py

        motion = _make_synthetic_motion(num_frames=60)
        samples = generate_rig_samples_from_motion(
            motion, frame_skips=[1], augment_mirror=False, augment_noise=False,
            rng=np.random.default_rng(0),
        )
        adjacency = build_adjacency(SMPL_22_PARENT_INDICES)
        output_path = tmp_path / "test_rig.h5"
        save_rig_samples_hdf5(samples, output_path, adjacency)

        with h5py.File(output_path, "r") as f:
            assert "adjacency" in f
            assert f["adjacency"].shape == (2, 42)
            assert "train" in f
            assert "joint_features" in f["train"]
            assert "target_deltas" in f["train"]
            assert "confidence_targets" in f["train"]
            assert "joint_types" in f["train"]

    def test_feature_shapes_in_hdf5(self, tmp_path: Path) -> None:
        import h5py

        motion = _make_synthetic_motion(num_frames=60)
        samples = generate_rig_samples_from_motion(
            motion, frame_skips=[1], augment_mirror=False, augment_noise=False,
            rng=np.random.default_rng(0),
        )
        adjacency = build_adjacency(SMPL_22_PARENT_INDICES)
        output_path = tmp_path / "test_rig.h5"
        save_rig_samples_hdf5(samples, output_path, adjacency)

        with h5py.File(output_path, "r") as f:
            train_grp = f["train"]
            n = train_grp["joint_features"].shape[0]
            assert train_grp["joint_features"].shape == (n, 22, 10)
            assert train_grp["target_deltas"].shape == (n, 22, 4)
            assert train_grp["confidence_targets"].shape == (n, 22)
            assert train_grp["joint_types"].shape == (n, 22)
