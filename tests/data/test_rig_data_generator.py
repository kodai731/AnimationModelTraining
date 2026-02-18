from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anim_ml.data.bvh_parser import MotionData
from anim_ml.data.rig_data_generator import (
    MAX_EDGES,
    MAX_JOINTS,
    build_adjacency,
    detect_mirror_pairs,
    generate_rig_samples_from_motion,
    save_rig_samples_hdf5,
)

SAMPLE_JOINT_NAMES = [
    "Hips", "LeftUpLeg", "RightUpLeg", "Spine",
    "LeftLeg", "RightLeg", "Spine1",
    "LeftFoot", "RightFoot", "Spine2",
    "Neck", "LeftShoulder", "RightShoulder",
    "Head", "LeftArm", "RightArm",
    "LeftForeArm", "RightForeArm",
    "LeftHand", "RightHand",
]

SAMPLE_PARENT_INDICES = [
    -1, 0, 0, 0,
    1, 2, 3,
    4, 5, 6,
    9, 9, 9,
    10, 11, 12,
    14, 15,
    16, 17,
]


def _make_synthetic_motion(num_frames: int = 30) -> MotionData:
    rng = np.random.default_rng(123)
    num_joints = len(SAMPLE_JOINT_NAMES)
    rotations = rng.uniform(-30.0, 30.0, size=(num_frames, num_joints, 3))
    positions = np.zeros((num_frames, num_joints, 3))
    positions[:, 0, 1] = 1.0

    return MotionData(
        joint_names=list(SAMPLE_JOINT_NAMES),
        parent_indices=list(SAMPLE_PARENT_INDICES),
        frame_time=1.0 / 30.0,
        positions=positions,
        rotations=rotations,
    )


@pytest.mark.unit
class TestBuildAdjacency:
    def test_bidirectional(self) -> None:
        adj = build_adjacency(SAMPLE_PARENT_INDICES)
        edge_set = {(adj[0, i], adj[1, i]) for i in range(adj.shape[1])}
        for src, tgt in list(edge_set):
            assert (tgt, src) in edge_set

    def test_no_self_loops(self) -> None:
        adj = build_adjacency(SAMPLE_PARENT_INDICES)
        for i in range(adj.shape[1]):
            assert adj[0, i] != adj[1, i]

    def test_edge_count(self) -> None:
        adj = build_adjacency(SAMPLE_PARENT_INDICES)
        num_non_root = sum(1 for p in SAMPLE_PARENT_INDICES if p != -1)
        assert adj.shape[1] == num_non_root * 2


@pytest.mark.unit
class TestDetectMirrorPairs:
    def test_detects_left_right(self) -> None:
        pairs = detect_mirror_pairs(SAMPLE_JOINT_NAMES)
        assert len(pairs) > 0

        pair_set = {(min(a, b), max(a, b)) for a, b in pairs}
        left_up = SAMPLE_JOINT_NAMES.index("LeftUpLeg")
        right_up = SAMPLE_JOINT_NAMES.index("RightUpLeg")
        assert (min(left_up, right_up), max(left_up, right_up)) in pair_set

    def test_no_duplicate_pairs(self) -> None:
        pairs = detect_mirror_pairs(SAMPLE_JOINT_NAMES)
        all_indices = []
        for a, b in pairs:
            all_indices.extend([a, b])
        assert len(all_indices) == len(set(all_indices))

    def test_underscore_style(self) -> None:
        names = ["root", "left_arm", "right_arm", "spine"]
        pairs = detect_mirror_pairs(names)
        assert len(pairs) == 1
        indices = {names[pairs[0][0]], names[pairs[0][1]]}
        assert indices == {"left_arm", "right_arm"}

    def test_no_pairs_for_symmetric(self) -> None:
        names = ["root", "spine", "head"]
        pairs = detect_mirror_pairs(names)
        assert len(pairs) == 0


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
        num_joints = len(SAMPLE_JOINT_NAMES)
        sample = samples[0]
        assert sample.joint_features.shape == (num_joints, 9)
        assert sample.target_deltas.shape == (num_joints, 4)
        assert sample.confidence_targets.shape == (num_joints,)
        assert sample.topology_features.shape == (num_joints, 6)
        assert sample.bone_name_tokens.shape == (num_joints, 32)
        assert len(sample.parent_indices) == num_joints

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
            num_joints = len(sample.parent_indices)
            for j in range(num_joints):
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
        output_path = tmp_path / "test_rig.h5"
        save_rig_samples_hdf5(samples, output_path)
        assert output_path.exists()

    def test_split_groups(self, tmp_path: Path) -> None:
        import h5py

        motion = _make_synthetic_motion(num_frames=60)
        samples = generate_rig_samples_from_motion(
            motion, frame_skips=[1], augment_mirror=False, augment_noise=False,
            rng=np.random.default_rng(0),
        )
        output_path = tmp_path / "test_rig.h5"
        save_rig_samples_hdf5(samples, output_path)

        with h5py.File(output_path, "r") as f:
            assert "train" in f
            train_grp = f["train"]
            assert "joint_features" in train_grp
            assert "target_deltas" in train_grp
            assert "confidence_targets" in train_grp
            assert "topology_features" in train_grp
            assert "bone_name_tokens" in train_grp
            assert "joint_mask" in train_grp
            assert "source_indices" in train_grp
            assert "target_indices" in train_grp
            assert "edge_direction" in train_grp
            assert "edge_mask" in train_grp

    def test_feature_shapes_in_hdf5(self, tmp_path: Path) -> None:
        import h5py

        motion = _make_synthetic_motion(num_frames=60)
        samples = generate_rig_samples_from_motion(
            motion, frame_skips=[1], augment_mirror=False, augment_noise=False,
            rng=np.random.default_rng(0),
        )
        output_path = tmp_path / "test_rig.h5"
        save_rig_samples_hdf5(samples, output_path)

        with h5py.File(output_path, "r") as f:
            train_grp = f["train"]
            n = train_grp["joint_features"].shape[0]
            assert train_grp["joint_features"].shape == (n, MAX_JOINTS, 9)
            assert train_grp["target_deltas"].shape == (n, MAX_JOINTS, 4)
            assert train_grp["confidence_targets"].shape == (n, MAX_JOINTS)
            assert train_grp["topology_features"].shape == (n, MAX_JOINTS, 6)
            assert train_grp["bone_name_tokens"].shape == (n, MAX_JOINTS, 32)
            assert train_grp["joint_mask"].shape == (n, MAX_JOINTS)
            assert train_grp["source_indices"].shape == (n, MAX_EDGES)
            assert train_grp["target_indices"].shape == (n, MAX_EDGES)
            assert train_grp["edge_direction"].shape == (n, MAX_EDGES)
            assert train_grp["edge_mask"].shape == (n, MAX_EDGES)
