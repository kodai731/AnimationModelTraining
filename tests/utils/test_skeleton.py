import numpy as np
import pytest
from numpy.testing import assert_allclose

from anim_ml.utils.rotation import euler_xyz_to_quaternion, quaternion_multiply
from anim_ml.utils.skeleton import (
    SMPL_22_JOINT_NAMES,
    SMPL_22_PARENT_INDICES,
    SMPL_TO_VRM_MAPPING,
    JointCategory,
    classify_joint,
    compute_world_rotation,
)

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


@pytest.mark.unit
class TestMappingCompleteness:
    def test_joint_count(self) -> None:
        assert len(SMPL_22_JOINT_NAMES) == 22

    def test_parent_count(self) -> None:
        assert len(SMPL_22_PARENT_INDICES) == 22

    def test_vrm_mapping_covers_all_joints(self) -> None:
        assert len(SMPL_TO_VRM_MAPPING) == 22
        for i in range(22):
            assert i in SMPL_TO_VRM_MAPPING


@pytest.mark.unit
class TestParentValidity:
    def test_root_is_negative_one(self) -> None:
        assert SMPL_22_PARENT_INDICES[0] == -1

    def test_all_parents_precede_child(self) -> None:
        for i in range(1, 22):
            parent = SMPL_22_PARENT_INDICES[i]
            name = SMPL_22_JOINT_NAMES[i]
            assert 0 <= parent < i, f"Joint {i} ({name}) has invalid parent {parent}"

    def test_no_orphans(self) -> None:
        reachable = {0}
        for i in range(1, 22):
            parent = SMPL_22_PARENT_INDICES[i]
            assert parent in reachable, f"Joint {i} parent {parent} not yet reachable"
            reachable.add(i)
        assert len(reachable) == 22


@pytest.mark.unit
class TestJointClassification:
    def test_known_classifications(self) -> None:
        assert classify_joint("pelvis") == JointCategory.ROOT
        assert classify_joint("left_hip") == JointCategory.LEG
        assert classify_joint("right_knee") == JointCategory.LEG
        assert classify_joint("left_shoulder") == JointCategory.ARM
        assert classify_joint("right_elbow") == JointCategory.ARM
        assert classify_joint("head") == JointCategory.HEAD
        assert classify_joint("neck") == JointCategory.HEAD
        assert classify_joint("left_wrist") == JointCategory.HAND
        assert classify_joint("left_foot") == JointCategory.FOOT

    def test_all_smpl_joints_classifiable(self) -> None:
        for name in SMPL_22_JOINT_NAMES:
            category = classify_joint(name)
            assert isinstance(category, JointCategory)

    def test_fallback_heuristic(self) -> None:
        assert classify_joint("some_shoulder_joint") == JointCategory.ARM
        assert classify_joint("left_ankle_extra") == JointCategory.FOOT


@pytest.mark.unit
class TestWorldRotation:
    def test_tpose_all_identity(self) -> None:
        local_rots = np.tile(IDENTITY_QUAT, (22, 1))
        for i in range(22):
            world = compute_world_rotation(SMPL_22_PARENT_INDICES, local_rots, i)
            assert_allclose(world, IDENTITY_QUAT, atol=1e-7)

    def test_parent_rotation_propagates(self) -> None:
        local_rots = np.tile(IDENTITY_QUAT, (22, 1))

        parent_rot = euler_xyz_to_quaternion(np.array([np.pi / 2, 0.0, 0.0]))
        pelvis_idx = 0
        local_rots[pelvis_idx] = parent_rot

        spine1_idx = 3
        world = compute_world_rotation(SMPL_22_PARENT_INDICES, local_rots, spine1_idx)
        assert_allclose(world, parent_rot, atol=1e-6)

    def test_chain_composition(self) -> None:
        local_rots = np.tile(IDENTITY_QUAT, (22, 1))

        rot_a = euler_xyz_to_quaternion(np.array([np.pi / 4, 0.0, 0.0]))
        rot_b = euler_xyz_to_quaternion(np.array([0.0, np.pi / 4, 0.0]))

        pelvis_idx = 0
        spine1_idx = 3
        local_rots[pelvis_idx] = rot_a
        local_rots[spine1_idx] = rot_b

        expected = quaternion_multiply(rot_a, rot_b)
        world = compute_world_rotation(SMPL_22_PARENT_INDICES, local_rots, spine1_idx)
        assert_allclose(world, expected, atol=1e-6)
