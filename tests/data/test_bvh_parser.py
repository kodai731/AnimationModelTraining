from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from anim_ml.data.bvh_parser import parse_bvh

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.mark.unit
class TestParseHierarchy:
    def test_joint_names(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        assert motion.joint_names == ["Hips", "Spine", "Head"]

    def test_parent_indices(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        assert motion.parent_indices == [-1, 0, 1]

    def test_end_site_excluded(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        assert len(motion.joint_names) == 3


@pytest.mark.unit
class TestParseMotion:
    def test_frame_time(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        assert_allclose(motion.frame_time, 0.0333333, atol=1e-5)

    def test_frame_count(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        assert motion.positions.shape[0] == 10
        assert motion.rotations.shape[0] == 10

    def test_shapes(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        assert motion.positions.shape == (10, 3, 3)
        assert motion.rotations.shape == (10, 3, 3)

    def test_root_position_values(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        assert_allclose(motion.positions[0, 0], [0.0, 90.0, 0.0], atol=1e-5)
        assert_allclose(motion.positions[4, 0], [4.0, 90.0, 4.0], atol=1e-5)

    def test_non_root_positions_zero(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        assert_allclose(motion.positions[:, 1:, :], 0.0, atol=1e-7)


@pytest.mark.unit
class TestEulerNormalization:
    def test_rotations_are_xyz_euler(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        assert motion.rotations[0, 0, 0] == pytest.approx(0.0, abs=1e-3)
        assert motion.rotations[0, 0, 1] == pytest.approx(0.0, abs=1e-3)
        assert motion.rotations[0, 0, 2] == pytest.approx(0.0, abs=1e-3)

    def test_nonzero_rotations_preserved(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        spine_rot_frame4 = motion.rotations[4, 1]
        assert not np.allclose(spine_rot_frame4, 0.0, atol=0.1)


@pytest.mark.unit
class TestZUpConversion:
    def test_z_up_root_height_on_y_axis(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple_zup.bvh", z_up=True)
        assert motion.positions[0, 0, 1] == pytest.approx(90.0, abs=1e-3)

    def test_z_up_shape_preserved(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple_zup.bvh", z_up=True)
        assert motion.positions.shape == (10, 3, 3)
        assert motion.rotations.shape == (10, 3, 3)

    def test_z_up_and_y_up_same_structure(self) -> None:
        motion_yup = parse_bvh(FIXTURES_DIR / "simple.bvh")
        motion_zup = parse_bvh(FIXTURES_DIR / "simple_zup.bvh", z_up=True)
        assert motion_yup.joint_names == motion_zup.joint_names
        assert motion_yup.parent_indices == motion_zup.parent_indices
