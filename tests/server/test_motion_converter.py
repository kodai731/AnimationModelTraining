import numpy as np
import pytest
from numpy.testing import assert_allclose

from anim_ml.server.motion_converter import (
    PROPERTY_ROTATION_X,
    PROPERTY_ROTATION_Y,
    PROPERTY_ROTATION_Z,
    PROPERTY_TRANSLATION_X,
    PROPERTY_TRANSLATION_Y,
    PROPERTY_TRANSLATION_Z,
    ConversionConfig,
    MotionCurveData,
    convert_6d_rotations_to_euler,
    convert_humanml3d_to_curves,
    extract_joint_rotations_6d,
    extract_root_rotation,
    extract_root_translation,
    resample_motion,
    retarget_to_vrm,
    validate_motion_curves,
)


def _make_synthetic_tensor(num_frames: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tensor = np.zeros((num_frames, 263), dtype=np.float32)

    identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    for j in range(21):
        offset = 130 + j * 6
        tensor[:, offset:offset + 6] = identity_6d

    tensor[:, 3] = 0.9
    tensor[:, 4:130] = rng.standard_normal((num_frames, 126)).astype(np.float32) * 0.01
    tensor[:, 256:260] = 0.0
    tensor[:, 260:263] = 0.0

    return tensor


@pytest.mark.unit
class TestExtractJointRotations:
    def test_shape(self) -> None:
        tensor = _make_synthetic_tensor(60)
        result = extract_joint_rotations_6d(tensor)
        assert result.shape == (60, 21, 6)

    def test_identity_rotation_values(self) -> None:
        tensor = _make_synthetic_tensor(10)
        result = extract_joint_rotations_6d(tensor)

        identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        for frame in range(10):
            for joint in range(21):
                assert_allclose(result[frame, joint], identity_6d, atol=1e-6)

    def test_single_frame(self) -> None:
        tensor = _make_synthetic_tensor(1)
        result = extract_joint_rotations_6d(tensor)
        assert result.shape == (1, 21, 6)


@pytest.mark.unit
class TestConvert6dToEuler:
    def test_identity_gives_zero_euler(self) -> None:
        rot_6d = np.tile([1, 0, 0, 0, 1, 0], (5, 3, 1)).astype(np.float64)
        euler = convert_6d_rotations_to_euler(rot_6d)
        assert euler.shape == (5, 3, 3)
        assert_allclose(euler, 0.0, atol=0.1)

    def test_known_90deg_x_rotation(self) -> None:
        rot_6d = np.array([[
            [1, 0, 0, 0, 0, 1],
        ]], dtype=np.float64)
        euler = convert_6d_rotations_to_euler(rot_6d)
        assert_allclose(euler[0, 0, 0], 90.0, atol=1.0)

    def test_batch_shape(self) -> None:
        rot_6d = np.tile([1, 0, 0, 0, 1, 0], (30, 21, 1)).astype(np.float64)
        euler = convert_6d_rotations_to_euler(rot_6d)
        assert euler.shape == (30, 21, 3)


@pytest.mark.unit
class TestExtractRootTranslation:
    def test_zero_velocity_gives_zero_position(self) -> None:
        tensor = np.zeros((20, 263), dtype=np.float32)
        tensor[:, 3] = 1.0
        result = extract_root_translation(tensor)
        assert result.shape == (20, 3)
        assert_allclose(result[:, 0], 0.0, atol=1e-7)
        assert_allclose(result[:, 2], 0.0, atol=1e-7)
        assert_allclose(result[:, 1], 1.0, atol=1e-7)

    def test_constant_velocity_gives_linear(self) -> None:
        tensor = np.zeros((10, 263), dtype=np.float32)
        tensor[:, 1] = 0.1
        tensor[:, 3] = 0.0
        result = extract_root_translation(tensor)
        expected_x = np.cumsum(np.full(10, 0.1))
        assert_allclose(result[:, 0], expected_x, atol=1e-5)


@pytest.mark.unit
class TestExtractRootRotation:
    def test_zero_angular_velocity(self) -> None:
        tensor = np.zeros((20, 263), dtype=np.float32)
        result = extract_root_rotation(tensor)
        assert result.shape == (20, 3)
        assert_allclose(result, 0.0, atol=1e-7)

    def test_constant_angular_velocity(self) -> None:
        tensor = np.zeros((10, 263), dtype=np.float32)
        angular_vel = 0.1
        tensor[:, 0] = angular_vel
        result = extract_root_rotation(tensor)
        expected_y_deg = np.degrees(np.cumsum(np.full(10, angular_vel)))
        assert_allclose(result[:, 1], expected_y_deg, atol=1e-4)


@pytest.mark.unit
class TestResampleMotion:
    def test_same_fps_no_change(self) -> None:
        euler = np.random.default_rng(42).standard_normal((20, 21, 3))
        trans = np.random.default_rng(42).standard_normal((20, 3))
        root_rot = np.random.default_rng(42).standard_normal((20, 3))

        r_euler, r_trans, r_root = resample_motion(euler, trans, root_rot, 30, 30)

        assert r_euler.shape == euler.shape
        assert_allclose(r_euler, euler, atol=1e-10)

    def test_upsample_frame_count(self) -> None:
        num_source = 20
        euler = np.random.default_rng(42).standard_normal((num_source, 21, 3))
        trans = np.random.default_rng(42).standard_normal((num_source, 3))
        root_rot = np.random.default_rng(42).standard_normal((num_source, 3))

        r_euler, r_trans, r_root = resample_motion(euler, trans, root_rot, 20, 30)

        expected_frames = int((num_source - 1) / 20 * 30) + 1
        assert r_euler.shape[0] == expected_frames
        assert r_trans.shape[0] == expected_frames
        assert r_euler.shape[1:] == (21, 3)

    def test_endpoints_preserved(self) -> None:
        euler = np.random.default_rng(42).standard_normal((20, 21, 3))
        trans = np.random.default_rng(42).standard_normal((20, 3))
        root_rot = np.random.default_rng(42).standard_normal((20, 3))

        r_euler, r_trans, _ = resample_motion(euler, trans, root_rot, 20, 30)

        assert_allclose(r_euler[0], euler[0], atol=1e-6)
        assert_allclose(r_trans[0], trans[0], atol=1e-6)
        assert_allclose(r_euler[-1], euler[-1], atol=1e-6)
        assert_allclose(r_trans[-1], trans[-1], atol=1e-6)


@pytest.mark.unit
class TestRetargetSkeleton:
    def test_identity_pose_produces_curves(self) -> None:
        num_frames = 20
        euler = np.zeros((num_frames, 21, 3))
        root_rot = np.zeros((num_frames, 3))
        root_trans = np.zeros((num_frames, 3))
        duration = (num_frames - 1) / 20.0

        curves = retarget_to_vrm(euler, root_rot, root_trans, duration)

        assert len(curves) > 0

    def test_vrm_bone_names_present(self) -> None:
        num_frames = 10
        euler = np.zeros((num_frames, 21, 3))
        root_rot = np.zeros((num_frames, 3))
        root_trans = np.zeros((num_frames, 3))
        duration = (num_frames - 1) / 20.0

        curves = retarget_to_vrm(euler, root_rot, root_trans, duration)

        bone_names = {c.bone_name for c in curves}
        assert "hips" in bone_names
        assert "leftUpperLeg" in bone_names
        assert "rightUpperLeg" in bone_names
        assert "spine" in bone_names

    def test_curve_count(self) -> None:
        num_frames = 10
        euler = np.zeros((num_frames, 21, 3))
        root_rot = np.zeros((num_frames, 3))
        root_trans = np.zeros((num_frames, 3))
        duration = (num_frames - 1) / 20.0

        curves = retarget_to_vrm(euler, root_rot, root_trans, duration)

        root_curves = 6
        joint_curves = 21 * 3
        expected_total = root_curves + joint_curves
        assert len(curves) == expected_total

    def test_property_types_correct(self) -> None:
        euler = np.zeros((10, 21, 3))
        root_rot = np.zeros((10, 3))
        root_trans = np.zeros((10, 3))
        duration = 9.0 / 20.0

        curves = retarget_to_vrm(euler, root_rot, root_trans, duration)

        property_types = {c.property_type for c in curves}
        assert PROPERTY_TRANSLATION_X in property_types
        assert PROPERTY_TRANSLATION_Y in property_types
        assert PROPERTY_TRANSLATION_Z in property_types
        assert PROPERTY_ROTATION_X in property_types
        assert PROPERTY_ROTATION_Y in property_types
        assert PROPERTY_ROTATION_Z in property_types


@pytest.mark.unit
class TestEndToEnd:
    def test_full_pipeline(self) -> None:
        tensor = _make_synthetic_tensor(60)
        config = ConversionConfig(target_fps=30, keyframe_epsilon=0.5, bezier_max_error=0.01)

        curves = convert_humanml3d_to_curves(tensor, config)

        assert len(curves) > 0
        for curve in curves:
            assert len(curve.times) >= 2
            assert len(curve.values) == len(curve.times)
            assert len(curve.tangent_in) == len(curve.times)
            assert len(curve.tangent_out) == len(curve.times)

    def test_keyframe_time_range(self) -> None:
        tensor = _make_synthetic_tensor(40)
        config = ConversionConfig(target_fps=30)

        curves = convert_humanml3d_to_curves(tensor, config)

        for curve in curves:
            assert curve.times[0] >= 0.0
            assert curve.times[-1] <= 3.0

    def test_reduced_keyframe_count(self) -> None:
        tensor = _make_synthetic_tensor(60)
        config = ConversionConfig(target_fps=30, keyframe_epsilon=0.5)

        curves = convert_humanml3d_to_curves(tensor, config)

        for curve in curves:
            num_source_frames = 60
            assert len(curve.times) <= num_source_frames

    def test_custom_bone_mappings(self) -> None:
        tensor = _make_synthetic_tensor(20)
        config = ConversionConfig(target_fps=30)
        mappings = [(1, "leftUpperLeg"), (2, "rightUpperLeg")]

        curves = convert_humanml3d_to_curves(tensor, config, bone_mappings=mappings)

        bone_names = {c.bone_name for c in curves}
        assert "leftUpperLeg" in bone_names
        assert "rightUpperLeg" in bone_names
        assert "hips" in bone_names

    def test_invalid_bone_names_filtered(self) -> None:
        tensor = _make_synthetic_tensor(20)
        config = ConversionConfig(target_fps=30)
        mappings = [(1, "invalidBone"), (2, "leftUpperLeg")]

        curves = convert_humanml3d_to_curves(tensor, config, bone_mappings=mappings)

        bone_names = {c.bone_name for c in curves}
        assert "invalidBone" not in bone_names
        assert "leftUpperLeg" in bone_names


@pytest.mark.unit
class TestValidateMotionCurves:
    def test_filters_short_curves(self) -> None:
        curves = [
            MotionCurveData(
                bone_name="hips",
                property_type=PROPERTY_ROTATION_X,
                times=np.array([0.0]),
                values=np.array([1.0]),
                tangent_in=[(0.0, 0.0)],
                tangent_out=[(0.0, 0.0)],
            ),
            MotionCurveData(
                bone_name="hips",
                property_type=PROPERTY_ROTATION_Y,
                times=np.array([0.0, 1.0]),
                values=np.array([0.0, 10.0]),
                tangent_in=[(0.0, 0.0)] * 2,
                tangent_out=[(0.0, 0.0)] * 2,
            ),
        ]

        result = validate_motion_curves(curves, 1.0)
        assert len(result) == 1
        assert result[0].property_type == PROPERTY_ROTATION_Y

    def test_sanitizes_nan_values(self) -> None:
        curves = [
            MotionCurveData(
                bone_name="hips",
                property_type=PROPERTY_ROTATION_X,
                times=np.array([0.0, 1.0]),
                values=np.array([float("nan"), 10.0]),
                tangent_in=[(0.0, 0.0)] * 2,
                tangent_out=[(0.0, 0.0)] * 2,
            ),
        ]

        result = validate_motion_curves(curves, 1.0)
        assert len(result) == 1
        assert result[0].values[0] == 0.0
        assert result[0].values[1] == 10.0

    def test_clamps_time_to_duration(self) -> None:
        curves = [
            MotionCurveData(
                bone_name="spine",
                property_type=PROPERTY_ROTATION_X,
                times=np.array([-0.1, 0.5, 2.5]),
                values=np.array([0.0, 5.0, 10.0]),
                tangent_in=[(0.0, 0.0)] * 3,
                tangent_out=[(0.0, 0.0)] * 3,
            ),
        ]

        result = validate_motion_curves(curves, 2.0)
        assert result[0].times[0] == 0.0
        assert result[0].times[2] == 2.0

    def test_filters_invalid_bone_names(self) -> None:
        curves = [
            MotionCurveData(
                bone_name="invalidBone",
                property_type=PROPERTY_ROTATION_X,
                times=np.array([0.0, 1.0]),
                values=np.array([0.0, 10.0]),
                tangent_in=[(0.0, 0.0)] * 2,
                tangent_out=[(0.0, 0.0)] * 2,
            ),
        ]

        result = validate_motion_curves(curves, 1.0)
        assert len(result) == 0

    def test_sanitizes_nan_tangents(self) -> None:
        curves = [
            MotionCurveData(
                bone_name="hips",
                property_type=PROPERTY_ROTATION_X,
                times=np.array([0.0, 1.0]),
                values=np.array([0.0, 10.0]),
                tangent_in=[(float("nan"), 0.0), (0.0, float("inf"))],
                tangent_out=[(0.0, 0.0)] * 2,
            ),
        ]

        result = validate_motion_curves(curves, 1.0)
        assert result[0].tangent_in[0] == (0.0, 0.0)
        assert result[0].tangent_in[1] == (0.0, 0.0)
