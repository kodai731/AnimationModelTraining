import numpy as np
import pytest
from numpy.testing import assert_allclose

from anim_ml.utils.rotation import (
    euler_to_matrix,
    euler_xyz_to_quaternion,
    matrix_to_quaternion,
    quaternion_geodesic_distance,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_euler_xyz,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])
IDENTITY_MATRIX = np.eye(3)


@pytest.mark.unit
class TestRotation6dToMatrix:
    def test_identity(self) -> None:
        rot_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)
        result = rotation_6d_to_matrix(rot_6d)
        assert_allclose(result, IDENTITY_MATRIX, atol=1e-7)

    def test_batch(self) -> None:
        rot_6d = np.tile([1, 0, 0, 0, 1, 0], (5, 1)).astype(np.float64)
        result = rotation_6d_to_matrix(rot_6d)
        assert result.shape == (5, 3, 3)
        for i in range(5):
            assert_allclose(result[i], IDENTITY_MATRIX, atol=1e-7)

    def test_orthogonality(self) -> None:
        rng = np.random.default_rng(42)
        rot_6d = rng.standard_normal((10, 6))
        matrices = rotation_6d_to_matrix(rot_6d)
        for m in matrices:
            assert_allclose(m @ m.T, IDENTITY_MATRIX, atol=1e-6)
            assert_allclose(np.linalg.det(m), 1.0, atol=1e-6)


@pytest.mark.unit
class TestMatrixQuaternionRoundtrip:
    def test_identity(self) -> None:
        quat = matrix_to_quaternion(IDENTITY_MATRIX)
        assert_allclose(np.abs(quat), [0, 0, 0, 1], atol=1e-7)

    def test_known_90deg_rotations(self) -> None:
        rx90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        q = matrix_to_quaternion(rx90)
        expected_angle = np.pi / 2
        expected_q = np.array([np.sin(expected_angle / 2), 0, 0, np.cos(expected_angle / 2)])
        assert_allclose(np.abs(np.dot(q, expected_q)), 1.0, atol=1e-6)

    def test_roundtrip(self) -> None:
        rng = np.random.default_rng(42)
        euler = rng.uniform(-np.pi, np.pi, (20, 3))
        quat_original = euler_xyz_to_quaternion(euler)

        matrices = quaternion_to_matrix(quat_original)
        quat_recovered = matrix_to_quaternion(matrices)

        dot = np.abs(np.sum(quat_original * quat_recovered, axis=-1))
        assert_allclose(dot, 1.0, atol=1e-6)

    def test_batch_shape(self) -> None:
        matrices = np.tile(IDENTITY_MATRIX, (3, 4, 1, 1))
        quat = matrix_to_quaternion(matrices)
        assert quat.shape == (3, 4, 4)


@pytest.mark.unit
class TestEulerQuaternionRoundtrip:
    def test_identity(self) -> None:
        euler = np.array([0.0, 0.0, 0.0])
        q = euler_xyz_to_quaternion(euler)
        assert_allclose(q, IDENTITY_QUAT, atol=1e-7)

    def test_90deg_x(self) -> None:
        euler = np.array([np.pi / 2, 0.0, 0.0])
        q = euler_xyz_to_quaternion(euler)
        euler_back = quaternion_to_euler_xyz(q)
        assert_allclose(euler_back, euler, atol=1e-6)

    def test_90deg_y(self) -> None:
        euler = np.array([0.0, np.pi / 2, 0.0])
        q = euler_xyz_to_quaternion(euler)
        euler_back = quaternion_to_euler_xyz(q)
        assert_allclose(euler_back, euler, atol=1e-6)

    def test_90deg_z(self) -> None:
        euler = np.array([0.0, 0.0, np.pi / 2])
        q = euler_xyz_to_quaternion(euler)
        euler_back = quaternion_to_euler_xyz(q)
        assert_allclose(euler_back, euler, atol=1e-6)

    def test_roundtrip_batch(self) -> None:
        rng = np.random.default_rng(123)
        rx = rng.uniform(-np.pi, np.pi, 100)
        ry = rng.uniform(-1.4, 1.4, 100)
        rz = rng.uniform(-np.pi, np.pi, 100)
        euler_original = np.stack([rx, ry, rz], axis=-1)

        quat = euler_xyz_to_quaternion(euler_original)
        euler_recovered = quaternion_to_euler_xyz(quat)
        assert_allclose(euler_recovered, euler_original, atol=1e-5)


@pytest.mark.unit
class TestQuaternionMultiply:
    def test_identity_multiply(self) -> None:
        q = np.array([0.5, 0.5, 0.5, 0.5])
        result = quaternion_multiply(q, IDENTITY_QUAT)
        assert_allclose(result, q, atol=1e-7)

    def test_inverse_gives_identity(self) -> None:
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q_inv = quaternion_inverse(q)
        result = quaternion_multiply(q, q_inv)
        assert_allclose(result, IDENTITY_QUAT, atol=1e-6)

    def test_batch(self) -> None:
        rng = np.random.default_rng(77)
        q1 = rng.standard_normal((100, 4))
        q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
        result = quaternion_multiply(q1, IDENTITY_QUAT)
        assert result.shape == (100, 4)
        assert_allclose(result, q1, atol=1e-6)


@pytest.mark.unit
class TestGeodesicDistance:
    def test_same_quaternion(self) -> None:
        q = np.array([0.5, 0.5, 0.5, 0.5])
        assert_allclose(quaternion_geodesic_distance(q, q), 0.0, atol=1e-7)

    def test_opposite_sign_same_rotation(self) -> None:
        q = np.array([0.5, 0.5, 0.5, 0.5])
        assert_allclose(quaternion_geodesic_distance(q, -q), 0.0, atol=1e-7)

    def test_90deg_distance(self) -> None:
        q1 = IDENTITY_QUAT
        q2 = euler_xyz_to_quaternion(np.array([np.pi / 2, 0.0, 0.0]))
        dist = quaternion_geodesic_distance(q1, q2)
        assert_allclose(dist, np.pi / 2, atol=1e-6)

    def test_batch(self) -> None:
        q1 = np.tile(IDENTITY_QUAT, (5, 1))
        q2 = np.tile(IDENTITY_QUAT, (5, 1))
        dist = quaternion_geodesic_distance(q1, q2)
        assert dist.shape == (5,)
        assert_allclose(dist, 0.0, atol=1e-7)


@pytest.mark.unit
class TestEulerToMatrix:
    def test_identity(self) -> None:
        angles = np.array([0.0, 0.0, 0.0])
        result = euler_to_matrix(angles, "XYZ")
        assert_allclose(result, IDENTITY_MATRIX, atol=1e-7)

    def test_xyz_matches_quaternion_path(self) -> None:
        rng = np.random.default_rng(55)
        euler = rng.uniform(-np.pi, np.pi, (20, 3))

        mat_direct = euler_to_matrix(euler, "XYZ")
        quat = euler_xyz_to_quaternion(euler)
        mat_via_quat = quaternion_to_matrix(quat)

        assert_allclose(mat_direct, mat_via_quat, atol=1e-6)

    def test_zxy_90deg_x(self) -> None:
        angles = np.array([0.0, np.pi / 2, 0.0])
        mat = euler_to_matrix(angles, "ZXY")
        expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        assert_allclose(mat, expected, atol=1e-7)

    def test_all_orders_identity(self) -> None:
        angles = np.array([0.0, 0.0, 0.0])
        for order in ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]:
            result = euler_to_matrix(angles, order)
            assert_allclose(result, IDENTITY_MATRIX, atol=1e-7)

    def test_batch_shape(self) -> None:
        angles = np.zeros((3, 5, 3))
        result = euler_to_matrix(angles, "ZXY")
        assert result.shape == (3, 5, 3, 3)

    def test_invalid_order_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid Euler order"):
            euler_to_matrix(np.zeros(3), "ABC")

    def test_single_axis_rotation(self) -> None:
        angle = np.pi / 4
        mat = euler_to_matrix(np.array([angle, 0.0, 0.0]), "XYZ")
        c, s = np.cos(angle), np.sin(angle)
        expected = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        assert_allclose(mat, expected, atol=1e-7)


@pytest.mark.unit
class TestFullRoundtrip:
    def test_6d_matrix_quat_euler_quat_matrix_6d(self) -> None:
        rng = np.random.default_rng(99)
        rot_6d_raw = rng.standard_normal((10, 6))

        matrices_1 = rotation_6d_to_matrix(rot_6d_raw)
        quat_1 = matrix_to_quaternion(matrices_1)
        euler = quaternion_to_euler_xyz(quat_1)
        quat_2 = euler_xyz_to_quaternion(euler)

        dot = np.abs(np.sum(quat_1 * quat_2, axis=-1))
        assert_allclose(dot, 1.0, atol=1e-5)
