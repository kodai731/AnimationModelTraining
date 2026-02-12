from __future__ import annotations

import numpy as np


def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]

    b1 = _normalize_vectors(a1)
    b2 = _normalize_vectors(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3).astype(np.float64)
    n = m.shape[0]

    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    quat = np.zeros((n, 4), dtype=np.float64)

    _fill_trace_positive(m, trace, quat)
    _fill_diag0_largest(m, trace, quat)
    _fill_diag1_largest(m, trace, quat)
    _fill_diag2_largest(m, trace, quat)

    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.where(norms > 0, norms, 1.0)

    return quat.reshape(*batch_shape, 4)


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = quat.astype(np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    matrix = np.empty((*q.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y2 + z2)
    matrix[..., 0, 1] = 2.0 * (xy - wz)
    matrix[..., 0, 2] = 2.0 * (xz + wy)
    matrix[..., 1, 0] = 2.0 * (xy + wz)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x2 + z2)
    matrix[..., 1, 2] = 2.0 * (yz - wx)
    matrix[..., 2, 0] = 2.0 * (xz - wy)
    matrix[..., 2, 1] = 2.0 * (yz + wx)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x2 + y2)

    return matrix


def quaternion_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    m = quaternion_to_matrix(quat)

    sy = np.clip(m[..., 0, 2], -1.0, 1.0)
    ry = np.arcsin(sy)

    cos_ry = np.cos(ry)
    not_gimbal = np.abs(cos_ry) > 1e-6

    rx_normal = np.arctan2(-m[..., 1, 2], m[..., 2, 2])
    rx_gimbal = np.arctan2(m[..., 2, 1], m[..., 1, 1])
    rx = np.where(not_gimbal, rx_normal, rx_gimbal)

    rz_normal = np.arctan2(-m[..., 0, 1], m[..., 0, 0])
    rz = np.where(not_gimbal, rz_normal, np.zeros_like(ry))

    return np.stack([rx, ry, rz], axis=-1)


def euler_xyz_to_quaternion(euler: np.ndarray) -> np.ndarray:
    half = euler.astype(np.float64) * 0.5
    cx, cy, cz = np.cos(half[..., 0]), np.cos(half[..., 1]), np.cos(half[..., 2])
    sx, sy, sz = np.sin(half[..., 0]), np.sin(half[..., 1]), np.sin(half[..., 2])

    w = cx * cy * cz - sx * sy * sz
    x = sx * cy * cz + cx * sy * sz
    y = cx * sy * cz - sx * cy * sz
    z = cx * cy * sz + sx * sy * cz

    return np.stack([x, y, z, w], axis=-1)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([x, y, z, w], axis=-1)


def quaternion_inverse(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[..., :3] *= -1.0
    return result


def quaternion_geodesic_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    dot = np.sum(q1 * q2, axis=-1)
    return 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))


def _normalize_vectors(v: np.ndarray) -> np.ndarray:
    norm: np.ndarray = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return v / norm  # type: ignore[no-any-return]


def _fill_trace_positive(m: np.ndarray, trace: np.ndarray, quat: np.ndarray) -> None:
    mask = trace > 0
    if not np.any(mask):
        return
    s = np.sqrt(trace[mask] + 1.0) * 2.0
    quat[mask, 3] = 0.25 * s
    quat[mask, 0] = (m[mask, 2, 1] - m[mask, 1, 2]) / s
    quat[mask, 1] = (m[mask, 0, 2] - m[mask, 2, 0]) / s
    quat[mask, 2] = (m[mask, 1, 0] - m[mask, 0, 1]) / s


def _fill_diag0_largest(m: np.ndarray, trace: np.ndarray, quat: np.ndarray) -> None:
    mask = (trace <= 0) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    if not np.any(mask):
        return
    s = np.sqrt(1.0 + m[mask, 0, 0] - m[mask, 1, 1] - m[mask, 2, 2]) * 2.0
    quat[mask, 3] = (m[mask, 2, 1] - m[mask, 1, 2]) / s
    quat[mask, 0] = 0.25 * s
    quat[mask, 1] = (m[mask, 0, 1] + m[mask, 1, 0]) / s
    quat[mask, 2] = (m[mask, 0, 2] + m[mask, 2, 0]) / s


def _fill_diag1_largest(m: np.ndarray, trace: np.ndarray, quat: np.ndarray) -> None:
    not_diag0 = ~((m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2]))
    mask = (trace <= 0) & not_diag0 & (m[:, 1, 1] > m[:, 2, 2])
    if not np.any(mask):
        return
    s = np.sqrt(1.0 + m[mask, 1, 1] - m[mask, 0, 0] - m[mask, 2, 2]) * 2.0
    quat[mask, 3] = (m[mask, 0, 2] - m[mask, 2, 0]) / s
    quat[mask, 0] = (m[mask, 0, 1] + m[mask, 1, 0]) / s
    quat[mask, 1] = 0.25 * s
    quat[mask, 2] = (m[mask, 1, 2] + m[mask, 2, 1]) / s


def _fill_diag2_largest(m: np.ndarray, trace: np.ndarray, quat: np.ndarray) -> None:
    not_diag0 = ~((m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2]))
    mask = (trace <= 0) & not_diag0 & ~(m[:, 1, 1] > m[:, 2, 2])
    if not np.any(mask):
        return
    s = np.sqrt(1.0 + m[mask, 2, 2] - m[mask, 0, 0] - m[mask, 1, 1]) * 2.0
    quat[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s
    quat[mask, 0] = (m[mask, 0, 2] + m[mask, 2, 0]) / s
    quat[mask, 1] = (m[mask, 1, 2] + m[mask, 2, 1]) / s
    quat[mask, 2] = 0.25 * s
