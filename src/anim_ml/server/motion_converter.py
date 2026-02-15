from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

from anim_ml.utils.bezier_fitter import fit_bezier_segments
from anim_ml.utils.keyframe_reducer import reduce_keyframes
from anim_ml.utils.rotation import (
    matrix_to_quaternion,
    quaternion_to_euler_xyz,
    rotation_6d_to_matrix,
)
from anim_ml.utils.skeleton import SMPL_TO_VRM_MAPPING

PROPERTY_TRANSLATION_X = 0
PROPERTY_TRANSLATION_Y = 1
PROPERTY_TRANSLATION_Z = 2
PROPERTY_ROTATION_X = 3
PROPERTY_ROTATION_Y = 4
PROPERTY_ROTATION_Z = 5

HUMANML3D_ROT_6D_OFFSET = 130
HUMANML3D_NUM_JOINTS = 21
SOURCE_FPS = 20


@dataclass
class MotionCurveData:
    bone_name: str
    property_type: int
    times: np.ndarray
    values: np.ndarray
    tangent_in: list[tuple[float, float]]
    tangent_out: list[tuple[float, float]]


@dataclass
class ConversionConfig:
    target_fps: int = 30
    keyframe_epsilon: float = 0.5
    bezier_max_error: float = 0.01


def convert_humanml3d_to_curves(
    motion_tensor: np.ndarray,
    config: ConversionConfig,
    bone_mappings: list[tuple[int, str]] | None = None,
) -> list[MotionCurveData]:
    num_source_frames = motion_tensor.shape[0]
    duration = (num_source_frames - 1) / SOURCE_FPS if num_source_frames > 1 else 0.0

    rot_6d = extract_joint_rotations_6d(motion_tensor)
    euler_rotations = convert_6d_rotations_to_euler(rot_6d)

    root_translation = extract_root_translation(motion_tensor)
    root_rotation = extract_root_rotation(motion_tensor)

    euler_rotations, root_translation, root_rotation = resample_motion(
        euler_rotations, root_translation, root_rotation,
        SOURCE_FPS, config.target_fps,
    )

    curves = retarget_to_vrm(
        euler_rotations, root_rotation, root_translation,
        duration, bone_mappings,
    )

    fitted_curves = reduce_and_fit_curves(curves, config)
    return validate_motion_curves(fitted_curves, duration)


def extract_joint_rotations_6d(motion_tensor: np.ndarray) -> np.ndarray:
    num_frames = motion_tensor.shape[0]
    end_offset = HUMANML3D_ROT_6D_OFFSET + HUMANML3D_NUM_JOINTS * 6
    rot_6d_flat = motion_tensor[:, HUMANML3D_ROT_6D_OFFSET:end_offset]
    return rot_6d_flat.reshape(num_frames, HUMANML3D_NUM_JOINTS, 6)


def convert_6d_rotations_to_euler(rot_6d: np.ndarray) -> np.ndarray:
    rot_matrix = rotation_6d_to_matrix(rot_6d)
    quaternion = matrix_to_quaternion(rot_matrix)
    euler_rad = quaternion_to_euler_xyz(quaternion)
    return np.degrees(euler_rad).astype(np.float64)


def extract_root_translation(motion_tensor: np.ndarray) -> np.ndarray:
    num_frames = motion_tensor.shape[0]
    translation = np.zeros((num_frames, 3), dtype=np.float64)

    vx = motion_tensor[:, 1].astype(np.float64)
    vz = motion_tensor[:, 2].astype(np.float64)
    translation[:, 0] = np.cumsum(vx)
    translation[:, 1] = motion_tensor[:, 3].astype(np.float64)
    translation[:, 2] = np.cumsum(vz)

    return translation


def extract_root_rotation(motion_tensor: np.ndarray) -> np.ndarray:
    num_frames = motion_tensor.shape[0]
    root_euler = np.zeros((num_frames, 3), dtype=np.float64)

    angular_velocity_y = motion_tensor[:, 0].astype(np.float64)
    root_euler[:, 1] = np.degrees(np.cumsum(angular_velocity_y))

    root_orient = motion_tensor[:, 260:263].astype(np.float64)
    root_euler[:, 0] = np.degrees(root_orient[:, 0])
    root_euler[:, 2] = np.degrees(root_orient[:, 2])

    return root_euler


def resample_motion(
    euler_rotations: np.ndarray,
    root_translation: np.ndarray,
    root_rotation: np.ndarray,
    source_fps: int,
    target_fps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source_fps == target_fps:
        return euler_rotations, root_translation, root_rotation

    num_source_frames = euler_rotations.shape[0]
    duration = (num_source_frames - 1) / source_fps
    num_target_frames = max(2, int(duration * target_fps) + 1)

    source_times = np.linspace(  # pyright: ignore[reportUnknownVariableType]
        0.0, duration, num_source_frames,
    )
    target_times = np.linspace(  # pyright: ignore[reportUnknownVariableType]
        0.0, duration, num_target_frames,
    )

    resampled_rot = _resample_array(
        source_times, euler_rotations, target_times,  # pyright: ignore[reportUnknownArgumentType]
    )
    resampled_trans = _resample_array(
        source_times, root_translation, target_times,  # pyright: ignore[reportUnknownArgumentType]
    )
    resampled_root_rot = _resample_array(
        source_times, root_rotation, target_times,  # pyright: ignore[reportUnknownArgumentType]
    )

    return resampled_rot, resampled_trans, resampled_root_rot


def retarget_to_vrm(
    euler_rotations: np.ndarray,
    root_rotation: np.ndarray,
    root_translation: np.ndarray,
    duration: float,
    bone_mappings: list[tuple[int, str]] | None = None,
) -> list[MotionCurveData]:
    if bone_mappings is None:
        bone_mappings = [
            (smpl_idx, vrm_name)
            for smpl_idx, vrm_name in SMPL_TO_VRM_MAPPING.items()
            if smpl_idx > 0
        ]

    num_frames = euler_rotations.shape[0]
    times = np.linspace(0.0, duration, num_frames)

    curves: list[MotionCurveData] = []

    curves.extend(_decompose_translation_curves("hips", times, root_translation))
    curves.extend(_decompose_rotation_curves("hips", times, root_rotation))

    for smpl_idx, vrm_name in bone_mappings:
        joint_idx = smpl_idx - 1
        if joint_idx < 0 or joint_idx >= euler_rotations.shape[1]:
            continue

        joint_euler = _retarget_joint_rotation(euler_rotations[:, joint_idx])
        curves.extend(_decompose_rotation_curves(vrm_name, times, joint_euler))

    return curves


def reduce_and_fit_curves(
    curves: list[MotionCurveData],
    config: ConversionConfig,
) -> list[MotionCurveData]:
    result: list[MotionCurveData] = []

    for curve in curves:
        reduced = reduce_keyframes(curve.times, curve.values, config.keyframe_epsilon)

        if len(reduced) < 2:
            result.append(curve)
            continue

        bezier = fit_bezier_segments(
            reduced, curve.times, curve.values, config.bezier_max_error,
        )

        times = np.array([kf.time for kf in bezier])
        values = np.array([kf.value for kf in bezier])
        tangent_in = [kf.tangent_in for kf in bezier]
        tangent_out = [kf.tangent_out for kf in bezier]

        result.append(MotionCurveData(
            bone_name=curve.bone_name,
            property_type=curve.property_type,
            times=times,
            values=values,
            tangent_in=tangent_in,
            tangent_out=tangent_out,
        ))

    return result


def validate_motion_curves(
    curves: list[MotionCurveData],
    duration: float,
) -> list[MotionCurveData]:
    valid_vrm_names = set(SMPL_TO_VRM_MAPPING.values())
    validated: list[MotionCurveData] = []

    for curve in curves:
        if curve.bone_name not in valid_vrm_names:
            continue

        if len(curve.times) < 2:
            continue

        nan_mask = np.isnan(curve.values) | np.isinf(curve.values)
        if np.any(nan_mask):
            curve.values = np.where(nan_mask, 0.0, curve.values)

        curve.tangent_in = [_sanitize_tangent(t) for t in curve.tangent_in]
        curve.tangent_out = [_sanitize_tangent(t) for t in curve.tangent_out]

        curve.times = np.clip(curve.times, 0.0, max(duration, 0.0))
        validated.append(curve)

    return validated


def _sanitize_tangent(tangent: tuple[float, float]) -> tuple[float, float]:
    dt, dv = tangent
    if not np.isfinite(dt):
        dt = 0.0
    if not np.isfinite(dv):
        dv = 0.0
    return (dt, dv)


def _retarget_joint_rotation(euler_degrees: np.ndarray) -> np.ndarray:
    return euler_degrees


def _decompose_translation_curves(
    bone_name: str,
    times: np.ndarray,
    translation: np.ndarray,
) -> list[MotionCurveData]:
    property_types = [PROPERTY_TRANSLATION_X, PROPERTY_TRANSLATION_Y, PROPERTY_TRANSLATION_Z]
    return [
        MotionCurveData(
            bone_name=bone_name,
            property_type=prop,
            times=times.copy(),
            values=translation[:, axis].copy(),
            tangent_in=[(0.0, 0.0)] * len(times),
            tangent_out=[(0.0, 0.0)] * len(times),
        )
        for axis, prop in enumerate(property_types)
    ]


def _decompose_rotation_curves(
    bone_name: str,
    times: np.ndarray,
    euler_degrees: np.ndarray,
) -> list[MotionCurveData]:
    property_types = [PROPERTY_ROTATION_X, PROPERTY_ROTATION_Y, PROPERTY_ROTATION_Z]
    return [
        MotionCurveData(
            bone_name=bone_name,
            property_type=prop,
            times=times.copy(),
            values=euler_degrees[:, axis].copy(),
            tangent_in=[(0.0, 0.0)] * len(times),
            tangent_out=[(0.0, 0.0)] * len(times),
        )
        for axis, prop in enumerate(property_types)
    ]


def _resample_array(
    source_times: np.ndarray,
    data: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    original_shape = data.shape[1:]
    flat = data.reshape(len(source_times), -1)

    resampled_flat = np.empty((len(target_times), flat.shape[1]), dtype=np.float64)
    for col in range(flat.shape[1]):
        cs = CubicSpline(source_times, flat[:, col])
        resampled_flat[:, col] = cs(target_times)

    return resampled_flat.reshape(len(target_times), *original_shape)
