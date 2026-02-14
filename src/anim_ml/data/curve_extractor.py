from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d  # type: ignore[import-untyped]

from anim_ml.utils.bezier_fitter import BezierKeyframe, fit_bezier_segments
from anim_ml.utils.keyframe_reducer import reduce_keyframes
from anim_ml.utils.skeleton import classify_joint

if TYPE_CHECKING:
    from anim_ml.data.bvh_parser import MotionData


@dataclass
class CurveSample:
    context_keyframes: np.ndarray
    target_keyframe: np.ndarray
    property_type: int
    joint_category: int
    query_time: float
    clip_duration: float
    joint_depth: int
    curve_mean: float
    curve_std: float


CONTEXT_LENGTH = 8
TARGET_FPS = 30.0

POSITION_EPSILON = 0.01
ROTATION_EPSILON = 0.5
BEZIER_MAX_ERROR = 0.005


def extract_curve_samples(
    motion: MotionData,
    parent_indices: list[int] | None = None,
    joint_names: list[str] | None = None,
) -> list[CurveSample]:
    if parent_indices is None:
        parent_indices = motion.parent_indices
    if joint_names is None:
        joint_names = motion.joint_names

    num_frames = motion.positions.shape[0]
    if num_frames < 3:
        return []

    original_fps = 1.0 / motion.frame_time if motion.frame_time > 0 else TARGET_FPS
    duration = (num_frames - 1) * motion.frame_time

    positions, rotations, times = _resample_to_target_fps(
        motion.positions, motion.rotations, original_fps, duration,
    )

    skeleton_height = _estimate_skeleton_height(motion)

    samples: list[CurveSample] = []
    num_joints = positions.shape[1]

    for joint_idx in range(num_joints):
        joint_name = joint_names[joint_idx] if joint_idx < len(joint_names) else "unknown"
        category = int(classify_joint(joint_name))
        depth = _compute_joint_depth(parent_indices, joint_idx)

        for channel_idx in range(6):
            values = _get_channel_values(positions, rotations, joint_idx, channel_idx)
            is_position = channel_idx < 3

            epsilon = POSITION_EPSILON if is_position else ROTATION_EPSILON
            scale = skeleton_height if is_position else 180.0

            if scale < 1e-6:
                continue

            normalized_values = values / scale

            keyframes = reduce_keyframes(times, normalized_values, epsilon / scale)
            if len(keyframes) < 2:
                continue

            bezier_keyframes = fit_bezier_segments(
                keyframes, times, normalized_values, BEZIER_MAX_ERROR,
            )

            channel_samples = _generate_sliding_window_samples(
                bezier_keyframes,
                property_type=channel_idx,
                joint_category=category,
                clip_duration=duration,
                joint_depth=depth,
                scale=scale,
            )
            samples.extend(channel_samples)

    return samples


def _resample_to_target_fps(
    positions: np.ndarray,
    rotations: np.ndarray,
    original_fps: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(original_fps - TARGET_FPS) < 0.5:
        num_frames = positions.shape[0]
        times = np.linspace(0.0, duration, num_frames)
        return positions, rotations, times

    original_frames = positions.shape[0]
    original_times = np.linspace(0.0, duration, original_frames)

    target_frames = max(int(round(duration * TARGET_FPS)) + 1, 2)
    target_times = np.linspace(0.0, duration, target_frames)

    num_joints = positions.shape[1]
    new_positions = np.zeros((target_frames, num_joints, 3))
    new_rotations = np.zeros((target_frames, num_joints, 3))

    for j in range(num_joints):
        for axis in range(3):
            f_pos = interp1d(original_times, positions[:, j, axis], kind="linear")
            new_positions[:, j, axis] = f_pos(target_times)

            f_rot = interp1d(original_times, rotations[:, j, axis], kind="linear")
            new_rotations[:, j, axis] = f_rot(target_times)

    return new_positions, new_rotations, target_times


def _estimate_skeleton_height(motion: MotionData) -> float:
    root_y = motion.positions[:, 0, 1]
    height = float(np.median(np.abs(root_y)))
    return max(height, 1.0)


def _compute_joint_depth(parent_indices: list[int], joint_idx: int) -> int:
    depth = 0
    current = joint_idx
    while parent_indices[current] != -1:
        depth += 1
        current = parent_indices[current]
    return depth


def _get_channel_values(
    positions: np.ndarray, rotations: np.ndarray, joint_idx: int, channel_idx: int,
) -> np.ndarray:
    if channel_idx < 3:
        return positions[:, joint_idx, channel_idx].copy()
    return rotations[:, joint_idx, channel_idx - 3].copy()


def _generate_sliding_window_samples(
    bezier_keyframes: list[BezierKeyframe],
    property_type: int,
    joint_category: int,
    clip_duration: float,
    joint_depth: int,
    scale: float,
) -> list[CurveSample]:
    if len(bezier_keyframes) < 2:
        return []

    time_scale = clip_duration if clip_duration > 0 else 1.0
    samples: list[CurveSample] = []

    for target_idx in range(1, len(bezier_keyframes)):
        context_start = max(0, target_idx - CONTEXT_LENGTH)
        context_kfs = bezier_keyframes[context_start:target_idx]

        context_values = [kf.value for kf in context_kfs]
        curve_mean = float(np.mean(context_values))
        curve_std = float(max(np.std(context_values), 1e-6))

        context_array = np.zeros((CONTEXT_LENGTH, 6), dtype=np.float32)
        offset = CONTEXT_LENGTH - len(context_kfs)

        for i, kf in enumerate(context_kfs):
            row = offset + i
            context_array[row, 0] = kf.time / time_scale
            context_array[row, 1] = (kf.value - curve_mean) / curve_std
            context_array[row, 2] = kf.tangent_in[0] / time_scale
            context_array[row, 3] = kf.tangent_in[1] / curve_std
            context_array[row, 4] = kf.tangent_out[0] / time_scale
            context_array[row, 5] = kf.tangent_out[1] / curve_std

        target_kf = bezier_keyframes[target_idx]
        target_array = np.array([
            target_kf.time / time_scale,
            (target_kf.value - curve_mean) / curve_std,
            target_kf.tangent_in[0] / time_scale,
            target_kf.tangent_in[1] / curve_std,
            target_kf.tangent_out[0] / time_scale,
            target_kf.tangent_out[1] / curve_std,
        ], dtype=np.float32)

        samples.append(CurveSample(
            context_keyframes=context_array,
            target_keyframe=target_array,
            property_type=property_type,
            joint_category=joint_category,
            query_time=float(target_kf.time / time_scale),
            clip_duration=clip_duration,
            joint_depth=joint_depth,
            curve_mean=curve_mean * scale,
            curve_std=curve_std * scale,
        ))

    return samples
