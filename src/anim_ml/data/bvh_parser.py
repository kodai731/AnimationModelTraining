from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from anim_ml.utils.rotation import (
    euler_to_matrix,
    matrix_to_quaternion,
    quaternion_to_euler_xyz,
)


@dataclass
class MotionData:
    joint_names: list[str]
    parent_indices: list[int]
    frame_time: float
    positions: np.ndarray
    rotations: np.ndarray


def parse_bvh(filepath: Path, z_up: bool = False) -> MotionData:
    text = filepath.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    joint_names: list[str] = []
    parent_indices: list[int] = []
    channel_defs: list[list[str]] = []
    parent_stack: list[int] = [-1]

    idx = 0
    idx = _parse_hierarchy(lines, idx, joint_names, parent_indices, channel_defs, parent_stack)

    frame_count, frame_time, raw_data = _parse_motion(lines, idx)

    positions, rotations = _extract_channels(
        raw_data, frame_count, len(joint_names), channel_defs,
    )

    rotations = _normalize_euler_order(rotations, channel_defs, joint_names)

    if z_up:
        positions, rotations = _convert_z_up_to_y_up(positions, rotations)

    return MotionData(
        joint_names=joint_names,
        parent_indices=parent_indices,
        frame_time=frame_time,
        positions=positions,
        rotations=rotations,
    )


def _parse_hierarchy(
    lines: list[str],
    idx: int,
    joint_names: list[str],
    parent_indices: list[int],
    channel_defs: list[list[str]],
    parent_stack: list[int],
) -> int:
    while idx < len(lines):
        line = lines[idx]

        if line.startswith("MOTION"):
            return idx

        if line.startswith("ROOT") or line.startswith("JOINT"):
            name = line.split()[-1]
            joint_idx = len(joint_names)
            joint_names.append(name)
            parent_indices.append(parent_stack[-1])
            parent_stack.append(joint_idx)
            idx += 1
            continue

        if line.startswith("End Site"):
            depth = 0
            idx += 1
            while idx < len(lines):
                if "{" in lines[idx]:
                    depth += 1
                elif "}" in lines[idx]:
                    depth -= 1
                    if depth <= 0:
                        idx += 1
                        break
                idx += 1
            continue

        if line.startswith("CHANNELS"):
            parts = line.split()
            channel_count = int(parts[1])
            channels = parts[2 : 2 + channel_count]
            channel_defs.append(channels)
            idx += 1
            continue

        if line == "}":
            parent_stack.pop()
            idx += 1
            continue

        idx += 1

    return idx


def _parse_motion(
    lines: list[str], idx: int,
) -> tuple[int, float, np.ndarray]:
    frame_count = 0
    frame_time = 0.0

    while idx < len(lines):
        line = lines[idx]
        if line.startswith("Frames:"):
            frame_count = int(line.split(":")[1].strip())
        elif line.startswith("Frame Time:"):
            frame_time = float(line.split(":")[1].strip())
            idx += 1
            break
        idx += 1

    rows: list[list[float]] = []
    while idx < len(lines) and len(rows) < frame_count:
        values = [float(v) for v in lines[idx].split()]
        rows.append(values)
        idx += 1

    return frame_count, frame_time, np.array(rows, dtype=np.float64)


def _extract_channels(
    raw_data: np.ndarray,
    frame_count: int,
    num_joints: int,
    channel_defs: list[list[str]],
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.zeros((frame_count, num_joints, 3), dtype=np.float64)
    rotations = np.zeros((frame_count, num_joints, 3), dtype=np.float64)

    col = 0
    for joint_idx, channels in enumerate(channel_defs):
        for ch in channels:
            ch_upper = ch.upper()
            if ch_upper == "XPOSITION":
                positions[:, joint_idx, 0] = raw_data[:, col]
            elif ch_upper == "YPOSITION":
                positions[:, joint_idx, 1] = raw_data[:, col]
            elif ch_upper == "ZPOSITION":
                positions[:, joint_idx, 2] = raw_data[:, col]
            elif ch_upper == "XROTATION":
                rotations[:, joint_idx, 0] = raw_data[:, col]
            elif ch_upper == "YROTATION":
                rotations[:, joint_idx, 1] = raw_data[:, col]
            elif ch_upper == "ZROTATION":
                rotations[:, joint_idx, 2] = raw_data[:, col]
            col += 1

    return positions, rotations


def _get_euler_order(channels: list[str]) -> str:
    rotation_channels = [ch for ch in channels if ch.upper().endswith("ROTATION")]
    if len(rotation_channels) != 3:
        return "XYZ"

    order = ""
    for ch in rotation_channels:
        order += ch[0].upper()
    return order


def _normalize_euler_order(
    rotations: np.ndarray,
    channel_defs: list[list[str]],
    joint_names: list[str],
) -> np.ndarray:
    num_frames = rotations.shape[0]
    normalized = np.zeros_like(rotations)

    for joint_idx in range(len(joint_names)):
        order = _get_euler_order(channel_defs[joint_idx])

        if order == "XYZ":
            normalized[:, joint_idx] = rotations[:, joint_idx]
            continue

        angles_rad = np.deg2rad(rotations[:, joint_idx])
        reordered = _reorder_angles_for_euler(angles_rad, order, channel_defs[joint_idx])
        matrices = euler_to_matrix(reordered, order)
        quats = matrix_to_quaternion(matrices.reshape(num_frames, 3, 3))
        euler_xyz = quaternion_to_euler_xyz(quats)
        normalized[:, joint_idx] = np.rad2deg(euler_xyz)

    return normalized


def _reorder_angles_for_euler(
    angles_rad: np.ndarray,
    order: str,
    channels: list[str],
) -> np.ndarray:
    rotation_channels = [ch for ch in channels if ch.upper().endswith("ROTATION")]
    axis_map = {"X": 0, "Y": 1, "Z": 2}

    result = np.zeros_like(angles_rad)
    for i, ch in enumerate(rotation_channels):
        src_axis = axis_map[ch[0].upper()]
        result[:, i] = angles_rad[:, src_axis]

    return result


def _convert_z_up_to_y_up(
    positions: np.ndarray, rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    new_positions = np.empty_like(positions)
    new_positions[..., 0] = positions[..., 0]
    new_positions[..., 1] = positions[..., 2]
    new_positions[..., 2] = -positions[..., 1]

    correction_angle = np.array([-np.pi / 2, 0.0, 0.0])
    correction_mat = euler_to_matrix(correction_angle, "XYZ")

    root_rots_rad = np.deg2rad(rotations[:, 0])
    root_mats = euler_to_matrix(root_rots_rad, "XYZ")
    corrected_mats = correction_mat @ root_mats
    corrected_quats = matrix_to_quaternion(corrected_mats)
    corrected_euler = quaternion_to_euler_xyz(corrected_quats)

    new_rotations = rotations.copy()
    new_rotations[:, 0] = np.rad2deg(corrected_euler)

    return new_positions, new_rotations
