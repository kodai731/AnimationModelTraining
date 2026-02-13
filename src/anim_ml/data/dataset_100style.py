from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from anim_ml.data.bvh_parser import MotionData

STYLE100_TO_SMPL: dict[str, int] = {
    "Hips": 0,
    "LeftUpLeg": 1,
    "RightUpLeg": 2,
    "Spine": 3,
    "LeftLeg": 4,
    "RightLeg": 5,
    "Spine1": 6,
    "LeftFoot": 7,
    "RightFoot": 8,
    "Spine2": 9,
    "LeftToeBase": 10,
    "RightToeBase": 11,
    "Neck": 12,
    "LeftShoulder": 13,
    "RightShoulder": 14,
    "Head": 15,
    "LeftArm": 16,
    "RightArm": 17,
    "LeftForeArm": 18,
    "RightForeArm": 19,
    "LeftHand": 20,
    "RightHand": 21,
}

SMPL_22_JOINT_NAMES: list[str] = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]

SMPL_22_PARENT_INDICES: list[int] = [
    -1, 0, 0, 0,
    1, 2, 3,
    4, 5, 6,
    7, 8, 9, 9, 9,
    12, 13, 14,
    16, 17,
    18, 19,
]


def map_100style_to_smpl(motion: MotionData) -> MotionData | None:
    src_to_smpl: dict[int, int] = {}
    assigned_smpl: set[int] = set()

    for src_idx, name in enumerate(motion.joint_names):
        if name in STYLE100_TO_SMPL:
            smpl_idx = STYLE100_TO_SMPL[name]
            if smpl_idx not in assigned_smpl:
                src_to_smpl[src_idx] = smpl_idx
                assigned_smpl.add(smpl_idx)

    if len(assigned_smpl) < 15:
        return None

    num_frames = motion.positions.shape[0]
    new_positions = np.zeros((num_frames, 22, 3), dtype=np.float64)
    new_rotations = np.zeros((num_frames, 22, 3), dtype=np.float64)

    for src_idx, smpl_idx in src_to_smpl.items():
        if src_idx < motion.positions.shape[1]:
            new_positions[:, smpl_idx] = motion.positions[:, src_idx]
            new_rotations[:, smpl_idx] = motion.rotations[:, src_idx]

    from anim_ml.data.bvh_parser import MotionData as MD
    return MD(
        joint_names=SMPL_22_JOINT_NAMES,
        parent_indices=SMPL_22_PARENT_INDICES,
        frame_time=motion.frame_time,
        positions=new_positions,
        rotations=new_rotations,
    )


def find_100style_bvh_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*.bvh"))
