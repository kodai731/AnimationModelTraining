from __future__ import annotations

from enum import IntEnum

import numpy as np

from anim_ml.utils.rotation import quaternion_multiply  # noqa: TC001

SMPL_22_JOINT_NAMES: list[str] = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
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

SMPL_TO_VRM_MAPPING: dict[int, str] = {
    0: "hips",
    1: "leftUpperLeg",
    2: "rightUpperLeg",
    3: "spine",
    4: "leftLowerLeg",
    5: "rightLowerLeg",
    6: "chest",
    7: "leftFoot",
    8: "rightFoot",
    9: "upperChest",
    10: "leftToes",
    11: "rightToes",
    12: "neck",
    13: "leftShoulder",
    14: "rightShoulder",
    15: "head",
    16: "leftUpperArm",
    17: "rightUpperArm",
    18: "leftLowerArm",
    19: "rightLowerArm",
    20: "leftHand",
    21: "rightHand",
}


class JointCategory(IntEnum):
    ROOT = 0
    SPINE = 1
    ARM = 2
    LEG = 3
    HEAD = 4
    HAND = 5
    FOOT = 6


_JOINT_CATEGORY_MAP: dict[str, JointCategory] = {
    "pelvis": JointCategory.ROOT,
    "spine1": JointCategory.SPINE,
    "spine2": JointCategory.SPINE,
    "spine3": JointCategory.SPINE,
    "left_hip": JointCategory.LEG,
    "right_hip": JointCategory.LEG,
    "left_knee": JointCategory.LEG,
    "right_knee": JointCategory.LEG,
    "left_collar": JointCategory.ARM,
    "right_collar": JointCategory.ARM,
    "left_shoulder": JointCategory.ARM,
    "right_shoulder": JointCategory.ARM,
    "left_elbow": JointCategory.ARM,
    "right_elbow": JointCategory.ARM,
    "neck": JointCategory.HEAD,
    "head": JointCategory.HEAD,
    "left_wrist": JointCategory.HAND,
    "right_wrist": JointCategory.HAND,
    "left_ankle": JointCategory.FOOT,
    "right_ankle": JointCategory.FOOT,
    "left_foot": JointCategory.FOOT,
    "right_foot": JointCategory.FOOT,
}


def classify_joint(joint_name: str) -> JointCategory:
    if joint_name in _JOINT_CATEGORY_MAP:
        return _JOINT_CATEGORY_MAP[joint_name]

    name_lower = joint_name.lower()
    if "spine" in name_lower or "chest" in name_lower:
        return JointCategory.SPINE
    if "shoulder" in name_lower or "elbow" in name_lower or "collar" in name_lower:
        return JointCategory.ARM
    if "hip" in name_lower or "knee" in name_lower:
        return JointCategory.LEG
    if "wrist" in name_lower or "hand" in name_lower:
        return JointCategory.HAND
    if "ankle" in name_lower or "foot" in name_lower or "toe" in name_lower:
        return JointCategory.FOOT
    if "head" in name_lower or "neck" in name_lower:
        return JointCategory.HEAD

    return JointCategory.ROOT


def compute_world_rotation(
    parent_indices: list[int],
    local_rotations: np.ndarray,
    joint_index: int,
) -> np.ndarray:
    chain = _build_ancestor_chain(parent_indices, joint_index)

    world_rot = np.array([0.0, 0.0, 0.0, 1.0])
    for idx in chain:
        world_rot = quaternion_multiply(world_rot, local_rotations[idx])

    return world_rot


def _build_ancestor_chain(parent_indices: list[int], joint_index: int) -> list[int]:
    chain: list[int] = []
    current = joint_index
    while current != -1:
        chain.append(current)
        current = parent_indices[current]
    chain.reverse()
    return chain
