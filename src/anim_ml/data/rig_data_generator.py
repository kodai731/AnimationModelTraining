from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np

from anim_ml.utils.rotation import (
    euler_xyz_to_quaternion,
    quaternion_geodesic_distance,
    quaternion_inverse,
    quaternion_multiply,
)

if TYPE_CHECKING:
    from pathlib import Path

    from anim_ml.data.bvh_parser import MotionData


class RigJointType(IntEnum):
    ROOT = 0
    SPINE = 1
    CHEST = 2
    NECK = 3
    HEAD = 4
    SHOULDER = 5
    UPPER_ARM = 6
    LOWER_ARM = 7
    HAND = 8
    UPPER_LEG = 9
    LOWER_LEG = 10
    FOOT = 11
    TOES = 12


RIG_JOINT_TYPE_MAP: dict[str, RigJointType] = {
    "pelvis": RigJointType.ROOT,
    "left_hip": RigJointType.UPPER_LEG,
    "right_hip": RigJointType.UPPER_LEG,
    "spine1": RigJointType.SPINE,
    "left_knee": RigJointType.LOWER_LEG,
    "right_knee": RigJointType.LOWER_LEG,
    "spine2": RigJointType.CHEST,
    "left_ankle": RigJointType.FOOT,
    "right_ankle": RigJointType.FOOT,
    "spine3": RigJointType.CHEST,
    "left_foot": RigJointType.TOES,
    "right_foot": RigJointType.TOES,
    "neck": RigJointType.NECK,
    "left_collar": RigJointType.SHOULDER,
    "right_collar": RigJointType.SHOULDER,
    "head": RigJointType.HEAD,
    "left_shoulder": RigJointType.UPPER_ARM,
    "right_shoulder": RigJointType.UPPER_ARM,
    "left_elbow": RigJointType.LOWER_ARM,
    "right_elbow": RigJointType.LOWER_ARM,
    "left_wrist": RigJointType.HAND,
    "right_wrist": RigJointType.HAND,
}

SMPL_MIRROR_PAIRS: list[tuple[int, int]] = [
    (1, 2),
    (4, 5),
    (7, 8),
    (10, 11),
    (13, 14),
    (16, 17),
    (18, 19),
    (20, 21),
]

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
CONFIDENCE_THRESHOLD = 0.01


@dataclass
class RigSample:
    joint_features: np.ndarray
    joint_types: np.ndarray
    target_deltas: np.ndarray
    confidence_targets: np.ndarray


def classify_rig_joint(joint_name: str) -> RigJointType:
    return RIG_JOINT_TYPE_MAP[joint_name]


def build_adjacency(parent_indices: list[int]) -> np.ndarray:
    sources: list[int] = []
    targets: list[int] = []

    for child, parent in enumerate(parent_indices):
        if parent == -1:
            continue
        sources.append(parent)
        targets.append(child)
        sources.append(child)
        targets.append(parent)

    return np.array([sources, targets], dtype=np.int64)


def compute_hierarchy_depth(parent_indices: list[int]) -> np.ndarray:
    num_joints = len(parent_indices)
    depths = np.zeros(num_joints, dtype=np.float64)

    for joint_idx in range(num_joints):
        depth = 0
        current = joint_idx
        while parent_indices[current] != -1:
            depth += 1
            current = parent_indices[current]
        depths[joint_idx] = depth

    max_depth = depths.max()
    if max_depth > 0:
        depths /= max_depth

    return depths


def generate_rig_samples_from_motion(
    motion: MotionData,
    frame_skips: list[int] | None = None,
    num_edited_range: tuple[int, int] = (1, 3),
    augment_mirror: bool = True,
    augment_noise: bool = True,
    rng: np.random.Generator | None = None,
) -> list[RigSample]:
    if rng is None:
        rng = np.random.default_rng()
    if frame_skips is None:
        frame_skips = [1, 2, 4, 8]

    num_joints = len(motion.joint_names)
    num_frames = motion.rotations.shape[0]
    if num_frames < 2 or num_joints != 22:
        return []

    rotations_rad = np.deg2rad(motion.rotations)
    quats = euler_xyz_to_quaternion(rotations_rad)

    joint_types = np.array(
        [classify_rig_joint(name) for name in motion.joint_names],
        dtype=np.int64,
    )
    depths = compute_hierarchy_depth(motion.parent_indices)

    samples: list[RigSample] = []

    for skip in frame_skips:
        for frame_a in range(0, num_frames - skip, max(skip, 1)):
            frame_b = frame_a + skip
            if frame_b >= num_frames:
                continue

            q_a = quats[frame_a]
            q_b = quats[frame_b]
            deltas = quaternion_multiply(quaternion_inverse(q_a), q_b)

            sample = _create_sample_from_frame_pair(
                q_a, deltas, joint_types, depths, num_edited_range, rng,
            )
            if sample is not None:
                samples.append(sample)

            if augment_mirror:
                mirrored_sample = _create_mirrored_sample(
                    q_a, deltas, joint_types, depths, num_edited_range, rng,
                )
                if mirrored_sample is not None:
                    samples.append(mirrored_sample)

    if augment_noise and samples:
        noise_samples = _create_noise_augmented_samples(samples, rng)
        samples.extend(noise_samples)

    return samples


def _create_sample_from_frame_pair(
    base_quats: np.ndarray,
    deltas: np.ndarray,
    joint_types: np.ndarray,
    depths: np.ndarray,
    num_edited_range: tuple[int, int],
    rng: np.random.Generator,
) -> RigSample | None:
    num_joints = base_quats.shape[0]
    num_edited = rng.integers(num_edited_range[0], num_edited_range[1] + 1)
    edited_joints = rng.choice(num_joints, size=min(num_edited, num_joints), replace=False)

    joint_features = np.zeros((num_joints, 10), dtype=np.float32)
    target_deltas = np.zeros((num_joints, 4), dtype=np.float32)
    confidence_targets = np.zeros(num_joints, dtype=np.float32)

    for j in range(num_joints):
        joint_features[j, 0:4] = base_quats[j].astype(np.float32)
        joint_features[j, 9] = depths[j].astype(np.float32)

        dist = quaternion_geodesic_distance(
            deltas[j:j + 1], IDENTITY_QUAT.reshape(1, 4),
        )[0]

        if j in edited_joints:
            joint_features[j, 4:8] = deltas[j].astype(np.float32)
            joint_features[j, 8] = 1.0
            target_deltas[j] = IDENTITY_QUAT.astype(np.float32)
        else:
            joint_features[j, 4:8] = IDENTITY_QUAT.astype(np.float32)
            target_deltas[j] = deltas[j].astype(np.float32)

        confidence_targets[j] = 1.0 if dist > CONFIDENCE_THRESHOLD else 0.0

    return RigSample(
        joint_features=joint_features,
        joint_types=joint_types.copy(),
        target_deltas=target_deltas,
        confidence_targets=confidence_targets,
    )


def _mirror_quaternion(quat: np.ndarray) -> np.ndarray:
    mirrored = quat.copy()
    mirrored[..., 0] *= -1.0
    mirrored[..., 2] *= -1.0
    return mirrored


def _create_mirrored_sample(
    base_quats: np.ndarray,
    deltas: np.ndarray,
    joint_types: np.ndarray,
    depths: np.ndarray,
    num_edited_range: tuple[int, int],
    rng: np.random.Generator,
) -> RigSample | None:
    mirrored_base = _mirror_quaternion(base_quats.copy())
    mirrored_deltas = _mirror_quaternion(deltas.copy())

    swap_map = np.arange(len(base_quats))
    for left, right in SMPL_MIRROR_PAIRS:
        swap_map[left] = right
        swap_map[right] = left

    mirrored_base = mirrored_base[swap_map]
    mirrored_deltas = mirrored_deltas[swap_map]

    return _create_sample_from_frame_pair(
        mirrored_base, mirrored_deltas, joint_types, depths,
        num_edited_range, rng,
    )


def _create_noise_augmented_samples(
    samples: list[RigSample],
    rng: np.random.Generator,
    noise_ratio: float = 0.1,
) -> list[RigSample]:
    num_noise = max(1, int(len(samples) * noise_ratio))
    indices = rng.choice(len(samples), size=min(num_noise, len(samples)), replace=False)

    augmented: list[RigSample] = []
    for idx in indices:
        original: RigSample = samples[int(idx)]
        features_copy = np.array(original.joint_features, dtype=np.float32, copy=True)
        noise_shape = features_copy[:, :8].shape
        noise = rng.normal(0, 0.01, size=noise_shape).astype(np.float32)
        features_copy[:, :8] += noise

        quat_norms: np.ndarray = np.linalg.norm(
            features_copy[:, :4], axis=-1, keepdims=True,
        )
        quat_norms = np.maximum(quat_norms, 1e-8)
        features_copy[:, :4] /= quat_norms

        augmented.append(RigSample(
            joint_features=features_copy,
            joint_types=np.array(original.joint_types, dtype=np.int64, copy=True),
            target_deltas=np.array(original.target_deltas, dtype=np.float32, copy=True),
            confidence_targets=np.array(original.confidence_targets, dtype=np.float32, copy=True),
        ))

    return augmented


def save_rig_samples_hdf5(
    samples: list[RigSample],
    output_path: Path,
    adjacency: np.ndarray,
    split_ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> None:
    if split_ratios is None:
        split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    n = len(samples)
    train_end = int(n * split_ratios["train"])
    val_end = train_end + int(n * split_ratios["val"])

    split_indices: dict[str, np.ndarray] = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:  # type: ignore[no-untyped-call]
        f.create_dataset("adjacency", data=adjacency)  # type: ignore[no-untyped-call]

        for split_name, split_idx in split_indices.items():
            if len(split_idx) == 0:
                continue

            grp = f.create_group(split_name)  # type: ignore[no-untyped-call]

            split_samples: list[RigSample] = [samples[int(i)] for i in split_idx]
            stacked_features = np.stack([s.joint_features for s in split_samples])
            stacked_deltas = np.stack([s.target_deltas for s in split_samples])
            stacked_conf = np.stack([s.confidence_targets for s in split_samples])
            stacked_types = np.stack([s.joint_types for s in split_samples])

            grp.create_dataset("joint_features", data=stacked_features)  # type: ignore[no-untyped-call]
            grp.create_dataset("target_deltas", data=stacked_deltas)  # type: ignore[no-untyped-call]
            grp.create_dataset("confidence_targets", data=stacked_conf)  # type: ignore[no-untyped-call]
            grp.create_dataset("joint_types", data=stacked_types)  # type: ignore[no-untyped-call]
