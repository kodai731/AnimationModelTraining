from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np

from anim_ml.utils.bone_tokenizer import tokenize_bone_name
from anim_ml.utils.rotation import (
    euler_xyz_to_quaternion,
    quaternion_geodesic_distance,
    quaternion_inverse,
    quaternion_multiply,
)
from anim_ml.utils.topology import compute_topology_features

if TYPE_CHECKING:
    from pathlib import Path

    from anim_ml.data.bvh_parser import MotionData

MAX_JOINTS = 64
MAX_EDGES = 126
TOPOLOGY_DIM = 6
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
CONFIDENCE_THRESHOLD = 0.01

_MIRROR_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"^Left"), "Left", "Right"),
    (re.compile(r"^Right"), "Right", "Left"),
    (re.compile(r"^left_"), "left_", "right_"),
    (re.compile(r"^right_"), "right_", "left_"),
    (re.compile(r"_[Ll]eft_"), "_left_", "_right_"),
    (re.compile(r"_[Rr]ight_"), "_right_", "_left_"),
    (re.compile(r"_[Ll]eft$"), "_left", "_right"),
    (re.compile(r"_[Rr]ight$"), "_right", "_left"),
    (re.compile(r"(?<=[a-z])Left"), "Left", "Right"),
    (re.compile(r"(?<=[a-z])Right"), "Right", "Left"),
    (re.compile(r"^L_"), "L_", "R_"),
    (re.compile(r"^R_"), "R_", "L_"),
    (re.compile(r"_L$"), "_L", "_R"),
    (re.compile(r"_R$"), "_R", "_L"),
    (re.compile(r"\.L$"), ".L", ".R"),
    (re.compile(r"\.R$"), ".R", ".L"),
    (re.compile(r"\.L\."), ".L.", ".R."),
    (re.compile(r"\.R\."), ".R.", ".L."),
]


@dataclass
class RigSample:
    joint_features: np.ndarray
    target_deltas: np.ndarray
    confidence_targets: np.ndarray
    topology_features: np.ndarray
    bone_name_tokens: np.ndarray
    parent_indices: list[int]


def detect_mirror_pairs(joint_names: list[str]) -> list[tuple[int, int]]:
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    pairs: list[tuple[int, int]] = []
    visited: set[int] = set()

    for idx, name in enumerate(joint_names):
        if idx in visited:
            continue

        mirror_name = _find_mirror_name(name)
        if mirror_name is None:
            continue

        mirror_idx = name_to_idx.get(mirror_name)
        if mirror_idx is None or mirror_idx == idx or mirror_idx in visited:
            continue

        left_idx, right_idx = sorted([idx, mirror_idx])
        pairs.append((left_idx, right_idx))
        visited.add(idx)
        visited.add(mirror_idx)

    return pairs


def _find_mirror_name(name: str) -> str | None:
    for pattern, src_text, dst_text in _MIRROR_PATTERNS:
        if pattern.search(name):
            return name.replace(src_text, dst_text)
    return None


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


def build_edge_direction(
    source_indices: list[int],
    target_indices: list[int],
    parent_indices: list[int],
) -> list[int]:
    directions: list[int] = []
    for src, tgt in zip(source_indices, target_indices, strict=True):
        if parent_indices[tgt] == src:
            directions.append(0)
        else:
            directions.append(1)
    return directions


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
    if num_frames < 2 or num_joints < 2:
        return []

    rotations_rad = np.deg2rad(motion.rotations)
    quats = euler_xyz_to_quaternion(rotations_rad)

    topo_dict = compute_topology_features(motion.joint_names, motion.parent_indices)
    topo_array = np.array(
        [topo_dict[name] for name in motion.joint_names],
        dtype=np.float32,
    )

    tokens_array = np.array(
        [tokenize_bone_name(name) for name in motion.joint_names],
        dtype=np.int64,
    )

    mirror_pairs = detect_mirror_pairs(motion.joint_names) if augment_mirror else []

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
                q_a, deltas, topo_array, tokens_array,
                motion.parent_indices, num_edited_range, rng,
            )
            if sample is not None:
                samples.append(sample)

            if augment_mirror and mirror_pairs:
                mirrored_sample = _create_mirrored_sample(
                    q_a, deltas, topo_array, tokens_array,
                    motion.parent_indices, mirror_pairs,
                    num_edited_range, rng,
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
    topo_array: np.ndarray,
    tokens_array: np.ndarray,
    parent_indices: list[int],
    num_edited_range: tuple[int, int],
    rng: np.random.Generator,
) -> RigSample | None:
    num_joints = base_quats.shape[0]
    num_edited = rng.integers(num_edited_range[0], num_edited_range[1] + 1)
    edited_joints = rng.choice(num_joints, size=min(num_edited, num_joints), replace=False)

    joint_features = np.zeros((num_joints, 9), dtype=np.float32)
    target_deltas = np.zeros((num_joints, 4), dtype=np.float32)
    confidence_targets = np.zeros(num_joints, dtype=np.float32)

    for j in range(num_joints):
        joint_features[j, 0:4] = base_quats[j].astype(np.float32)

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
        target_deltas=target_deltas,
        confidence_targets=confidence_targets,
        topology_features=topo_array.copy(),
        bone_name_tokens=tokens_array.copy(),
        parent_indices=list(parent_indices),
    )


def _mirror_quaternion(quat: np.ndarray) -> np.ndarray:
    mirrored = quat.copy()
    mirrored[..., 0] *= -1.0
    mirrored[..., 2] *= -1.0
    return mirrored


def _create_mirrored_sample(
    base_quats: np.ndarray,
    deltas: np.ndarray,
    topo_array: np.ndarray,
    tokens_array: np.ndarray,
    parent_indices: list[int],
    mirror_pairs: list[tuple[int, int]],
    num_edited_range: tuple[int, int],
    rng: np.random.Generator,
) -> RigSample | None:
    mirrored_base = _mirror_quaternion(base_quats.copy())
    mirrored_deltas = _mirror_quaternion(deltas.copy())

    swap_map = np.arange(len(base_quats))
    for left, right in mirror_pairs:
        swap_map[left] = right
        swap_map[right] = left

    mirrored_base = mirrored_base[swap_map]
    mirrored_deltas = mirrored_deltas[swap_map]

    return _create_sample_from_frame_pair(
        mirrored_base, mirrored_deltas, topo_array, tokens_array,
        parent_indices, num_edited_range, rng,
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
        original = samples[int(idx)]
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
            target_deltas=np.array(original.target_deltas, dtype=np.float32, copy=True),
            confidence_targets=np.array(original.confidence_targets, dtype=np.float32, copy=True),
            topology_features=np.array(original.topology_features, dtype=np.float32, copy=True),
            bone_name_tokens=np.array(original.bone_name_tokens, dtype=np.int64, copy=True),
            parent_indices=list(original.parent_indices),
        ))

    return augmented


def _pad_to_max(array: np.ndarray, max_size: int, axis: int = 0) -> np.ndarray:
    current_size = array.shape[axis]
    if current_size >= max_size:
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(0, max_size)
        return array[tuple(slices)]

    pad_widths = [(0, 0)] * array.ndim
    pad_widths[axis] = (0, max_size - current_size)
    return np.pad(array, pad_widths, mode="constant", constant_values=0)


def _rig_samples_to_arrays(samples: list[RigSample]) -> dict[str, np.ndarray]:
    stacked_features = np.stack([_pad_to_max(s.joint_features, MAX_JOINTS) for s in samples])
    stacked_deltas = np.stack([_pad_to_max(s.target_deltas, MAX_JOINTS) for s in samples])
    stacked_conf = np.stack([_pad_to_max(s.confidence_targets, MAX_JOINTS) for s in samples])
    stacked_topo = np.stack([_pad_to_max(s.topology_features, MAX_JOINTS) for s in samples])
    stacked_tokens = np.stack([_pad_to_max(s.bone_name_tokens, MAX_JOINTS) for s in samples])
    joint_masks = np.stack([_build_joint_mask(len(s.parent_indices)) for s in samples])
    edge_data = np.stack([_build_padded_edge_data(s.parent_indices) for s in samples])

    return {
        "joint_features": stacked_features,
        "target_deltas": stacked_deltas,
        "confidence_targets": stacked_conf,
        "topology_features": stacked_topo,
        "bone_name_tokens": stacked_tokens,
        "joint_mask": joint_masks,
        "source_indices": edge_data[:, 0, :],
        "target_indices": edge_data[:, 1, :],
        "edge_direction": edge_data[:, 2, :],
        "edge_mask": edge_data[:, 3, :],
    }


def append_rig_samples_to_hdf5(
    grp: h5py.Group,  # type: ignore[type-arg]
    samples: list[RigSample],
    current_count: int,
) -> int:
    arrays = _rig_samples_to_arrays(samples)
    n_new = len(samples)

    if current_count == 0:
        for name, arr in arrays.items():
            maxshape = (None,) + arr.shape[1:]
            grp.create_dataset(name, data=arr, maxshape=maxshape, chunks=True)  # type: ignore[no-untyped-call]
    else:
        for name, arr in arrays.items():
            ds = grp[name]
            ds.resize(current_count + n_new, axis=0)
            ds[current_count : current_count + n_new] = arr

    return current_count + n_new


def save_rig_samples_hdf5(
    samples: list[RigSample],
    output_path: Path,
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
        for split_name, split_idx in split_indices.items():
            if len(split_idx) == 0:
                continue

            grp = f.create_group(split_name)  # type: ignore[no-untyped-call]
            split_samples = [samples[int(i)] for i in split_idx]

            _write_split_to_hdf5(grp, split_samples)


def _write_split_to_hdf5(
    grp: h5py.Group,  # type: ignore[type-arg]
    split_samples: list[RigSample],
) -> None:
    stacked_features = np.stack([
        _pad_to_max(s.joint_features, MAX_JOINTS) for s in split_samples
    ])
    stacked_deltas = np.stack([
        _pad_to_max(s.target_deltas, MAX_JOINTS) for s in split_samples
    ])
    stacked_conf = np.stack([
        _pad_to_max(s.confidence_targets, MAX_JOINTS) for s in split_samples
    ])
    stacked_topo = np.stack([
        _pad_to_max(s.topology_features, MAX_JOINTS) for s in split_samples
    ])
    stacked_tokens = np.stack([
        _pad_to_max(s.bone_name_tokens, MAX_JOINTS) for s in split_samples
    ])

    joint_masks = np.stack([
        _build_joint_mask(len(s.parent_indices)) for s in split_samples
    ])

    edge_data = np.stack([
        _build_padded_edge_data(s.parent_indices) for s in split_samples
    ])

    compress = dict(compression="gzip", compression_opts=4)
    grp.create_dataset("joint_features", data=stacked_features, **compress)  # type: ignore[no-untyped-call]
    grp.create_dataset("target_deltas", data=stacked_deltas, **compress)  # type: ignore[no-untyped-call]
    grp.create_dataset("confidence_targets", data=stacked_conf, **compress)  # type: ignore[no-untyped-call]
    grp.create_dataset("topology_features", data=stacked_topo, **compress)  # type: ignore[no-untyped-call]
    grp.create_dataset("bone_name_tokens", data=stacked_tokens.astype(np.int32), **compress)  # type: ignore[no-untyped-call]
    grp.create_dataset("joint_mask", data=joint_masks, **compress)  # type: ignore[no-untyped-call]
    grp.create_dataset("source_indices", data=edge_data[:, 0, :], **compress)  # type: ignore[no-untyped-call]
    grp.create_dataset("target_indices", data=edge_data[:, 1, :], **compress)  # type: ignore[no-untyped-call]
    grp.create_dataset("edge_direction", data=edge_data[:, 2, :], **compress)  # type: ignore[no-untyped-call]
    grp.create_dataset("edge_mask", data=edge_data[:, 3, :], **compress)  # type: ignore[no-untyped-call]


def _build_joint_mask(num_joints: int) -> np.ndarray:
    mask = np.zeros(MAX_JOINTS, dtype=np.float32)
    mask[:num_joints] = 1.0
    return mask


def _build_padded_edge_data(parent_indices: list[int]) -> np.ndarray:
    adj = build_adjacency(parent_indices)
    src_list = adj[0].tolist()
    tgt_list = adj[1].tolist()
    directions = build_edge_direction(src_list, tgt_list, parent_indices)
    num_edges = len(src_list)

    result = np.zeros((4, MAX_EDGES), dtype=np.int64)
    edge_count = min(num_edges, MAX_EDGES)
    result[0, :edge_count] = src_list[:edge_count]
    result[1, :edge_count] = tgt_list[:edge_count]
    result[2, :edge_count] = directions[:edge_count]
    result[3, :edge_count] = 1

    return result
