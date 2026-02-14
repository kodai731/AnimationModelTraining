from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from anim_ml.data.bvh_parser import parse_bvh
from anim_ml.data.curve_extractor import CurveSample, extract_curve_samples
from anim_ml.data.dataset_100style import find_100style_bvh_files, map_100style_to_smpl
from anim_ml.data.dataset_cmu import find_cmu_bvh_files, map_cmu_to_smpl

logger = logging.getLogger(__name__)

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42


def collect_bvh_files(dataset: str, raw_dir: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []

    if dataset in ("cmu", "all"):
        cmu_dir = raw_dir / "cmu" / "bvh"
        if cmu_dir.exists():
            for p in find_cmu_bvh_files(cmu_dir):
                files.append((p, "cmu"))

    if dataset in ("100style", "all"):
        style_dir = raw_dir / "100style" / "bvh"
        if style_dir.exists():
            for p in find_100style_bvh_files(style_dir):
                files.append((p, "100style"))

    return files


def process_single_bvh(filepath: Path, dataset_type: str) -> list[CurveSample]:
    try:
        z_up = dataset_type == "cmu"
        motion = parse_bvh(filepath, z_up=z_up)

        mapped = map_cmu_to_smpl(motion) if dataset_type == "cmu" else map_100style_to_smpl(motion)

        if mapped is None:
            logger.warning("Skipping (insufficient joint mapping): %s", filepath.name)
            return []

        return extract_curve_samples(mapped)

    except Exception:
        logger.exception("Failed to process: %s", filepath)
        return []


def split_clips(
    clip_samples: list[list[CurveSample]],
) -> dict[str, list[CurveSample]]:
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(len(clip_samples))
    rng.shuffle(indices)

    n = len(clip_samples)
    train_end = int(n * SPLIT_RATIOS["train"])
    val_end = train_end + int(n * SPLIT_RATIOS["val"])

    splits: dict[str, list[CurveSample]] = {"train": [], "val": [], "test": []}

    for i in indices[:train_end]:
        splits["train"].extend(clip_samples[i])
    for i in indices[train_end:val_end]:
        splits["val"].extend(clip_samples[i])
    for i in indices[val_end:]:
        splits["test"].extend(clip_samples[i])

    return splits


def save_hdf5(splits: dict[str, list[CurveSample]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        for split_name, samples in splits.items():
            if not samples:
                continue

            n = len(samples)
            grp = f.create_group(split_name)

            context = np.stack([s.context_keyframes for s in samples])
            target = np.stack([s.target_keyframe for s in samples])
            prop_type = np.array([s.property_type for s in samples], dtype=np.int32)
            joint_cat = np.array([s.joint_category for s in samples], dtype=np.int32)
            query_time = np.array([s.query_time for s in samples], dtype=np.float32)
            clip_dur = np.array([s.clip_duration for s in samples], dtype=np.float32)
            joint_depth = np.array([s.joint_depth for s in samples], dtype=np.int32)
            curve_mean = np.array([s.curve_mean for s in samples], dtype=np.float32)
            curve_std = np.array([s.curve_std for s in samples], dtype=np.float32)

            grp.create_dataset("context_keyframes", data=context)
            grp.create_dataset("target", data=target)
            grp.create_dataset("property_type", data=prop_type)
            grp.create_dataset("joint_category", data=joint_cat)
            grp.create_dataset("query_time", data=query_time)
            grp.create_dataset("clip_duration", data=clip_dur)
            grp.create_dataset("joint_depth", data=joint_depth)
            grp.create_dataset("curve_mean", data=curve_mean)
            grp.create_dataset("curve_std", data=curve_std)

            logger.info("Split '%s': %d samples", split_name, n)


def run_pipeline(dataset: str, raw_dir: Path, output_dir: Path, limit: int | None) -> None:
    files = collect_bvh_files(dataset, raw_dir)
    if limit:
        files = files[:limit]

    logger.info("Processing %d BVH files", len(files))

    clip_samples: list[list[CurveSample]] = []
    total_samples = 0

    for filepath, ds_type in files:
        samples = process_single_bvh(filepath, ds_type)
        if samples:
            clip_samples.append(samples)
            total_samples += len(samples)
            logger.info("  %s: %d samples", filepath.name, len(samples))

    logger.info("Total clips: %d, Total samples: %d", len(clip_samples), total_samples)

    if not clip_samples:
        logger.warning("No samples generated. Check raw data directory.")
        return

    splits = split_clips(clip_samples)

    output_path = output_dir / f"{dataset}_curves.h5"
    save_hdf5(splits, output_path)
    logger.info("Saved to %s", output_path)


def main() -> None:
    from anim_ml.paths import get_processed_data_dir, get_raw_data_dir

    parser = argparse.ArgumentParser(description="Prepare curve training dataset")
    parser.add_argument(
        "--dataset", choices=["cmu", "100style", "all"], default="all",
    )
    parser.add_argument("--raw-dir", type=Path, default=get_raw_data_dir())
    parser.add_argument("--output-dir", type=Path, default=get_processed_data_dir())
    parser.add_argument("--limit", type=int, default=None, help="Limit number of BVH files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_pipeline(args.dataset, args.raw_dir, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
