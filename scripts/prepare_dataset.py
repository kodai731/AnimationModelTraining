from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

from anim_ml.data.bvh_parser import parse_bvh
from anim_ml.data.curve_extractor import CurveSample, extract_curve_samples
from anim_ml.data.dataset_100style import find_100style_bvh_files
from anim_ml.data.dataset_cmu import find_cmu_bvh_files
from anim_ml.utils.preparation_log import PreparationLog

logger = logging.getLogger(__name__)

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42
Z_UP_DATASETS = {"cmu"}


def find_generic_bvh_files(data_dir: Path) -> list[Path]:
    return sorted(
        p for p in data_dir.rglob("*.bvh")
        if "__MACOSX" not in p.parts and not p.name.startswith("._")
    )


def detect_z_up(dataset_type: str) -> bool:
    return dataset_type in Z_UP_DATASETS


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

    if dataset in ("accad", "all"):
        accad_dir = raw_dir / "accad" / "bvh"
        if accad_dir.exists():
            for p in find_generic_bvh_files(accad_dir):
                files.append((p, "accad"))

    if dataset in ("generic", "all"):
        generic_dir = raw_dir / "generic" / "bvh"
        if generic_dir.exists():
            for p in find_generic_bvh_files(generic_dir):
                files.append((p, "generic"))

    return files


def process_single_bvh(
    filepath: Path,
    dataset_type: str,
    prep_log: PreparationLog,
) -> list[CurveSample]:
    try:
        z_up = detect_z_up(dataset_type)

        prep_log.log("parse_bvh_start", file=filepath.name, dataset=dataset_type)
        motion = parse_bvh(filepath, z_up=z_up)
        prep_log.log("parse_bvh_done", file=filepath.name, n_frames=motion.n_frames, n_joints=len(motion.joint_names))

        prep_log.log("extract_samples_start", file=filepath.name)
        samples = extract_curve_samples(motion)
        prep_log.log("extract_samples_done", file=filepath.name, n_samples=len(samples))

        del motion
        return samples

    except Exception:
        logger.exception("Failed to process: %s", filepath)
        prep_log.log("process_bvh_error", file=filepath.name)
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
            topo_features = np.stack([s.topology_features for s in samples])
            bone_name_tokens = np.stack([s.bone_name_tokens for s in samples])
            query_time = np.array([s.query_time for s in samples], dtype=np.float32)
            clip_dur = np.array([s.clip_duration for s in samples], dtype=np.float32)
            joint_depth = np.array([s.joint_depth for s in samples], dtype=np.int32)
            curve_mean = np.array([s.curve_mean for s in samples], dtype=np.float32)
            curve_std = np.array([s.curve_std for s in samples], dtype=np.float32)

            grp.create_dataset("context_keyframes", data=context)
            grp.create_dataset("target", data=target)
            grp.create_dataset("property_type", data=prop_type)
            grp.create_dataset("topology_features", data=topo_features)
            grp.create_dataset("bone_name_tokens", data=bone_name_tokens)
            grp.create_dataset("query_time", data=query_time)
            grp.create_dataset("clip_duration", data=clip_dur)
            grp.create_dataset("joint_depth", data=joint_depth)
            grp.create_dataset("curve_mean", data=curve_mean)
            grp.create_dataset("curve_std", data=curve_std)

            logger.info("Split '%s': %d samples", split_name, n)


def _estimate_samples_memory_mb(clip_samples: list[list[CurveSample]]) -> float:
    total_bytes = sys.getsizeof(clip_samples)
    for clip in clip_samples:
        total_bytes += sys.getsizeof(clip)
        for sample in clip:
            total_bytes += sys.getsizeof(sample)
            for arr in (sample.context_keyframes, sample.target_keyframe,
                        sample.topology_features, sample.bone_name_tokens):
                total_bytes += arr.nbytes
    return total_bytes / (1024 * 1024)


def run_pipeline(
    dataset: str,
    raw_dir: Path,
    output_dir: Path,
    limit: int | None,
    prep_log: PreparationLog,
) -> None:
    files = collect_bvh_files(dataset, raw_dir)
    if limit:
        files = files[:limit]

    prep_log.log("pipeline_start", dataset=dataset, n_files=len(files))
    logger.info("Processing %d BVH files", len(files))

    clip_samples: list[list[CurveSample]] = []
    total_samples = 0

    for i, (filepath, ds_type) in enumerate(files):
        samples = process_single_bvh(filepath, ds_type, prep_log)
        if samples:
            clip_samples.append(samples)
            total_samples += len(samples)
            logger.info("  %s: %d samples", filepath.name, len(samples))

        if (i + 1) % 10 == 0:
            gc.collect()
            samples_mb = _estimate_samples_memory_mb(clip_samples)
            prep_log.log(
                "progress",
                files_done=i + 1,
                total_files=len(files),
                total_clips=len(clip_samples),
                total_samples=total_samples,
                samples_est_mb=round(samples_mb, 1),
            )

    prep_log.log(
        "all_bvh_done",
        total_clips=len(clip_samples),
        total_samples=total_samples,
        samples_est_mb=round(_estimate_samples_memory_mb(clip_samples), 1),
    )
    logger.info("Total clips: %d, Total samples: %d", len(clip_samples), total_samples)

    if not clip_samples:
        logger.warning("No samples generated. Check raw data directory.")
        return

    prep_log.log("split_clips_start")
    splits = split_clips(clip_samples)
    del clip_samples
    gc.collect()
    prep_log.log("split_clips_done", splits={k: len(v) for k, v in splits.items()})

    output_path = output_dir / f"{dataset}_curves.h5"
    prep_log.log("save_hdf5_start", output=str(output_path))
    save_hdf5(splits, output_path)
    del splits
    gc.collect()
    prep_log.log("save_hdf5_done", output=str(output_path))
    logger.info("Saved to %s", output_path)


def main() -> None:
    from anim_ml.paths import get_processed_data_dir, get_raw_data_dir

    parser = argparse.ArgumentParser(description="Prepare curve training dataset")
    parser.add_argument(
        "--dataset",
        choices=["cmu", "100style", "accad", "generic", "all"],
        default="all",
    )
    parser.add_argument("--raw-dir", type=Path, default=get_raw_data_dir())
    parser.add_argument("--output-dir", type=Path, default=get_processed_data_dir())
    parser.add_argument("--limit", type=int, default=None, help="Limit number of BVH files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    prep_log = PreparationLog("prepare_curve")
    prep_log.log("main_started", dataset=args.dataset)
    run_pipeline(args.dataset, args.raw_dir, args.output_dir, args.limit, prep_log)
    prep_log.log("main_finished")


if __name__ == "__main__":
    main()
