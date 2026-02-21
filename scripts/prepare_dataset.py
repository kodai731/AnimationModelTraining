from __future__ import annotations

import argparse
import gc
import logging
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
        prep_log.log("parse_bvh_done", file=filepath.name, n_frames=motion.positions.shape[0], n_joints=len(motion.joint_names))

        prep_log.log("extract_samples_start", file=filepath.name)
        samples = extract_curve_samples(motion)
        prep_log.log("extract_samples_done", file=filepath.name, n_samples=len(samples))

        del motion
        return samples

    except Exception:
        logger.exception("Failed to process: %s", filepath)
        prep_log.log("process_bvh_error", file=filepath.name)
        return []


def _assign_splits(n_files: int) -> dict[int, str]:
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(n_files)
    rng.shuffle(indices)

    train_end = int(n_files * SPLIT_RATIOS["train"])
    val_end = train_end + int(n_files * SPLIT_RATIOS["val"])

    assignment: dict[int, str] = {}
    for i in indices[:train_end]:
        assignment[int(i)] = "train"
    for i in indices[train_end:val_end]:
        assignment[int(i)] = "val"
    for i in indices[val_end:]:
        assignment[int(i)] = "test"

    return assignment


def _samples_to_arrays(samples: list[CurveSample]) -> dict[str, np.ndarray]:
    return {
        "context_keyframes": np.stack([s.context_keyframes for s in samples]),
        "target": np.stack([s.target_keyframe for s in samples]),
        "property_type": np.array([s.property_type for s in samples], dtype=np.int32),
        "topology_features": np.stack([s.topology_features for s in samples]),
        "bone_name_tokens": np.stack([s.bone_name_tokens for s in samples]).astype(np.int32),
        "query_time": np.array([s.query_time for s in samples], dtype=np.float32),
        "clip_duration": np.array([s.clip_duration for s in samples], dtype=np.float32),
        "joint_depth": np.array([s.joint_depth for s in samples], dtype=np.int32),
        "curve_mean": np.array([s.curve_mean for s in samples], dtype=np.float32),
        "curve_std": np.array([s.curve_std for s in samples], dtype=np.float32),
    }


def _append_to_hdf5(
    grp: h5py.Group,
    samples: list[CurveSample],
    current_count: int,
) -> int:
    arrays = _samples_to_arrays(samples)
    n_new = len(samples)

    if current_count == 0:
        for name, arr in arrays.items():
            maxshape = (None,) + arr.shape[1:]
            grp.create_dataset(
                name, data=arr, maxshape=maxshape, chunks=True,
                compression="gzip", compression_opts=4,
            )
    else:
        for name, arr in arrays.items():
            ds = grp[name]
            ds.resize(current_count + n_new, axis=0)
            ds[current_count : current_count + n_new] = arr

    return current_count + n_new


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

    split_assignment = _assign_splits(len(files))
    output_path = output_dir / f"{dataset}_curves.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    total_samples = 0
    valid_clips = 0

    with h5py.File(output_path, "w") as f:
        groups = {name: f.create_group(name) for name in ("train", "val", "test")}

        for i, (filepath, ds_type) in enumerate(files):
            samples = process_single_bvh(filepath, ds_type, prep_log)

            if samples:
                split = split_assignment[i]
                split_counts[split] = _append_to_hdf5(groups[split], samples, split_counts[split])
                total_samples += len(samples)
                valid_clips += 1
                logger.info("  %s: %d samples -> %s", filepath.name, len(samples), split)

                del samples
                gc.collect()

            if (i + 1) % 50 == 0:
                prep_log.log(
                    "progress",
                    files_done=i + 1,
                    total_files=len(files),
                    valid_clips=valid_clips,
                    total_samples=total_samples,
                    split_counts=split_counts,
                )

    prep_log.log(
        "pipeline_done",
        valid_clips=valid_clips,
        total_samples=total_samples,
        split_counts=split_counts,
        output=str(output_path),
    )
    logger.info("Total clips: %d, Total samples: %d", valid_clips, total_samples)
    logger.info("Splits: %s", split_counts)
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
