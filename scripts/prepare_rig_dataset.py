from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np

from anim_ml.data.bvh_parser import parse_bvh
from anim_ml.data.dataset_100style import find_100style_bvh_files
from anim_ml.data.dataset_cmu import find_cmu_bvh_files
from anim_ml.data.rig_data_generator import (
    generate_rig_samples_from_motion,
    save_rig_samples_hdf5,
)
from anim_ml.utils.preparation_log import PreparationLog

logger = logging.getLogger(__name__)

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
    rng: np.random.Generator,
    prep_log: PreparationLog,
) -> list:
    try:
        z_up = detect_z_up(dataset_type)

        prep_log.log("parse_bvh_start", file=filepath.name, dataset=dataset_type)
        motion = parse_bvh(filepath, z_up=z_up)
        prep_log.log("parse_bvh_done", file=filepath.name, n_frames=motion.n_frames, n_joints=len(motion.joint_names))

        prep_log.log("generate_samples_start", file=filepath.name)
        samples = generate_rig_samples_from_motion(motion, rng=rng)
        prep_log.log("generate_samples_done", file=filepath.name, n_samples=len(samples))

        del motion
        return samples

    except Exception:
        logger.exception("Failed to process: %s", filepath)
        prep_log.log("process_bvh_error", file=filepath.name)
        return []


def _estimate_samples_memory_mb(all_samples: list) -> float:
    total_bytes = sys.getsizeof(all_samples)
    for sample in all_samples:
        total_bytes += sys.getsizeof(sample)
        if hasattr(sample, "__dict__"):
            for val in sample.__dict__.values():
                if hasattr(val, "nbytes"):
                    total_bytes += val.nbytes
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

    rng = np.random.default_rng(42)
    all_samples: list = []

    for i, (filepath, ds_type) in enumerate(files):
        samples = process_single_bvh(filepath, ds_type, rng, prep_log)
        if samples:
            all_samples.extend(samples)
            logger.info("  %s: %d samples", filepath.name, len(samples))

        if (i + 1) % 10 == 0:
            gc.collect()
            samples_mb = _estimate_samples_memory_mb(all_samples)
            prep_log.log(
                "progress",
                files_done=i + 1,
                total_files=len(files),
                total_samples=len(all_samples),
                samples_est_mb=round(samples_mb, 1),
            )

    prep_log.log(
        "all_bvh_done",
        total_samples=len(all_samples),
        samples_est_mb=round(_estimate_samples_memory_mb(all_samples), 1),
    )
    logger.info("Total samples: %d", len(all_samples))

    if not all_samples:
        logger.warning("No samples generated. Check raw data directory.")
        return

    output_path = output_dir / f"{dataset}_rig.h5"
    prep_log.log("save_hdf5_start", output=str(output_path))
    save_rig_samples_hdf5(all_samples, output_path)
    del all_samples
    gc.collect()
    prep_log.log("save_hdf5_done", output=str(output_path))
    logger.info("Saved to %s", output_path)


def main() -> None:
    from anim_ml.paths import get_processed_data_dir, get_raw_data_dir

    parser = argparse.ArgumentParser(description="Prepare rig propagation training dataset")
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

    prep_log = PreparationLog("prepare_rig")
    prep_log.log("main_started", dataset=args.dataset)
    run_pipeline(args.dataset, args.raw_dir, args.output_dir, args.limit, prep_log)
    prep_log.log("main_finished")


if __name__ == "__main__":
    main()
