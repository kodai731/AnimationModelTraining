from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from anim_ml.data.bvh_parser import parse_bvh
from anim_ml.data.dataset_100style import find_100style_bvh_files
from anim_ml.data.dataset_cmu import find_cmu_bvh_files
from anim_ml.data.rig_data_generator import (
    generate_rig_samples_from_motion,
    save_rig_samples_hdf5,
)

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
) -> list:
    try:
        z_up = detect_z_up(dataset_type)
        motion = parse_bvh(filepath, z_up=z_up)
        return generate_rig_samples_from_motion(motion, rng=rng)

    except Exception:
        logger.exception("Failed to process: %s", filepath)
        return []


def run_pipeline(dataset: str, raw_dir: Path, output_dir: Path, limit: int | None) -> None:
    files = collect_bvh_files(dataset, raw_dir)
    if limit:
        files = files[:limit]

    logger.info("Processing %d BVH files", len(files))

    rng = np.random.default_rng(42)
    all_samples: list = []

    for filepath, ds_type in files:
        samples = process_single_bvh(filepath, ds_type, rng)
        if samples:
            all_samples.extend(samples)
            logger.info("  %s: %d samples", filepath.name, len(samples))

    logger.info("Total samples: %d", len(all_samples))

    if not all_samples:
        logger.warning("No samples generated. Check raw data directory.")
        return

    output_path = output_dir / f"{dataset}_rig.h5"
    save_rig_samples_hdf5(all_samples, output_path)
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
    run_pipeline(args.dataset, args.raw_dir, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
