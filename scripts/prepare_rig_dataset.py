from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from anim_ml.data.bvh_parser import parse_bvh
from anim_ml.data.dataset_100style import find_100style_bvh_files, map_100style_to_smpl
from anim_ml.data.dataset_cmu import find_cmu_bvh_files, map_cmu_to_smpl
from anim_ml.data.rig_data_generator import (
    build_adjacency,
    generate_rig_samples_from_motion,
    save_rig_samples_hdf5,
)
from anim_ml.utils.skeleton import SMPL_22_PARENT_INDICES

logger = logging.getLogger(__name__)


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


def process_single_bvh(
    filepath: Path,
    dataset_type: str,
    rng: np.random.Generator,
) -> list:
    try:
        z_up = dataset_type == "cmu"
        motion = parse_bvh(filepath, z_up=z_up)

        mapped = map_cmu_to_smpl(motion) if dataset_type == "cmu" else map_100style_to_smpl(motion)

        if mapped is None:
            logger.warning("Skipping (insufficient joint mapping): %s", filepath.name)
            return []

        return generate_rig_samples_from_motion(mapped, rng=rng)

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

    adjacency = build_adjacency(SMPL_22_PARENT_INDICES)
    output_path = output_dir / f"{dataset}_rig.h5"
    save_rig_samples_hdf5(all_samples, output_path, adjacency)
    logger.info("Saved to %s", output_path)


def main() -> None:
    from anim_ml.paths import get_processed_data_dir, get_raw_data_dir

    parser = argparse.ArgumentParser(description="Prepare rig propagation training dataset")
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
