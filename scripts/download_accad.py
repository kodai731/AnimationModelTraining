from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ACCAD_GUIDE = """
ACCAD MoCap BVH files are available from the ACCAD Motion Lab at Ohio State University.

Download steps:
  1. Visit https://accad.osu.edu/research/motion-lab/mocap-system-and-calculation
  2. Download the BVH motion files
  3. Place .bvh files into: {output_dir}/bvh/

Alternative (via AMASS - research use only):
  1. Visit https://amass.is.tue.mpg.de/
  2. Register and download the ACCAD subset in BVH format
  3. Place .bvh files into: {output_dir}/bvh/

After placing the files, run:
  uv run python scripts/prepare_dataset.py --dataset accad
"""


def verify_bvh_files(bvh_dir: Path) -> list[Path]:
    if not bvh_dir.exists():
        return []

    return sorted(
        p for p in bvh_dir.rglob("*.bvh")
        if "__MACOSX" not in p.parts and not p.name.startswith("._")
    )


def main() -> None:
    from anim_ml.paths import get_raw_data_dir

    parser = argparse.ArgumentParser(description="Setup ACCAD MoCap BVH files")
    parser.add_argument("--output-dir", type=Path, default=get_raw_data_dir() / "accad")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    bvh_dir = args.output_dir / "bvh"
    existing = verify_bvh_files(bvh_dir)

    if existing:
        logger.info("Found %d ACCAD BVH files in %s", len(existing), bvh_dir)
        return

    bvh_dir.mkdir(parents=True, exist_ok=True)
    print(ACCAD_GUIDE.format(output_dir=args.output_dir))


if __name__ == "__main__":
    main()
