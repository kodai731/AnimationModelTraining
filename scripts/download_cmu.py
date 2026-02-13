from __future__ import annotations

import argparse
import logging
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

CMU_BVH_URL = "https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture"
CMU_MIRROR_BASE = "http://mocap.cs.cmu.edu/subjects"

CMU_SUBJECT_RANGES = list(range(1, 145))


def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        logger.info("Already exists: %s", dest.name)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        headers = {}
        if tmp.exists():
            existing_size = tmp.stat().st_size
            headers["Range"] = f"bytes={existing_size}-"
            logger.info("Resuming %s from %d bytes", dest.name, existing_size)

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response, open(tmp, "ab") as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)

        tmp.rename(dest)
        logger.info("Downloaded: %s", dest.name)
        return True

    except Exception:
        logger.exception("Failed to download %s", url)
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> list[Path]:
    bvh_files: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith(".bvh"):
                zf.extract(name, output_dir)
                bvh_files.append(output_dir / name)
    return bvh_files


def download_cmu(output_dir: Path, limit: int | None = None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_bvh: list[Path] = []

    subjects = CMU_SUBJECT_RANGES[:limit] if limit else CMU_SUBJECT_RANGES

    for subject_id in subjects:
        subject_str = f"{subject_id:02d}"
        url = f"{CMU_MIRROR_BASE}/{subject_str}/{subject_str}.zip"
        zip_path = output_dir / "zips" / f"cmu_{subject_str}.zip"

        if not download_file(url, zip_path):
            continue

        bvh_dir = output_dir / "bvh"
        extracted = extract_zip(zip_path, bvh_dir)
        all_bvh.extend(extracted)
        logger.info("Subject %s: %d BVH files", subject_str, len(extracted))

    logger.info("Total CMU BVH files: %d", len(all_bvh))
    return all_bvh


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CMU MoCap BVH files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/cmu"))
    parser.add_argument("--limit", type=int, default=None, help="Limit number of subjects")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_cmu(args.output_dir, args.limit)


if __name__ == "__main__":
    main()
