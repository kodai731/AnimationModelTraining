from __future__ import annotations

import argparse
import logging
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

CODEWELT_BASE = "https://codewelt.com/dl/cmuconvert"

CMU_ZIP_FILES = [
    "cmuconvert-mb2-01-09.zip",
    "cmuconvert-mb2-10-14.zip",
    "cmuconvert-mb2-15-19.zip",
    "cmuconvert-mb2-20-29.zip",
    "cmuconvert-mb2-30-34.zip",
    "cmuconvert-mb2-35-39.zip",
    "cmuconvert-mb2-40-45.zip",
    "cmuconvert-mb2-46-56.zip",
    "cmuconvert-mb2-60-75.zip",
    "cmuconvert-mb2-76-80.zip",
    "cmuconvert-mb2-81-85.zip",
    "cmuconvert-mb2-86-94.zip",
    "cmuconvert-mb2-102-111.zip",
    "cmuconvert-mb2-113-128.zip",
    "cmuconvert-mb2-131-135.zip",
    "cmuconvert-mb2-136-140.zip",
    "cmuconvert-mb2-141-144.zip",
]


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
        with urllib.request.urlopen(req, timeout=120) as response, open(tmp, "ab") as f:
            total = response.headers.get("Content-Length")
            downloaded = 0
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (1024 * 1024) < 8192:
                    pct = downloaded / int(total) * 100
                    logger.info("  %s: %.1f%%", dest.name, pct)

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

    zips = CMU_ZIP_FILES[:limit] if limit else CMU_ZIP_FILES

    for zip_name in zips:
        url = f"{CODEWELT_BASE}/{zip_name}"
        zip_path = output_dir / "zips" / zip_name

        if not download_file(url, zip_path):
            continue

        bvh_dir = output_dir / "bvh"
        extracted = extract_zip(zip_path, bvh_dir)
        all_bvh.extend(extracted)
        logger.info("%s: %d BVH files extracted", zip_name, len(extracted))

    logger.info("Total CMU BVH files: %d", len(all_bvh))
    return all_bvh


def main() -> None:
    from anim_ml.paths import get_raw_data_dir

    parser = argparse.ArgumentParser(description="Download CMU MoCap BVH files")
    parser.add_argument("--output-dir", type=Path, default=get_raw_data_dir() / "cmu")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of zip archives",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_cmu(args.output_dir, args.limit)


if __name__ == "__main__":
    main()
