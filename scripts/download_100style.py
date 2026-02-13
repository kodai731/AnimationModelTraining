from __future__ import annotations

import argparse
import logging
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

STYLE100_URL = "https://zenodo.org/records/4687771/files/100STYLE_BVH.zip"


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
                if total:
                    pct = downloaded / int(total) * 100
                    if downloaded % (1024 * 1024) < 8192:
                        logger.info("Progress: %.1f%%", pct)

        tmp.rename(dest)
        logger.info("Downloaded: %s", dest.name)
        return True

    except Exception:
        logger.exception("Failed to download %s", url)
        return False


def extract_zip(zip_path: Path, output_dir: Path, limit: int | None = None) -> list[Path]:
    bvh_files: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        bvh_names = [n for n in zf.namelist() if n.lower().endswith(".bvh")]
        if limit:
            bvh_names = bvh_names[:limit]

        for name in bvh_names:
            zf.extract(name, output_dir)
            bvh_files.append(output_dir / name)

    return bvh_files


def download_100style(output_dir: Path, limit: int | None = None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "100STYLE_BVH.zip"
    if not download_file(STYLE100_URL, zip_path):
        return []

    bvh_dir = output_dir / "bvh"
    extracted = extract_zip(zip_path, bvh_dir, limit)
    logger.info("Total 100STYLE BVH files: %d", len(extracted))
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Download 100STYLE BVH files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/100style"))
    parser.add_argument("--limit", type=int, default=None, help="Limit number of BVH files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_100style(args.output_dir, args.limit)


if __name__ == "__main__":
    main()
