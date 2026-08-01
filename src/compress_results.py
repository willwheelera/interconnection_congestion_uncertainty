"""Convert existing uc_*.json result files to compressed .json.gz format.

Walks results/ recursively, compresses each uc_*.json file in place
(writes .json.gz, then removes the original .json).

Usage:
    uv run python src/compress_results.py
"""

import gzip
import shutil
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def compress_json(path: Path) -> Path:
    """Gzip-compress a JSON file and remove the original."""
    gz_path = path.with_suffix(".json.gz")
    with open(path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    path.unlink()
    return gz_path


def main():
    paths = sorted(RESULTS_DIR.rglob("uc_*.json"))
    print(f"Found {len(paths)} files to compress")
    total_orig = 0
    total_gz = 0
    for p in paths:
        orig_size = p.stat().st_size
        gz = compress_json(p)
        gz_size = gz.stat().st_size
        total_orig += orig_size
        total_gz += gz_size
    print(f"Compressed {len(paths)} files: {total_orig/1e6:.1f} MB -> {total_gz/1e6:.1f} MB ({total_gz/total_orig:.1%})")


if __name__ == "__main__":
    main()
