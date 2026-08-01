from utils.solution_io import build_dual_cache
from pathlib import Path
import time

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

file = RESULTS_DIR / "case14_gen_per_bus" / "dual_cache.parquet"


if __name__ == "__main__":
    t0 = time.perf_counter()
    try:
        build_dual_cache(file)
    finally:
        print(f"time {time.perf_counter()-t0:.3f}")

