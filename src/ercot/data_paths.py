"""Where the fetched ERCOT datasets live.

`src/ercot/fetch_ercot.py` and friends download into `data/ercot/`, which is gitignored:
the files are large and re-downloadable.  Every reader in this package resolves its input
through here rather than building a path relative to its own location.
"""

from pathlib import Path

from paths import REPO_ROOT
ERCOT_DATA = REPO_ROOT / "data" / "ercot"
RESULTS = REPO_ROOT / "results"

SHADOW_PRICES = ERCOT_DATA / "shadow_prices_2025.parquet"
SAMPLE_DAYS_CSV = ERCOT_DATA / "sample_days.csv"
REAL_LMP_AVERAGE = ERCOT_DATA / "real_lmp_time_average.parquet"
