"""Pre-compute per-config sensitivities into a parquet cache.

For each cached OPF result (results/<case>/opf_*_avail<id>.json.gz) this
computes the barrier-regularised sensitivities and stores the full
*per-line, per-load-bus* Jacobian:

    S_{l,j}(x) = d(mu_l) / d(load_j)  evaluated at the OPF solution of x

where mu_l is the congestion dual of line l (upper plus lower limit
combined) and load_j is the active-power load at bus j.

The barrier eps parameters (EPS_MU = EPS_X = 1.0 in sensitivity_analysis)
keep the sensitivity finite at exact transitions.

Output: results/<case>/sensitivity_cache.parquet, indexed by avail_id with
a 2-level column MultiIndex (branch, bus).  Values are signed.

Usage:
    uv run python src/build_sensitivity_cache.py
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import re

from egret.data.model_data import ModelData

from sensitivity.barrier import compute_sensitivity_barrier, _read_pf_dual_dict
from utils.solution_io import get_lmps
EPS_MU, EPS_X = 1.0, 1.0

from paths import REPO_ROOT
DEFAULT_CASE = REPO_ROOT / "data" / "case14_gen_per_bus.json"
DEFAULT_DIR = REPO_ROOT / "results" / "case14_gen_per_bus"


def sensitivity_matrix(md: ModelData) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute the signed d(mu_l)/d(load_j) matrix for one config.

    Returns (S, branch_names, bus_names) with S shape (n_branch, n_bus).
    """
    lmp_base = get_lmps(md)
    pf_dual = _read_pf_dual_dict(md)
    _, mu_sens = compute_sensitivity_barrier(
        md, md, lmp_base, pf_dual, eps_mu=EPS_MU, eps_x=EPS_X
    )
    bu = mu_sens["branch_upper"]
    bl = mu_sens["branch_lower"]
    dmudL = bu.values + bl.values
    branch_names = list(bu.index)
    bus_names = list(bu.columns)
    return dmudL, branch_names, bus_names


def _worker(path: Path) -> tuple[int, np.ndarray, list[str], list[str]]:
    """Process pool entry: read one OPF result and return its sensitivity row."""
    stem = path.name.replace(".json.gz", "")
    avail_id = int(stem.split("avail")[1])
    md = ModelData.read(str(path))
    S, branch_names, bus_names = sensitivity_matrix(md)
    return avail_id, S.reshape(-1), branch_names, bus_names


def build_cache(
    case_dir: Path = DEFAULT_DIR,
    case_path: Path = DEFAULT_CASE,
    max_workers: int | None = None,
) -> Path:
    """Iterate over opf_*_avail<id>.json.gz files and build the sensitivity cache.

    Uses a process pool across `max_workers` workers (default: os.cpu_count()).
    """
    out_path = case_dir / "sensitivity_cache.parquet"
    paths = case_dir.glob("opf_*_avail*.json.gz")
    paths = [p for p in paths if not re.search(r"_scale", p.name)]
    paths = sorted(paths)
    n = len(paths)
    workers = max_workers or os.cpu_count() or 1
    print(f"Found {n} OPF results; using {workers} workers")

    rows: dict[int, np.ndarray] = {}
    col_index: pd.MultiIndex | None = None
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, p): p for p in paths}
        for i, fut in enumerate(as_completed(futures), start=1):
            avail_id, row, branch_names, bus_names = fut.result()
            if col_index is None:
                col_index = pd.MultiIndex.from_product(
                    [branch_names, bus_names], names=["branch", "bus"]
                )
            rows[avail_id] = row
            if i % 500 == 0 or i == n:
                dt = time.perf_counter() - t0
                print(f"  {i}/{n}  elapsed {dt:.1f}s ({i / dt:.0f}/s)")

    df = pd.DataFrame.from_dict(rows, orient="index", columns=col_index)
    df.index.name = "avail_id"
    df = df.sort_index()
    df.to_parquet(out_path)
    print(f"Wrote {out_path}  shape={df.shape}  total {time.perf_counter() - t0:.1f}s")
    return out_path


if __name__ == "__main__":
    build_cache()
