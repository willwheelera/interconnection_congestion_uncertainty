"""Joint builder for dual_cache.parquet and sensitivity_cache.parquet.

For each cached OPF solution under *case_dir* this opens the gzipped JSON
exactly once, walks the parsed dict to produce the dual-cache row, wraps
that same dict as a ModelData for the barrier-regularised KKT sensitivity
solve, and emits both rows.  The two parquets are written together with a
single shared sort, so they have identical row indices by construction --
positional row-axis stacking downstream is safe.

Replaces the split build path (solution_io.build_dual_cache and
build_sensitivity_cache.build_cache) whose independent sort policies let
the two row indices drift apart.

Output paths (under *case_dir*):
    dual_cache.parquet         columns: 2-level (group, name)
        groups: lmp, va, mu, nu, flow, p_gen, load, mu_pmax, mu_pmin
        - mu_pmax / mu_pmin are keyed by generator name and come from the
          KKT stationarity identity  mu_pmax - mu_pmin = LMP - dC/dPg
          with complementary slackness against the pg bounds.
    sensitivity_cache.parquet  columns: 2-level (branch, bus)
        values: d(mu_l) / d(load_j) at the OPF solution of each row

Usage:
    uv run python code/utils/build_caches.py
"""

from __future__ import annotations

import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from egret.data.model_data import ModelData

from sensitivity_analysis import compute_sensitivity_barrier
from utils.solution_io import _row_id_from_stem, _solution_files

EPS_MU, EPS_X = 1.0, 1.0
REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DIR = REPO_ROOT / "results" / "case14_gen_per_bus"


def _marginal_cost(p_cost: dict, pg: float) -> float:
    """Evaluate dC/dPg at pg for an EGRET p_cost dict.

    Handles piecewise-linear curves (slope of the active segment) and
    polynomial curves (analytic derivative).  JSON deserialization turns
    polynomial coefficient keys into strings, so cast before using them.
    """
    curve_type = p_cost["cost_curve_type"]
    values = p_cost["values"]
    if curve_type == "piecewise":
        for (p0, c0), (p1, c1) in zip(values, values[1:]):
            if pg <= p1:
                return (c1 - c0) / (p1 - p0)
        (p0, c0), (p1, c1) = values[-2], values[-1]
        return (c1 - c0) / (p1 - p0)
    if curve_type == "polynomial":
        return sum(
            int(k) * float(a) * pg ** (int(k) - 1)
            for k, a in values.items()
            if int(k) >= 1
        )
    raise ValueError(f"Unsupported cost_curve_type: {curve_type}")


def _generator_bound_duals(
    raw: dict,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute mu_pmax and mu_pmin per generator from KKT stationarity.

    For a generator with cost C(pg), KKT gives mu_pmax - mu_pmin = LMP - dC/dPg,
    and complementary slackness makes only one side active depending on which
    bound (if any) pg is on.  Interior pg leaves both at zero.
    """
    buses = raw["elements"]["bus"]
    mu_pmax: dict[str, float] = {}
    mu_pmin: dict[str, float] = {}
    for g_name, g in raw["elements"]["generator"].items():
        pg = float(g["pg"])
        p_max = float(g["p_max"])
        p_min = float(g["p_min"])
        lmp = float(buses[g["bus"]]["lmp"])
        slack = lmp - _marginal_cost(g["p_cost"], pg)
        atol = 1e-4 * max(1.0, abs(p_max))
        at_max = abs(pg - p_max) < atol
        at_min = abs(pg - p_min) < atol
        mu_pmax[g_name] = max(0.0,  slack) if at_max else 0.0
        mu_pmin[g_name] = max(0.0, -slack) if at_min else 0.0
    return mu_pmax, mu_pmin


def _extract_blocks(
    raw: dict, row_id,
) -> tuple[dict, dict[str, float], dict[str, tuple[float, float]]]:
    """Walk the parsed solution dict once, producing everything both sides need.

    Returns
    -------
    dual_block
        Dict keyed by ("group", name) plus "row_id"; becomes one row of
        dual_cache.parquet.  Groups: lmp, va, mu, nu, flow, p_gen, load.
    lmp_base
        ``{bus: lmp}`` input for ``compute_sensitivity_barrier``.
    pf_dual_dict
        ``{branch: (mu_upper, mu_lower)}`` input for ``compute_sensitivity_barrier``.
    """
    dual_block = {"row_id": row_id}
    lmp_base: dict[str, float] = {}
    pf_dual_dict: dict[str, tuple[float, float]] = {}

    for name, bus in raw["elements"]["bus"].items():
        if "lmp" in bus:
            lmp = float(bus["lmp"])
            dual_block[("lmp", name)] = lmp
            lmp_base[name] = lmp
        if "va" in bus:
            dual_block[("va", name)] = float(bus["va"])
        dual_block[("p_gen", name)] = 0.0
        dual_block[("load", name)] = 0.0

    for name, br in raw["elements"]["branch"].items():
        mu_u, mu_l = br.get("mu_upper"), br.get("mu_lower")
        pf_dual_dict[name] = (float(mu_u or 0.0), float(mu_l or 0.0))
        if mu_u is not None and mu_l is not None:
            dual_block[("mu", name)] = float(mu_u) + float(mu_l)
        if "nu" in br:
            dual_block[("nu", name)] = float(br["nu"])
        if "pf" in br:
            dual_block[("flow", name)] = float(br["pf"])

    for g in raw["elements"]["generator"].values():
        dual_block[("p_gen", g["bus"])] += float(g.get("pg", 0.0))
    for ld in raw["elements"]["load"].values():
        dual_block[("load", ld["bus"])] += float(ld["p_load"])

    mu_pmax, mu_pmin = _generator_bound_duals(raw)
    for g_name, val in mu_pmax.items():
        dual_block[("mu_pmax", g_name)] = val
    for g_name, val in mu_pmin.items():
        dual_block[("mu_pmin", g_name)] = val

    return dual_block, lmp_base, pf_dual_dict


def _worker(path: Path) -> tuple[dict, object, np.ndarray, list[str], list[str]]:
    """Per-file work: one disk read, one JSON parse, one KKT solve.

    The parsed raw dict feeds both the dual extraction and ModelData, so
    the solution file is touched only once.
    """
    row_id = _row_id_from_stem(path.stem.replace(".json", ""))
    with gzip.open(path, "rt") as f:
        raw = json.load(f)

    dual_block, lmp_base, pf_dual_dict = _extract_blocks(raw, row_id)

    md = ModelData(source=raw)
    _, mu_sens = compute_sensitivity_barrier(
        md, md, lmp_base, pf_dual_dict, eps_mu=EPS_MU, eps_x=EPS_X,
    )
    bu, bl = mu_sens["branch_upper"], mu_sens["branch_lower"]
    S = (bu.values + bl.values).reshape(-1)
    return dual_block, row_id, S, list(bu.index), list(bu.columns)


def _finalize_duals(blocks: list[dict], index_name: str) -> pd.DataFrame:
    """Stack per-file dual blocks into the dual_cache DataFrame, sorted both axes."""
    df = pd.DataFrame(blocks).set_index("row_id")
    df.index.name = index_name
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df.sort_index().sort_index(axis=1, level=0)


def _finalize_sens(
    rows: dict, branch_names: list[str], bus_names: list[str], index_name: str,
) -> pd.DataFrame:
    """Stack per-file sensitivity rows into the sens_cache DataFrame, row-sorted."""
    cols = pd.MultiIndex.from_product(
        [branch_names, bus_names], names=["branch", "bus"]
    )
    df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    df.index.name = index_name
    return df.sort_index()


def build_both_caches(
    case_dir: Path = DEFAULT_DIR,
    scan: str = "availability",
    max_workers: int | None = None,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Walk all solution files once and write both caches with aligned indices.

    Reads solution files from *case_dir*; writes parquets into *out_dir* if
    given, otherwise into *case_dir*.  Splitting them lets a consistency test
    write to a temp directory while leaving the trusted caches untouched.
    """
    out_dir = out_dir or case_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dual_path = out_dir / "dual_cache.parquet"
    sens_path = out_dir / "sensitivity_cache.parquet"
    paths = _solution_files(case_dir, scan)
    n = len(paths)
    workers = max_workers or os.cpu_count() or 1
    index_name = "load_scale" if scan == "load" else "avail_id"
    print(f"Found {n} solution files in {case_dir}; using {workers} workers")

    dual_blocks: list[dict] = []
    sens_rows: dict = {}
    branch_names: list[str] | None = None
    bus_names: list[str] | None = None
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, p) for p in paths]
        for i, fut in enumerate(as_completed(futures), start=1):
            dual_block, row_id, S, br, bu = fut.result()
            dual_blocks.append(dual_block)
            sens_rows[row_id] = S
            if branch_names is None:
                branch_names, bus_names = br, bu
            if i % 500 == 0 or i == n:
                dt = time.perf_counter() - t0
                print(f"  {i}/{n}  elapsed {dt:.1f}s ({i / dt:.0f}/s)", flush=True)

    dual_df = _finalize_duals(dual_blocks, index_name)
    sens_df = _finalize_sens(sens_rows, branch_names, bus_names, index_name)
    assert dual_df.index.equals(sens_df.index), \
        "dual and sens row indices disagree -- joint-build invariant violated"

    dual_df.to_parquet(dual_path)
    sens_df.to_parquet(sens_path)
    print(f"Wrote {dual_path}  shape={dual_df.shape}")
    print(f"Wrote {sens_path}  shape={sens_df.shape}")
    print(f"Total {time.perf_counter() - t0:.1f}s")
    return dual_path, sens_path


if __name__ == "__main__":
    build_both_caches()
