"""Measure how much each queue candidate actually produces at the sweep's recipe.

The 2^20 sweep found that 11 of the 20 candidates move the target by less than
0.01 $/MWh, including 1,536 MW of fully-available gas. Two explanations are possible:
the unit is out of merit and never dispatches, or it dispatches but sits electrically
where it cannot shift any binding constraint. This distinguishes them by solving the
all-on configuration once per load snapshot and reading each candidate's pg.

Usage:
    HARNESS_CASE=2025 uv run python src/texas/check_queue_dispatch.py
"""

import json
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
from egret.data.model_data import ModelData
from egret.models.scopf import create_scopf_model

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "texas"))

from sample_harness import (CASE, QUEUE, COLLECTION, RESULTS_NAME, EMERGENCY_FACTOR,
                            ORDC_C_MIN, ORDC_C_MAX, ORDC_BLOCKS, SHADOW_PRICE_CAP,
                            SOLVER_THREADS, RELIEF_BUSES, set_emergency_factor)
from load_averaged_target import (add_queue_generators, add_lostload_generators_ordc,
                                  scale_loads_by_zone, scale_renewables_by_snapshot)
from soft_limits import enable_soft_limits
from egret.models.dcopf import _lazy_ptdf_dcopf_model_solve_loop
from egret.common.solver_interface import _solve_model

OUT = REPO_ROOT / "results" / RESULTS_NAME / "queue_dispatch.parquet"


def solve_snapshot(snapshot, candidates):
    """All-on SCDCOPF for one snapshot; return each candidate's dispatch in MW."""
    cf = snapshot.get("renewable_cf") or {}
    md = ModelData.read(str(CASE))
    set_emergency_factor(md, EMERGENCY_FACTOR)
    add_queue_generators(md, candidates, [1] * len(candidates))
    scale_loads_by_zone(md, snapshot["zone_scales"])
    scale_renewables_by_snapshot(md, cf)
    add_lostload_generators_ordc(md, ORDC_C_MIN, ORDC_C_MAX, ORDC_BLOCKS,
                                 buses=RELIEF_BUSES)
    m, md_scaled = create_scopf_model(md, include_feasibility_slack=True)
    base_mva = md.data["system"]["baseMVA"]
    enable_soft_limits(m, SHADOW_PRICE_CAP, base_mva)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solver = pyo.SolverFactory("gurobi_persistent")
    solver.options["OutputFlag"] = 0
    if SOLVER_THREADS:
        solver.options["Threads"] = SOLVER_THREADS
    solver.set_instance(m)
    _solve_model(m, solver, solver_tee=False, return_solver=True, set_instance=False)
    _lazy_ptdf_dcopf_model_solve_loop(m, md_scaled, solver, solver_tee=False)
    return {c["inr"]: (m.pg[f"queue_{c['inr']}"].value or 0.0) * base_mva
            for c in candidates if f"queue_{c['inr']}" in m.pg}


def worker(snapshot, candidates, q):
    """Solve one pinned snapshot and post its dispatch dict."""
    import logging
    logging.getLogger("egret").setLevel(logging.WARNING)
    q.put((snapshot["timestamp"], solve_snapshot(snapshot, candidates)))


def report(df, candidates):
    """Print per-candidate mean dispatch against nameplate and available capacity."""
    by_inr = {c["inr"]: c for c in candidates}
    rows = []
    for inr, col in df.items():
        c = by_inr[inr]
        rows.append(dict(name=c["name"][:28], fuel=c["fuel"], p=c["p"],
                         plate=c["capacity_mw"], mean_mw=col.mean(),
                         max_mw=col.max(), n_zero=int((col < 0.5).sum())))
    out = pd.DataFrame(rows).sort_values("plate", ascending=False)
    out["util"] = out.mean_mw / out.plate
    print(out.to_string(index=False, float_format=lambda v: f"{v:8.1f}"))
    print(f"\ntotal nameplate {out.plate.sum():.0f} MW   "
          f"mean dispatched {out.mean_mw.sum():.0f} MW   "
          f"({out.mean_mw.sum() / out.plate.sum():.1%})")


def main():
    candidates = json.loads(Path(QUEUE).read_text())["candidates"]
    snapshots = json.loads(Path(COLLECTION).read_text())["snapshots"]
    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(s, candidates, q)) for s in snapshots]
    for p in procs:
        p.start()
    rows = {ts: disp for ts, disp in (q.get() for _ in procs)}
    for p in procs:
        p.join()
    df = pd.DataFrame(rows).T.sort_index()
    df.to_parquet(OUT)
    report(df, candidates)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
