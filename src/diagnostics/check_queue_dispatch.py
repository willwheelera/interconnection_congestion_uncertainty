"""Measure how much each queue candidate actually produces at the sweep's recipe.

The 2^20 Texas sweep found that 11 of the 20 candidates move the target by less than
0.01 $/MWh, including 1,536 MW of fully-available gas.  Two explanations are possible:
the unit is out of merit and never dispatches, or it dispatches but sits electrically
where it cannot shift any binding constraint.  This distinguishes them by solving the
all-on configuration once per load snapshot and reading each candidate's pg.

Usage:
    uv run python -m diagnostics.check_queue_dispatch --case activsg2000_2025
"""

import argparse
import logging
import multiprocessing as mp

import pandas as pd

from cases import registry
from model.scdcopf import build_persistent


def solve_snapshot(case, snapshot):
    """All-on SCDCOPF for one snapshot; return each candidate's dispatch in MW."""
    pm = build_persistent(case, snapshot)
    pm.solve([1] * case.ngen)
    return {c["id"]: (pm.m.pg[f"queue_{c['id']}"].value or 0.0) * pm.base_mva
            for c in case.candidates() if f"queue_{c['id']}" in pm.m.pg}


def worker(case, snapshot, q):
    """Solve one pinned snapshot and post its dispatch dict."""
    logging.getLogger("egret").setLevel(logging.WARNING)
    q.put((snapshot["timestamp"], solve_snapshot(case, snapshot)))


def report(df, candidates):
    """Print per-candidate mean dispatch against nameplate and available capacity."""
    by_id = {c["id"]: c for c in candidates}
    rows = []
    for cid, col in df.items():
        c = by_id[cid]
        rows.append(dict(name=(c["name"] or cid)[:28], fuel=c["fuel"], p=c["p"],
                         plate=c["capacity_mw"], mean_mw=col.mean(),
                         max_mw=col.max(), n_zero=int((col < 0.5).sum())))
    out = pd.DataFrame(rows).sort_values("plate", ascending=False)
    out["util"] = out.mean_mw / out.plate
    print(out.to_string(index=False, float_format=lambda v: f"{v:8.1f}"))
    print(f"\ntotal nameplate {out.plate.sum():.0f} MW   "
          f"mean dispatched {out.mean_mw.sum():.0f} MW   "
          f"({out.mean_mw.sum() / out.plate.sum():.1%})")


def main():
    parser = argparse.ArgumentParser()
    registry.add_case_arg(parser)
    case = registry.get(parser.parse_args().case)
    out = case.results_dir / "queue_dispatch.parquet"

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(case, s, q)) for s in case.snapshots()]
    for p in procs:
        p.start()
    rows = {ts: disp for ts, disp in (q.get() for _ in procs)}
    for p in procs:
        p.join()

    df = pd.DataFrame(rows).T.sort_index()
    df.to_parquet(out)
    report(df, case.candidates())
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
