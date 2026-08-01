"""Time the all-off SCDCOPF with universal nodal relief and certify the relief set.

All-off (no queue generators online) is the most transmission-constrained config: its
feasible set is contained in every on-config's, so the buses forced to shed load under
all-off are a certified superset of every bus that can ever need relief across the 2^20
space (turning a generator on only enlarges the feasible set). This solves all-off with a
relief generator at EVERY load bus (buses=None) on each snapshot and reports, per snapshot
and in aggregate:

  - build, cold-solve, and warm-resolve times WITH vs WITHOUT the relief columns, so the
    real slowdown from universal placement is measured rather than assumed, and
  - the union over snapshots of relief units that actually dispatch -- the necessary
    relief set to prune to -- written to results/activsg2000/relief_set.json.

Relief cost sits just below load_mismatch_cost (the system-wide slack penalty) so nodal
relief is preferred over EGRET's single non-locational slack, letting a locked pocket
price up locally.

Usage:
    uv run python src/texas/identify_relief_set.py
    uv run python src/texas/identify_relief_set.py --snapshots 2   # quick timing subset
"""

import argparse
import json
import time
from pathlib import Path

import pyomo.environ as pyo
from egret.data.model_data import ModelData
from egret.models.scopf import create_scopf_model
from egret.models.dcopf import _lazy_ptdf_dcopf_model_solve_loop
from egret.common.solver_interface import _solve_model

from paths import REPO_ROOT

from model.relief import add_queue_generators, add_lostload_generators, scale_loads_by_zone

CASE = REPO_ROOT / "data" / "activsg2000_screened.json"
QUEUE = REPO_ROOT / "data" / "activsg2000_queue20.json"
COLLECTION = REPO_ROOT / "data" / "activsg2000_load_collection.json"
OUT = REPO_ROOT / "results" / "activsg2000" / "relief_set.json"

RELIEF_COST = 8900.0   # just below load_mismatch_cost=9000 so nodal relief beats the slack
DISPATCH_TOL = 1e-4    # pu; ~0.01 MW on a 100 MVA base


def set_emergency_factor(md, factor):
    """Override every branch's post-contingency emergency rating to factor x long-term."""
    for b in md.data["elements"]["branch"].values():
        b["rating_emergency"] = factor * b["rating_long_term"]


def build(snapshot, candidates, with_relief, emergency_factor=None):
    """Build the all-off SCDCOPF for one snapshot; return (model, md, solver, base_mva)."""
    md = ModelData.read(str(CASE))
    if emergency_factor is not None:
        set_emergency_factor(md, emergency_factor)
    add_queue_generators(md, candidates, [1] * len(candidates))   # present but toggled off below
    scale_loads_by_zone(md, snapshot["zone_scales"])
    if with_relief:
        add_lostload_generators(md, RELIEF_COST)                  # buses=None -> every load bus
    m, md_scaled = create_scopf_model(md, include_feasibility_slack=True)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solver = pyo.SolverFactory("gurobi_persistent")
    solver.options["OutputFlag"] = 0
    solver.set_instance(m)
    for c in candidates:                                          # all-off: queue capacity -> 0
        m.pg[f"queue_{c['inr']}"].setub(0.0)
        solver.update_var(m.pg[f"queue_{c['inr']}"])
    return m, md_scaled, solver, md.data["system"]["baseMVA"]


def solve(m, md, solver):
    """Warm N-1 SCDCOPF solve (base solve + lazy contingency loop)."""
    _solve_model(m, solver, solver_tee=False, return_solver=True, set_instance=False)
    _lazy_ptdf_dcopf_model_solve_loop(m, md, solver, solver_tee=False)


def dispatched_relief(m, base_mva):
    """{bus: shed_MW} for relief generators dispatching above tolerance."""
    shed = {}
    for name in m.pg:
        if name.startswith("lostload_") and (m.pg[name].value or 0.0) > DISPATCH_TOL:
            shed[name] = (m.pg[name].value or 0.0) * base_mva
    return shed


def run_snapshot(snapshot, candidates, with_relief, emergency_factor=None):
    """Build + cold-solve + warm-resolve all-off once; return timings and relief shed."""
    t0 = time.time()
    m, md, solver, base_mva = build(snapshot, candidates, with_relief, emergency_factor)
    build_t = time.time() - t0
    t0 = time.time()
    solve(m, md, solver)
    cold_t = time.time() - t0
    t0 = time.time()
    solve(m, md, solver)
    warm_t = time.time() - t0
    shed = dispatched_relief(m, base_mva) if with_relief else {}
    glob = (m.p_load_shed.value or 0.0) * base_mva
    return dict(build=build_t, cold=cold_t, warm=warm_t, shed=shed, global_shed=glob)


def report(rows_relief, rows_base):
    """Print paired timing and the aggregate slowdown from the relief columns."""

    def tot(rows, key):
        return sum(r[key] for r in rows)

    print(f"\n{'':12}{'build':>9}{'cold':>9}{'warm':>9}   (sum over "
          f"{len(rows_relief)} snapshots, s)")
    print(f"{'no relief':12}{tot(rows_base, 'build'):9.1f}{tot(rows_base, 'cold'):9.1f}"
          f"{tot(rows_base, 'warm'):9.2f}")
    print(f"{'relief(all)':12}{tot(rows_relief, 'build'):9.1f}{tot(rows_relief, 'cold'):9.1f}"
          f"{tot(rows_relief, 'warm'):9.2f}")
    for key in ("build", "cold", "warm"):
        b, r = tot(rows_base, key), tot(rows_relief, key)
        print(f"  {key:5} slowdown from relief columns: {(r / b - 1) * 100:+5.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=int, default=0,
                        help="limit to the first N snapshots (0 = all, needed for the set)")
    parser.add_argument("--emergency-factor", type=float, default=None,
                        help="override post-contingency limits to F x rating_long_term "
                             "(case ships 1.5; lower F tightens N-1 and forces relief)")
    args = parser.parse_args()

    candidates = json.loads(QUEUE.read_text())["candidates"]
    snapshots = json.loads(COLLECTION.read_text())["snapshots"]
    if args.snapshots:
        snapshots = snapshots[:args.snapshots]

    rows_base, rows_relief, union = [], [], {}
    for i, s in enumerate(snapshots):
        print(f"snapshot {i + 1}/{len(snapshots)}  {s['timestamp']}", flush=True)
        rb = run_snapshot(s, candidates, with_relief=False, emergency_factor=args.emergency_factor)
        rr = run_snapshot(s, candidates, with_relief=True, emergency_factor=args.emergency_factor)
        rows_base.append(rb)
        rows_relief.append(rr)
        for name, mw in rr["shed"].items():
            union[name] = max(union.get(name, 0.0), mw)
        print(f"  relief units dispatched: {len(rr['shed'])}  "
              f"(system slack {rr['global_shed']:.1f} MW)", flush=True)

    report(rows_relief, rows_base)

    print(f"\nnecessary relief set: {len(union)} buses shed under all-off "
          f"(of {len(json.loads(CASE.read_text())['elements']['load'])} load buses)")
    for name, mw in sorted(union.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {name:28} {mw:9.1f} MW")
    OUT.write_text(json.dumps({"relief_generators": sorted(union),
                               "peak_shed_mw": union}, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
