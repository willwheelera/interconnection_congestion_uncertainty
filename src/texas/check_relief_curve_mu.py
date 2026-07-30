"""Compare flat vs ORDC-shaped relief and check whether mu spreads across a range.

A relief generator with one fixed marginal cost pins a shed pocket's price to that
single level, so the congestion dual mu is bimodal: generation-set values, then a
spike. The ORDC-shaped relief (ordc_relief_curve) instead prices shed depth
convexly from c_min to c_max, so pockets can settle at a range of prices. This runs
the certified-worst all-off config on every snapshot at a tightened emergency
factor (where relief actually dispatches) under each relief scheme and reports the
load-averaged mu distribution, so we can see whether the rising curve produces the
few-hundred-to-several-thousand spread observed in ERCOT LMP data.

Usage:
    uv run python src/texas/check_relief_curve_mu.py
    uv run python src/texas/check_relief_curve_mu.py --emergency-factor 1.0 --blocks 8
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
from egret.data.model_data import ModelData
from egret.models.scopf import create_scopf_model
from egret.models.dcopf import _lazy_ptdf_dcopf_model_solve_loop
from egret.common.solver_interface import _solve_model

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "texas"))

from load_averaged_target import (add_queue_generators, add_lostload_generators,
                                   add_lostload_generators_ordc, scale_loads_by_zone)
from soft_limits import enable_soft_limits, thermal_violations

CASE = REPO_ROOT / "data" / "activsg2000_screened.json"
QUEUE = REPO_ROOT / "data" / "activsg2000_queue20.json"
COLLECTION = REPO_ROOT / "data" / "activsg2000_load_collection.json"

C_MIN = 500.0            # $/MWh floor: where the ORDC adder becomes material
C_MAX = 5000.0           # $/MWh VOLL = current ERCOT system-wide offer cap
SHADOW_PRICE_CAP = 3500.0  # $/MWh ERCOT Max Shadow Price for a transmission constraint
MU_TOL = 1e-6            # $/MWh; ignore numerically non-binding constraints


def set_emergency_factor(md, factor):
    """Override every branch's post-contingency emergency rating to factor x long-term."""
    for b in md.data["elements"]["branch"].values():
        b["rating_emergency"] = factor * b["rating_long_term"]


def build(snapshot, candidates, relief, emergency_factor, n_blocks, soft_cap=None):
    """Build all-off SCDCOPF for one snapshot; optionally cap thermal duals at soft_cap."""
    md = ModelData.read(str(CASE))
    set_emergency_factor(md, emergency_factor)
    add_queue_generators(md, candidates, [1] * len(candidates))
    scale_loads_by_zone(md, snapshot["zone_scales"])
    if relief == "flat":
        add_lostload_generators(md, C_MAX)
    elif relief == "ordc":
        add_lostload_generators_ordc(md, C_MIN, C_MAX, n_blocks)
    m, md_scaled = create_scopf_model(md, include_feasibility_slack=True)
    base_mva = md.data["system"]["baseMVA"]
    if soft_cap is not None:
        enable_soft_limits(m, soft_cap, base_mva)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solver = pyo.SolverFactory("gurobi_persistent")
    solver.options["OutputFlag"] = 0
    solver.set_instance(m)
    for c in candidates:
        m.pg[f"queue_{c['inr']}"].setub(0.0)
        solver.update_var(m.pg[f"queue_{c['inr']}"])
    return m, md_scaled, solver, base_mva


def solve(m, md, solver):
    """Warm N-1 SCDCOPF solve (base solve + lazy contingency loop)."""
    _solve_model(m, solver, solver_tee=False, return_solver=True, set_instance=False)
    _lazy_ptdf_dcopf_model_solve_loop(m, md, solver, solver_tee=False)


def snapshot_mu(m, base_mva):
    """{constraint_id: mu} for all binding base and post-contingency thermal limits."""
    mus = {}
    for bn, con in m.ineq_pf_branch_thermal_bounds.items():
        mu = abs(m.dual.get(con, 0.0) or 0.0) / base_mva
        if mu > MU_TOL:
            mus[f"base:{bn}"] = mu
    for (cn, bn), con in m.ineq_pf_contingency_branch_thermal_bounds.items():
        mu = abs(m.dual.get(con, 0.0) or 0.0) / base_mva
        if mu > MU_TOL:
            mus[f"ctg:{cn}:{bn}"] = mu
    return mus


def load_averaged_mu(snapshots, candidates, relief, emergency_factor, n_blocks, soft_cap=None):
    """Solve every snapshot; return (load-averaged mu, per-snapshot overload MW dicts)."""
    totals, viols = {}, []
    for s in snapshots:
        m, md, solver, base_mva = build(s, candidates, relief, emergency_factor, n_blocks, soft_cap)
        solve(m, md, solver)
        for cid, mu in snapshot_mu(m, base_mva).items():
            totals[cid] = totals.get(cid, 0.0) + mu
        viols.append(thermal_violations(m, base_mva) if soft_cap is not None else {})
    return {cid: v / len(snapshots) for cid, v in totals.items()}, viols


def report(label, avg_mu):
    """Print the mu distribution: quantiles, tail counts, and the top constraints."""
    vals = np.array(sorted(avg_mu.values()))
    print(f"\n{label}: {len(vals)} binding constraints")
    if len(vals) == 0:
        return
    for q in (0.5, 0.9, 0.99, 1.0):
        print(f"  q{q:<4} {np.quantile(vals, q):9.1f} $/MWh")
    for t in (300, 1000, 3000):
        print(f"  # constraints with load-averaged mu > {t:>4}: {(vals > t).sum()}")
    print("  top constraints:")
    for cid, mu in sorted(avg_mu.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {mu:9.1f} $/MWh  {cid}")


def report_violations(viols, emergency_factor, ratings):
    """Per snapshot, list overloaded constraints as MW and percent of their thermal limit."""
    for i, v in enumerate(viols):
        if not v:
            print(f"  snapshot {i + 1}: no thermal violations")
            continue
        print(f"  snapshot {i + 1}: {len(v)} constraint(s) overloaded")
        for cid, mw in sorted(v.items(), key=lambda kv: -kv[1])[:8]:
            bn = cid.split(":")[-1]
            limit = ratings[bn] * (1.0 if cid.startswith("base:") else emergency_factor)
            print(f"    {mw:8.2f} MW  ({100 * mw / limit:5.1f}% of {limit:.0f} MW limit)  {cid}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emergency-factor", type=float, default=1.0)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--snapshots", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    candidates = json.loads(QUEUE.read_text())["candidates"]
    snapshots = json.loads(COLLECTION.read_text())["snapshots"]
    if args.snapshots:
        snapshots = snapshots[:args.snapshots]
    ratings = {bn: b["rating_long_term"]
               for bn, b in json.loads(CASE.read_text())["elements"]["branch"].items()}

    print(f"all-off, emergency factor {args.emergency_factor}, {len(snapshots)} snapshots, "
          f"ORDC curve {C_MIN:.0f}->{C_MAX:.0f} over {args.blocks} blocks")
    avg_nocap, _ = load_averaged_mu(snapshots, candidates, "ordc", args.emergency_factor, args.blocks)
    report("ordc, hard limits (no cap)", avg_nocap)
    avg_cap, viols = load_averaged_mu(snapshots, candidates, "ordc", args.emergency_factor,
                                      args.blocks, soft_cap=SHADOW_PRICE_CAP)
    report(f"ordc + shadow-price cap ${SHADOW_PRICE_CAP:.0f}", avg_cap)
    print(f"\nthermal-limit violations bought by the ${SHADOW_PRICE_CAP:.0f} cap:")
    report_violations(viols, args.emergency_factor, ratings)


if __name__ == "__main__":
    main()
