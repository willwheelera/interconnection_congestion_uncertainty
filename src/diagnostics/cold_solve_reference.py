"""Cold-solve reference for the load-averaged congestion target.

Solves a configuration the slow, obvious way: deep-copy the case, inject the built
candidates, and call `solve_scopf` once per load snapshot from scratch, averaging each
constraint's mu over the collection.  No persistent solver, no warm start, no bound
toggling.

Everything real goes through `model.evaluator` instead, which builds each snapshot's
model once and re-solves it per configuration -- orders of magnitude faster and the only
way a 2^20 sweep is affordable.  This file exists to check that machinery: if a warm
re-solve ever disagreed with a cold solve, this is what says so.

Note it deliberately solves at the case JSON's own emergency rating with no ORDC relief
and no shadow-price cap, so its duals are *not* the sweep's duals.  To compare against a
sweep, apply the case recipe first (`model.relief.set_emergency_factor` and friends).

Usage:
    uv run python -m src.diagnostics.cold_solve_reference --case activsg2000_2025
    uv run python -m src.diagnostics.cold_solve_reference --case activsg2000_2025 --config 1 0 1 ...
"""

import argparse
import copy

from egret.data.model_data import ModelData
from egret.models.scopf import solve_scopf

from src.cases import registry
from src.model.relief import add_queue_generators, scale_loads_by_zone, scale_renewables_by_snapshot


def snapshot_mu(md_config, zone_scales, solver, renewable_cf=None):
    """Solve one zone-scaled N-1 SCDCOPF snapshot from scratch; return {constraint: mu}."""
    md = copy.deepcopy(md_config)
    scale_loads_by_zone(md, zone_scales)
    scale_renewables_by_snapshot(md, renewable_cf)
    md_sol, m = solve_scopf(md, solver=solver, solver_tee=False,
                            return_model=True, include_feasibility_slack=True)
    base_mva = md_sol.data["system"]["baseMVA"]
    mus = {}
    for bn, con in m.ineq_pf_branch_thermal_bounds.items():
        mu = abs(m.dual.get(con, 0.0) or 0.0) / base_mva
        if mu > 1e-6:
            mus[f"base:{bn}"] = mu
    for (cn, bn), con in m.ineq_pf_contingency_branch_thermal_bounds.items():
        mu = abs(m.dual.get(con, 0.0) or 0.0) / base_mva
        if mu > 1e-6:
            mus[f"ctg:{cn}:{bn}"] = mu
    return mus


def load_averaged_mu(md_base, candidates, snapshots, config, solver="gurobi"):
    """Average each constraint's mu over the snapshot collection for one configuration."""
    md_config = copy.deepcopy(md_base)
    add_queue_generators(md_config, candidates, config)
    totals = {}
    for s in snapshots:
        mus = snapshot_mu(md_config, s["zone_scales"], solver, s.get("renewable_cf"))
        for cid, mu in mus.items():
            totals[cid] = totals.get(cid, 0.0) + mu
    return {cid: v / len(snapshots) for cid, v in totals.items()}


def summarize(label, avg_mu):
    """Print the peak and the binding count for one load-averaged result."""
    if not avg_mu:
        print(f"{label:<10} no binding constraints")
        return
    top = max(avg_mu.items(), key=lambda kv: kv[1])
    n_base = sum(1 for k in avg_mu if k.startswith("base:"))
    n_ctg = sum(1 for k in avg_mu if k.startswith("ctg:"))
    print(f"{label:<10} max load-averaged mu {top[1]:8.2f} $/MWh at {top[0]}"
          f"   binding: {n_base} base, {n_ctg} contingency")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="a registered case name")
    parser.add_argument("--config", type=int, nargs="+")
    parser.add_argument("--solver", default="gurobi")
    args = parser.parse_args()

    case = registry.get(args.case)
    md_base = ModelData.read(str(case.case_file))
    candidates = case.candidates()
    snapshots = case.snapshots()

    if args.config is not None:
        summarize("config", load_averaged_mu(md_base, candidates, snapshots,
                                             args.config, args.solver))
    else:
        n = case.ngen
        summarize("base", load_averaged_mu(md_base, candidates, snapshots, [0] * n, args.solver))
        summarize("all-on", load_averaged_mu(md_base, candidates, snapshots, [1] * n, args.solver))


if __name__ == "__main__":
    main()
