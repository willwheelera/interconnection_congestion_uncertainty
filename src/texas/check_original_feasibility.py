"""Feasibility probe: original ACTIVSg2000 fleet at a tightened emergency factor.

The activsg2000_original congestion study was built on the pre-cap factor-1.5,
uncapped setup. This checks whether the same fleet stays feasible at the 2025
recipe's emergency factor (1.2) WITHOUT the ORDC relief / shadow-price cap: each
load snapshot's N-1 SCDCOPF is solved with only the system feasibility slack
(priced at load_mismatch_cost $9000/MWh) for the all-off and all-on queue
configs. Nonzero load shed, or a congestion dual pinned near the slack price,
means factor 1.2 needs the relief/cap recipe used for the 2025 case.

The 13-contingency screened set was screened under the old factor-1.5 setup; a
full redo at 1.2 would re-screen, but it is enough here to expose infeasibility.

Usage:
    uv run python src/texas/check_original_feasibility.py
    uv run python src/texas/check_original_feasibility.py --factor 1.2
"""

import argparse
import copy
import json
from pathlib import Path

import pyomo.environ as pyo
from egret.data.model_data import ModelData
from egret.models.scopf import solve_scopf

from load_averaged_target import (add_queue_generators, scale_loads_by_zone,
                                   scale_renewables_by_snapshot)

REPO_ROOT = Path(__file__).parents[2]
CASE = REPO_ROOT / "data" / "activsg2000_screened.json"
QUEUE = REPO_ROOT / "data" / "activsg2000_queue20.json"
COLLECTION = REPO_ROOT / "data" / "activsg2000_load_collection.json"
SOLVER = "gurobi"


def set_emergency_factor(md: ModelData, factor: float) -> None:
    """Override every branch's post-contingency emergency rating to factor x long-term."""
    for b in md.data["elements"]["branch"].values():
        b["rating_emergency"] = factor * b["rating_long_term"]


def extract_mu(m, base_mva: float) -> dict[str, float]:
    """{constraint_id: mu} for binding base and post-contingency thermal limits."""
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


def build_config_md(factor: float, config) -> ModelData:
    """Original screened case at emergency `factor` with the queue config applied."""
    md = ModelData.read(str(CASE))
    set_emergency_factor(md, factor)
    candidates = json.loads(QUEUE.read_text())["candidates"]
    add_queue_generators(md, candidates, config)
    return md


def solve_snapshot(md_config: ModelData, snapshot: dict):
    """Solve one snapshot N-1 SCDCOPF (feasibility slack only); return (shed_mw, mus)."""
    md = copy.deepcopy(md_config)
    scale_loads_by_zone(md, snapshot["zone_scales"])
    scale_renewables_by_snapshot(md, snapshot.get("renewable_cf"))
    md_sol, m = solve_scopf(md, solver=SOLVER, solver_tee=False,
                            return_model=True, include_feasibility_slack=True)
    base_mva = md_sol.data["system"]["baseMVA"]
    shed = float(pyo.value(m.p_load_shed)) * base_mva
    return shed, extract_mu(m, base_mva)


def probe(factor: float, config, label: str, snapshots: list) -> tuple[float, float]:
    """Print per-snapshot shed + uncapped max mu for one config; return (total_shed, peak_mu)."""
    print(f"\n--- {label}  (emergency factor {factor}) ---")
    total_shed, peak_mu = 0.0, 0.0
    md_config = build_config_md(factor, config)
    for s in snapshots:
        shed, mus = solve_snapshot(md_config, s)
        con, mu = max(mus.items(), key=lambda kv: kv[1], default=("none", 0.0))
        total_shed += shed
        peak_mu = max(peak_mu, mu)
        print(f"  {s['timestamp']:>20}  shed {shed:9.1f} MW   max mu {mu:9.1f} $/MWh  ({con})")
    print(f"  TOTAL shed {total_shed:9.1f} MW over {len(snapshots)} snapshots;  peak mu {peak_mu:.1f}")
    return total_shed, peak_mu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=float, default=1.2)
    args = parser.parse_args()
    snapshots = json.loads(COLLECTION.read_text())["snapshots"]
    n = len(json.loads(QUEUE.read_text())["candidates"])
    probe(args.factor, [0] * n, "all-off", snapshots)
    probe(args.factor, [1] * n, "all-on", snapshots)


if __name__ == "__main__":
    main()
