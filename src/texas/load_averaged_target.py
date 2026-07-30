"""Load-averaged congestion target over ACTIVSg2000 generator configurations.

Given a binary configuration of the 20 candidate queue generators (each entry: 1
if that project comes online, 0 otherwise), this evaluates the congestion target:

  1. inject the operational candidates as generators at their matched buses,
  2. for each load snapshot, scale each bus's load by its weather zone's snapshot
     scale factor and solve the N-1 security-constrained DCOPF, extracting the
     congestion dual mu for every binding base and post-contingency limit,
  3. average each constraint's mu over the snapshot collection.

The scalar sampling target is the maximum over constraints of the load-averaged mu.
Candidate marginal cost depends on fuel (see FUEL_COST): renewables and storage
are zero-cost (dispatch to nameplate), thermal units carry a realistic cost and
run only when economic.

Usage:
    uv run python src/texas/load_averaged_target.py            # base vs all-on
    uv run python src/texas/load_averaged_target.py --config 1 0 1 ...
"""

import argparse
import copy
import json
from pathlib import Path

from egret.data.model_data import ModelData
from egret.models.scopf import solve_scopf

REPO_ROOT = Path(__file__).parents[2]

CASE = REPO_ROOT / "data" / "activsg2000.json"
QUEUE = REPO_ROOT / "data" / "activsg2000_queue20.json"
COLLECTION = REPO_ROOT / "data" / "activsg2000_load_collection.json"

# Representative marginal cost ($/MWh) by queue fuel. Renewables and storage
# discharge are zero-cost; thermal costs sit above the existing fleet's ~$17-19
# so new thermal runs only when cheaper units are exhausted.
FUEL_COST = {"Solar": 0.0, "Wind": 0.0, "Battery": 0.0, "Hydro": 0.0,
             "Nuclear": 8.0, "Gas": 25.0, "Oil": 40.0, "Other": 30.0}


def linear_cost(marginal: float) -> dict:
    """Polynomial cost curve with the given linear ($/MWh) coefficient."""
    return {"data_type": "cost_curve", "cost_curve_type": "polynomial",
            "values": {"0": 0.0, "1": marginal, "2": 0.0}}


def add_queue_generators(md: ModelData, candidates: list[dict], config) -> None:
    """Inject operational candidate generators (config==1) into the ModelData."""
    gens = md.data["elements"]["generator"]
    for on, c in zip(config, candidates):
        if not on:
            continue
        gens[f"queue_{c['inr']}"] = {
            "bus": c["bus"], "generator_type": "thermal", "in_service": True,
            "p_min": 0.0, "p_max": float(c["capacity_mw"]), "pg": 0.0,
            "p_cost": linear_cost(FUEL_COST[c["fuel"]]),
            "fuel_group": c["fuel"],
        }


def add_lostload_generators(md: ModelData, cost: float, buses=None) -> None:
    """Add a high-cost relief generator at each load bus (or only `buses` if given).

    Emulates the nodal involuntary load shed that EGRET's single system-wide slack
    cannot represent: each unit can inject up to its bus's load at marginal cost
    `cost`, so a congested or post-contingency-insecure pocket is relieved locally
    instead of the whole solve going infeasible, and `cost` becomes that pocket's
    price ceiling (a static VOLL / scarcity proxy). Call AFTER scale_loads_by_zone so
    p_max tracks the scaled demand (the exact sheddable amount).
    """
    gens = md.data["elements"]["generator"]
    for name, load in md.data["elements"]["load"].items():
        if buses is not None and load["bus"] not in buses:
            continue
        gens[f"lostload_{name}"] = {
            "bus": load["bus"], "generator_type": "thermal", "in_service": True,
            "p_min": 0.0, "p_max": float(load["p_load"]), "pg": 0.0,
            "p_cost": linear_cost(cost),
        }


def ordc_relief_curve(cap: float, c_min: float, c_max: float, n_blocks: int) -> dict:
    """Convex piecewise shed-cost curve rising c_min -> c_max over [0, cap].

    Emulates the ERCOT ORDC scarcity shape: cheap for a shallow shed, climbing
    geometrically toward VOLL (c_max) for a deep one, so the marginal (last)
    block that clears sets a range of local prices instead of one fixed level.
    `values` are (MW, cumulative $) breakpoints; block slopes = marginal shed price.
    """
    edges = [cap * k / n_blocks for k in range(n_blocks + 1)]
    mc = [c_min * (c_max / c_min) ** (k / (n_blocks - 1)) for k in range(n_blocks)]
    values, cum = [(0.0, 0.0)], 0.0
    for k in range(n_blocks):
        cum += mc[k] * (edges[k + 1] - edges[k])
        values.append((edges[k + 1], cum))
    return {"data_type": "cost_curve", "cost_curve_type": "piecewise", "values": values}


def add_lostload_generators_ordc(md: ModelData, c_min: float, c_max: float,
                                 n_blocks: int, buses=None) -> None:
    """Relief generator at each load bus with an ORDC-shaped rising shed price.

    Same nodal-shed role as add_lostload_generators, but the flat `cost` is
    replaced by ordc_relief_curve so the pocket's price sweeps c_min..c_max with
    shed depth rather than pinning at one VOLL. Keep c_max below load_mismatch_cost
    so nodal relief still beats EGRET's system-wide slack. Call AFTER scale_loads_by_zone.
    """
    gens = md.data["elements"]["generator"]
    for name, load in md.data["elements"]["load"].items():
        if buses is not None and load["bus"] not in buses:
            continue
        cap = float(load["p_load"])
        if cap <= 0.0:
            continue
        gens[f"lostload_{name}"] = {
            "bus": load["bus"], "generator_type": "thermal", "in_service": True,
            "p_min": 0.0, "p_max": cap, "pg": 0.0,
            "p_cost": ordc_relief_curve(cap, c_min, c_max, n_blocks),
        }


def line_endpoint_buses(md: ModelData, lines) -> set:
    """From/to buses of the given branch names -- the pockets to target relief at."""
    branches = md.data["elements"]["branch"]
    return {b for ln in lines for b in (branches[ln]["from_bus"], branches[ln]["to_bus"])}


def scale_loads_by_zone(md: ModelData, zone_scales: dict[str, float]) -> None:
    """Scale each bus load by its weather zone's snapshot scale factor."""
    buses = md.data["elements"]["bus"]
    for load in md.data["elements"]["load"].values():
        load["p_load"] *= zone_scales[buses[load["bus"]]["zone"]]


def scale_renewables_by_snapshot(md: ModelData, cf: dict[str, float] | None) -> None:
    """Derate tagged renewable generators to the snapshot's per-fuel capacity factor.

    Acts only on generators carrying a `fuel_group` in `cf` (the CDR/queue
    renewables, added by add_cdr_resources); each such unit's nameplate p_max is
    scaled to cf[fuel] * p_max, tying solar/wind availability to the snapshot hour.
    Untagged base units and non-listed fuels (e.g. storage) are left unchanged.
    """
    if cf is None:
        return
    for gen in md.data["elements"]["generator"].values():
        group = gen.get("fuel_group")
        if group in cf:
            gen["p_max"] *= cf[group]


def snapshot_mu(md_config: ModelData, zone_scales: dict[str, float], solver: str,
                renewable_cf: dict[str, float] | None = None) -> dict[str, float]:
    """Solve one zone-scaled N-1 SCDCOPF snapshot; return {constraint_id: mu}.

    Constraint ids are "base:{line}" for normal thermal limits and
    "ctg:{contingency}:{line}" for post-contingency limits. Only binding
    constraints are returned; mu is the congestion dual in $/MWh. When
    renewable_cf is given, tagged renewables are derated to that hour's
    capacity factor before solving.
    """
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


def load_averaged_mu(md_base, candidates, snapshots, config, solver="gurobi") -> dict[str, float]:
    """Average each constraint's mu over the snapshot collection for a config."""
    md_config = copy.deepcopy(md_base)
    add_queue_generators(md_config, candidates, config)
    totals: dict[str, float] = {}
    for s in snapshots:
        mus = snapshot_mu(md_config, s["zone_scales"], solver, s.get("renewable_cf"))
        for cid, mu in mus.items():
            totals[cid] = totals.get(cid, 0.0) + mu
    return {cid: v / len(snapshots) for cid, v in totals.items()}


def summarize(label: str, avg_mu: dict[str, float]) -> None:
    """Print the max and count of binding constraints for a load-averaged result."""
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
    parser.add_argument("--config", type=int, nargs="+")
    parser.add_argument("--solver", type=str, default="gurobi")
    parser.add_argument("--case", type=str, default=str(CASE))
    args = parser.parse_args()

    md_base = ModelData.read(args.case)
    candidates = json.loads(QUEUE.read_text())["candidates"]
    snapshots = json.loads(COLLECTION.read_text())["snapshots"]

    if args.config is not None:
        cfg = args.config
        summarize("config", load_averaged_mu(md_base, candidates, snapshots, cfg, args.solver))
    else:
        n = len(candidates)
        summarize("base", load_averaged_mu(md_base, candidates, snapshots, [0] * n, args.solver))
        summarize("all-on", load_averaged_mu(md_base, candidates, snapshots, [1] * n, args.solver))


if __name__ == "__main__":
    main()
