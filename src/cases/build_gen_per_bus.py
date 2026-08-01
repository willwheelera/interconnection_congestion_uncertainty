"""Build a larger, deliberately congested gen-per-bus case for tail-sampling tests.

The 14-bus model is too small: its high-mu tail has only a few dozen configs, so a
sampler that evaluates a few percent of the space can nearly enumerate it.  This
builds a 30-bus model with a generator at each of the 20 load buses (-> 20 binary
availability bits, 2^20 ~= 1.05M configs, still enumerable offline for exact ground
truth) on the standard IEEE-30 topology.

Two design choices create a richer congestion landscape than the 14-bus case:
  - generation is local to each load pocket, so dropping a pocket's generator forces
    imports across the limited tie-lines feeding it;
  - generator costs vary by region (a spatial merit order), so which pocket is cheap
    to serve --- and therefore which corridor congests --- shifts with the config.
Different unavailable-generator patterns light up different binding lines, giving the
multiple congestion paths and multimodal high-mu tail the sampler needs to stress.

Branch ratings are scaled down by RATING_SCALE to make congestion common without
making typical configs infeasible.

Usage:
    uv run python src/build_congested_case.py
"""

import json
from pathlib import Path
import numpy as np
import pandapower.networks as nw
from pandapower.converter import to_ppc

from paths import REPO_ROOT
OUT_PATH = REPO_ROOT / "data" / "case30_gen_per_bus.json"

RATING_SCALE = 0.5            # shrink line ratings to induce congestion
COST_GROUPS = [18.0, 26.0, 34.0, 42.0]   # $/MWh base cost by spatial quartile
GEN_PMAX = 55.0              # MW capacity per generator (total >> 189 MW load)


def ieee30_topology():
    """Return (bus_ids, branch list, load dict) from the IEEE-30 per-unit data."""
    ppc = to_ppc(nw.case30(), init="flat")
    bus, br = ppc["bus"], ppc["branch"]
    ids = [str(int(b)) for b in bus[:, 0]]
    branches = [
        (str(int(f)), str(int(t)), float(x), float(rate))
        for f, t, x, rate in zip(br[:, 0], br[:, 1], br[:, 3], br[:, 5])
    ]
    loads = {str(int(b)): float(pd) for b, pd in zip(bus[:, 0], bus[:, 2]) if pd > 0}
    return ids, branches, loads


def bus_element(is_ref):
    d = {"matpower_bustype": "ref" if is_ref else "PQ", "area": 1.0,
         "vm": 1.0, "va": 0.0, "zone": 1.0, "v_min": 0.94, "v_max": 1.06}
    return d


def branch_element(fr, to, reactance, rating):
    return {"from_bus": fr, "to_bus": to, "resistance": 0.0, "reactance": reactance,
            "charging_susceptance": 0.0, "branch_type": "line",
            "rating_long_term": rating * RATING_SCALE, "rating_short_term": None,
            "rating_emergency": None, "in_service": True,
            "angle_diff_min": -360.0, "angle_diff_max": 360.0,
            "pf": None, "qf": None, "pt": None, "qt": None}


def generator_element(bus, cost):
    """Linear-cost thermal generator at a load bus; p_min 0 so it is fully flexible."""
    return {"bus": bus, "pg": 0.0, "qg": 0.0, "vg": 1.0, "mbase": 100.0,
            "in_service": True, "p_min": 0.0, "p_max": GEN_PMAX,
            "q_min": 0.0, "q_max": 0.0, "generator_type": "thermal",
            "p_cost": {"data_type": "cost_curve", "cost_curve_type": "polynomial",
                       "values": {"0": 0.0, "1": cost, "2": 0.0}},
            "startup_cost": 0.0, "shutdown_cost": 0.0}


def assign_costs(gen_buses):
    """Spatial merit order: split the gen buses into quartiles by id, cost per group."""
    order = np.argsort([int(b) for b in gen_buses])
    costs = {}
    for rank, idx in enumerate(order):
        group = min(rank * len(COST_GROUPS) // len(gen_buses), len(COST_GROUPS) - 1)
        costs[gen_buses[idx]] = COST_GROUPS[group] + 0.5 * (rank % 3)
    return costs


def build():
    ids, branches, loads = ieee30_topology()
    ref = ids[0]
    gen_buses = list(loads.keys())                      # one generator per load bus
    costs = assign_costs(gen_buses)

    elements = {
        "bus": {b: bus_element(b == ref) for b in ids},
        "branch": {str(i + 1): branch_element(*br) for i, br in enumerate(branches)},
        "generator": {f"G{i + 1}": generator_element(b, costs[b])
                      for i, b in enumerate(gen_buses)},
        "load": {f"load_{b}": {"in_service": True, "p_load": p, "q_load": 0.0, "bus": b}
                 for b, p in loads.items()},
    }
    system = {"model_name": "case30_gen_per_bus", "baseMVA": 100.0,
              "load_mismatch_cost": 200.0, "reserve_shortfall_cost": 200.0,
              "reference_bus": ref, "reference_bus_angle": 0.0}

    md = {"elements": elements, "system": system}
    OUT_PATH.write_text(json.dumps(md))
    print(f"wrote {OUT_PATH}")
    print(f"  buses {len(ids)}  branches {len(branches)}  gens {len(gen_buses)} "
          f"(2^{len(gen_buses)} = {1 << len(gen_buses):,} configs)  "
          f"load {sum(loads.values()):.0f} MW  gen cap {GEN_PMAX * len(gen_buses):.0f} MW")


if __name__ == "__main__":
    build()
