"""Build IEEE queue testbeds: the original fleet as incumbents, a queue added on top.

The existing gen-per-bus testbeds (case14_gen_per_bus, case30_gen_per_bus) make every
generator a queue candidate, so the all-off configuration is an empty grid served
entirely by priced load shed -- an artifact rather than a grid.  These cases instead keep
the IEEE network's own generators permanently online as the incumbent fleet and add
`ncand` interconnection-queue candidates on top, matching the Texas setting where the
queue perturbs a real fleet.

Construction, applied identically to case14 / case30 / case118:

  - incumbents keep their bus and capacity; their quadratic cost is replaced by the
    linear marginal cost at full output, rank-mapped into COST_BAND so all three cases
    share one $/MWh scale (and therefore one shadow-price cap and one shed price);
  - load is scaled so total demand is LOAD_MARGIN of incumbent capacity, which puts the
    fleet near fully committed: a candidate that undercuts the marginal incumbent
    displaces it, so most queue bits change the dispatch instead of sitting idle;
  - candidate sites are chosen by greedy farthest-point sampling on the reactance-weighted
    network graph, seeded from the incumbent generator buses, so candidates land remote
    from existing generation and spread out -- the export-corridor geometry that produces
    distinct congestion modes;
  - candidate costs are linear and spread across the same COST_BAND as the incumbents,
    assigned to sites in a seeded random permutation so cost carries no spatial pattern;
  - branch ratings are provisional here: each line is rated at its incumbent-only flow
    plus base_margin headroom, the way real lines are sized, so a corridor absorbs only
    that much extra flow before congesting.  That covers the base case but says nothing
    about N-1, so calibrate_ratings.py follows and relaxes ratings until the incumbent-only
    grid is secure and only mildly priced -- leaving queue buildout as what drives the
    severe congestion the sweep measures;
  - an N-1 single-branch-outage contingency is added for every non-bridge branch.

Writes two files per case: the EGRET case (data/<case>.json) and its queue
(data/<case>_candidates.json), whose per-candidate build probabilities are the sampling law.

Usage:
    uv run python -m src.cases.build_ieee_queue
    uv run python -m src.cases.build_ieee_queue --case case30_queue
"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandapower.networks as nw
from egret.data.model_data import ModelData
from egret.models.dcopf import solve_dcopf
from pandapower.converter import to_ppc

from src.cases import registry

# The pandapower network, queue size, and rating headroom over the incumbent-only flows
# are per-case knobs and live in each case's registry `build` entry; everything below is
# shared across the three cases.
COST_BAND = (10.0, 150.0)       # $/MWh span shared by incumbents and candidates
LOAD_MARGIN = 0.85              # total load as a fraction of incumbent capacity
CAP_FRAC = 0.10                 # candidate capacity as a fraction of total load
CAP_SPREAD = (0.6, 1.4)         # multiplicative spread of candidate capacities
P_RANGE = (0.2, 0.8)            # per-candidate build probability range
MIN_RATING_FRAC = 0.05          # rating floor, as a fraction of the mean flow envelope
N_STRESS = 16                   # half-loaded stress configs behind the flow envelope
BIG_RATING = 1e5                # effectively unlimited rating for the envelope pass
LOAD_MISMATCH_COST = 9000.0     # system-wide slack price, above the nodal shed curve
SEED = 0


def ieee_tables(network):
    """(bus ids, reference bus, branches, loads, incumbents) from a pandapower network.

    branches: (from, to, reactance pu, rate MVA); loads: {bus: MW};
    incumbents: (bus, p_max MW, quadratic c2, linear c1)."""
    ppc = to_ppc(getattr(nw, network)(), init="flat")
    bus, br, gen, cost = ppc["bus"], ppc["branch"], ppc["gen"], ppc["gencost"]
    ids = [str(int(b)) for b in bus[:, 0].real]
    ref = str(int(bus[bus[:, 1].real == 3][0, 0].real))
    branches = [(str(int(f.real)), str(int(t.real)), float(x.real), float(r.real))
                for f, t, x, r in zip(br[:, 0], br[:, 1], br[:, 3], br[:, 5])]
    loads = {str(int(b.real)): float(p.real)
             for b, p in zip(bus[:, 0], bus[:, 2]) if p.real > 0}
    incumbents = [(str(int(g[0].real)), float(g[8].real), float(c[4]), float(c[5]))
                  for g, c in zip(gen, cost)]
    return ids, ref, branches, loads, incumbents


def incumbent_costs(incumbents):
    """Linear marginal cost per incumbent, rank-mapped onto COST_BAND.

    The IEEE quadratic cost at full output orders the fleet; the map onto a shared band
    puts all three cases on one $/MWh scale so a single shadow-price cap applies."""
    raw = np.array([c1 + c2 * p_max for _, p_max, c2, c1 in incumbents])
    lo, hi = raw.min(), raw.max()
    return COST_BAND[0] + (COST_BAND[1] - COST_BAND[0]) * (raw - lo) / (hi - lo)


def scaled_loads(loads, incumbents):
    """Loads scaled so total demand is LOAD_MARGIN of total incumbent capacity."""
    capacity = sum(p_max for _, p_max, _, _ in incumbents)
    scale = LOAD_MARGIN * capacity / sum(loads.values())
    return {b: p * scale for b, p in loads.items()}


def network_graph(branches):
    """Undirected graph with reactance as edge length (parallel circuits keep the min)."""
    graph = nx.Graph()
    for fr, to, x, _ in branches:
        if graph.has_edge(fr, to):
            graph[fr][to]["weight"] = min(graph[fr][to]["weight"], abs(x))
        else:
            graph.add_edge(fr, to, weight=abs(x))
    return graph


def spread_sites(graph, seeds, n):
    """Greedy farthest-point sampling: n buses maximizing distance from seeds and each other.

    Distances are reactance-weighted shortest paths, so the picks are electrically remote
    from existing generation and from one another -- candidates behind different corridors."""
    dist = dict(nx.all_pairs_dijkstra_path_length(graph))
    covered = list(seeds)
    sites = []
    for _ in range(n):
        far = max((b for b in graph if b not in sites),
                  key=lambda b: (min(dist[b][c] for c in covered), b))
        sites.append(far)
        covered.append(far)
    return sites


def make_candidates(sites, total_load, rng):
    """Queue records: bus, capacity, linear cost spread over COST_BAND, build probability."""
    n = len(sites)
    caps = np.linspace(*CAP_SPREAD, n) * CAP_FRAC * total_load
    costs = COST_BAND[0] + (COST_BAND[1] - COST_BAND[0]) * (np.arange(n) + 0.5) / n
    order = rng.permutation(n)                      # cost carries no spatial pattern
    probs = rng.uniform(*P_RANGE, n)
    return [{"id": f"Q{i + 1:02d}", "bus": sites[i], "capacity_mw": round(caps[i], 1),
             "cost": round(float(costs[order[i]]), 2), "p": round(float(probs[i]), 3)}
            for i in range(n)]


def linear_cost(marginal):
    """Polynomial cost curve carrying only the linear ($/MWh) coefficient."""
    return {"data_type": "cost_curve", "cost_curve_type": "polynomial",
            "values": {"0": 0.0, "1": float(marginal), "2": 0.0}}


def base_elements(ids, ref, branches, loads, incumbents, costs):
    """Element dict with unlimited ratings; ratings are calibrated in a second pass."""
    return {
        "bus": {b: {"matpower_bustype": "ref" if b == ref else "PQ", "area": 1.0,
                    "vm": 1.0, "va": 0.0, "zone": 1.0, "v_min": 0.94, "v_max": 1.06}
                for b in ids},
        "branch": {str(i + 1): {"from_bus": fr, "to_bus": to, "resistance": 0.0,
                                "reactance": x, "charging_susceptance": 0.0,
                                "branch_type": "line", "rating_long_term": BIG_RATING,
                                "rating_short_term": None, "rating_emergency": None,
                                "in_service": True, "angle_diff_min": -360.0,
                                "angle_diff_max": 360.0, "pf": None, "qf": None,
                                "pt": None, "qt": None}
                   for i, (fr, to, x, _) in enumerate(branches)},
        "generator": {f"inc_{i + 1}": {"bus": bus, "pg": 0.0, "qg": 0.0, "vg": 1.0,
                                       "mbase": 100.0, "in_service": True, "p_min": 0.0,
                                       "p_max": p_max, "q_min": 0.0, "q_max": 0.0,
                                       "generator_type": "thermal",
                                       "p_cost": linear_cost(cost),
                                       "startup_cost": 0.0, "shutdown_cost": 0.0}
                      for i, ((bus, p_max, _, _), cost) in enumerate(zip(incumbents, costs))},
        "load": {f"load_{b}": {"in_service": True, "p_load": p, "q_load": 0.0, "bus": b}
                 for b, p in loads.items()},
    }


def add_queue_generators(md, candidates, config):
    """Inject the candidates whose bit is 1 as linear-cost generators at their buses."""
    gens = md.data["elements"]["generator"]
    for on, c in zip(config, candidates):
        if not on:
            continue
        gens[f"queue_{c['id']}"] = {
            "bus": c["bus"], "generator_type": "thermal", "in_service": True,
            "p_min": 0.0, "p_max": float(c["capacity_mw"]), "pg": 0.0,
            "p_cost": linear_cost(c["cost"]),
        }


def stress_configs(n, rng):
    """All-off, all-on, and N_STRESS half-loaded configurations."""
    configs = [[0] * n, [1] * n]
    for _ in range(N_STRESS):
        on = set(rng.choice(n, n // 2, replace=False).tolist())
        configs.append([1 if j in on else 0 for j in range(n)])
    return configs


def flow_envelope(elements, system, candidates, configs, solver):
    """Per-branch |flow| envelope over the stress configs, and the all-off flows (MW).

    configs[0] must be the all-off (incumbent-only) configuration."""
    envelope, off = {}, {}
    for k, config in enumerate(configs):
        md = ModelData({"elements": copy.deepcopy(elements), "system": dict(system)})
        add_queue_generators(md, candidates, config)
        md_sol, _ = solve_dcopf(md, solver=solver, solver_tee=False, return_model=True,
                                include_feasibility_slack=True)
        for bn, br in md_sol.data["elements"]["branch"].items():
            flow = abs(float(br["pf"]))
            envelope[bn] = max(envelope.get(bn, 0.0), flow)
            if k == 0:
                off[bn] = flow
    return envelope, off


def apply_ratings(elements, envelope, off, base_margin):
    """Rate each branch to its incumbent-only flow plus base_margin headroom.

    Ratings track what the existing fleet requires, the way real lines are sized, so a
    corridor absorbs only base_margin of extra flow before it congests and the queue --
    not the incumbent dispatch -- is what pushes it there.  The stress-flow envelope only
    supplies a floor, keeping lines that carry almost nothing today from congesting on any
    perturbation at all."""
    floor = MIN_RATING_FRAC * float(np.mean(list(envelope.values())))
    for bn, br in elements["branch"].items():
        br["rating_long_term"] = max(floor, base_margin * off[bn])


def branch_contingencies(elements):
    """One single-branch-outage contingency per non-bridge branch.

    A bridge outage islands load rather than posing a security constraint, and EGRET
    rejects it, so those branches are skipped."""
    branches = elements["branch"]
    graph = nx.Graph()
    pairs = Counter()
    for b in branches.values():
        pairs[frozenset((b["from_bus"], b["to_bus"]))] += 1
        graph.add_edge(b["from_bus"], b["to_bus"])
    bridges = {frozenset(e) for e in nx.bridges(graph)}
    return {name: {"branch_contingency": name} for name, b in branches.items()
            if not (pairs[frozenset((b["from_bus"], b["to_bus"]))] == 1
                    and frozenset((b["from_bus"], b["to_bus"])) in bridges)}


def build(case, solver="gurobi"):
    """Build one queue testbed and write its case and queue files."""
    name, spec = case.name, case.build
    rng = np.random.default_rng(SEED)
    ids, ref, branches, loads, incumbents = ieee_tables(spec["network"])
    loads = scaled_loads(loads, incumbents)
    costs = incumbent_costs(incumbents)
    graph = network_graph(branches)
    sites = spread_sites(graph, sorted({b for b, _, _, _ in incumbents}), spec["ncand"])
    candidates = make_candidates(sites, sum(loads.values()), rng)

    elements = base_elements(ids, ref, branches, loads, incumbents, costs)
    system = {"model_name": name, "baseMVA": 100.0,
              "load_mismatch_cost": LOAD_MISMATCH_COST,
              "reserve_shortfall_cost": LOAD_MISMATCH_COST,
              "reference_bus": ref, "reference_bus_angle": 0.0}
    configs = stress_configs(len(candidates), rng)
    envelope, off = flow_envelope(elements, system, candidates, configs, solver)
    apply_ratings(elements, envelope, off, spec["base_margin"])
    elements["contingency"] = branch_contingencies(elements)

    case.case_file.write_text(json.dumps({"elements": elements, "system": system}))
    case.paths.get("candidates", registry.DATA / f"{name}_candidates.json").write_text(
        json.dumps({"seed": SEED, "network": spec["network"], "candidates": candidates},
                   indent=2))
    report(name, elements, candidates, costs, loads)


def report(name, elements, candidates, costs, loads):
    """Print the built case's size, cost range, and queue scale."""
    caps = [c["capacity_mw"] for c in candidates]
    print(f"{name}: buses {len(elements['bus'])}  branches {len(elements['branch'])}  "
          f"contingencies {len(elements['contingency'])}  incumbents {len(costs)}  "
          f"candidates {len(candidates)} (2^{len(candidates)} = {1 << len(candidates):,})")
    print(f"  load {sum(loads.values()):.0f} MW   incumbent cap "
          f"{sum(g['p_max'] for g in elements['generator'].values()):.0f} MW   "
          f"queue cap {sum(caps):.0f} MW ({min(caps):.0f}-{max(caps):.0f} each)")
    print(f"  incumbent cost {costs.min():.1f}-{costs.max():.1f} $/MWh   candidate cost "
          f"{min(c['cost'] for c in candidates):.1f}-{max(c['cost'] for c in candidates):.1f} "
          f"$/MWh   E[built] {sum(c['p'] for c in candidates):.1f}")
    ratings = [b["rating_long_term"] for b in elements["branch"].values()]
    print(f"  ratings {min(ratings):.1f}-{max(ratings):.1f} MW")


def main():
    parser = argparse.ArgumentParser()
    registry.add_case_arg(parser, default="all", allow_all=True)
    parser.add_argument("--solver", default="gurobi")
    args = parser.parse_args()
    for case in registry.resolve(args.case, family="ieee_queue"):
        build(case, args.solver)


if __name__ == "__main__":
    main()
