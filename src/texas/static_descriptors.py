"""Static descriptors (slack, swing sigma) per congestion constraint of a texas case.

The generic descriptor path in src/mu_static_descriptors.py reads flows and dispatch out
of a wide dual_cache, which the texas sweep does not store: its cache holds only mu, and
its constraints are N-1 pairs ("base:{line}" or "ctg:{outage}:{line}"), not plain lines.
This rebuilds the same two descriptors from ONE SCDCOPF solve per load snapshot at a chosen
base configuration, using the harness's own model so the recipe (emergency factor, ORDC
relief, soft caps) matches the sweep exactly:

  - slack = limit - |flow|, on the base flow against the long-term rating for a "base:"
    constraint and on the post-outage flow against the emergency rating for a "ctg:" one;
  - swing sigma, the std of that same flow under the 20 queue candidates switching on and
    off at their calibrated Bernoulli p, where candidate g coming online injects its
    (capacity-factor derated) capacity at its bus and the existing fleet backs down in
    proportion to its dispatch at the base configuration.

Unlike case14/case30, this case has an incumbent fleet, so the queue-empty configuration is
itself a real grid and the base configuration is a choice: "off" builds nothing, "on" builds
every candidate, "modal" builds the most probable set (each candidate with p > 0.5).  The
queue is unlikely to build out -- E[candidates online] is 5.4 of 20 -- so "off" and "modal"
carry nominal probability mass and "on" carries almost none.

Both descriptors are averaged over the load snapshots, matching the load averaging of the mu
target, and written with that configuration's load-averaged mu to
results/<case>/static_descriptors_<base>.parquet.

Usage:
    uv run python src/texas/static_descriptors.py --base off      # HARNESS_CASE selects the case
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "texas"))

from sample_harness import (CASE, COLLECTION, QUEUE, RESULTS_NAME, build_persistent,
                            extract_mu, solve_config)

RESULTS = REPO_ROOT / "results" / RESULTS_NAME
BASES = ["off", "on", "modal"]


def base_config(base, p):
    """The chosen base configuration as a 0/1 list over the queue candidates."""
    if base == "off":
        return [0] * len(p)
    if base == "on":
        return [1] * len(p)
    return [int(pi > 0.5) for pi in p]


def parse_constraint(cid):
    """("base:L", "ctg:C:L") -> (outaged contingency or None, monitored line)."""
    parts = cid.split(":")
    return (None, parts[1]) if parts[0] == "base" else (parts[1], parts[2])


def constraint_row(ptdf, cid):
    """Full bus-indexed sensitivity of the constraint's flow to bus injection."""
    ctg, line = parse_constraint(cid)
    row = (ptdf._get_ptdf_row(line) if ctg is None
           else ptdf._get_contingency_row(ctg, line))
    return ptdf._insert_reference_bus(row, 0.0)


def constraint_slack(ptdf, m, constraints, base_mva):
    """Limit minus |flow| (MW) per constraint at the current solve; negative if overloaded.

    Base constraints use the long-term rating and the base flow; contingency constraints
    use the emergency rating and the post-outage flow for their own outage."""
    pfv, _, va = ptdf.calculate_PFV(m)
    idx = ptdf._branchname_to_index_map
    post = {}
    slack = {}
    for cid in constraints:
        ctg, line = parse_constraint(cid)
        i = idx[line]
        if ctg is None:
            flow, limit = pfv[i], ptdf.branch_limits_array[i]
        else:
            if ctg not in post:
                post[ctg] = pfv + ptdf.calculate_PFV_delta(ctg, pfv, va)
            flow, limit = post[ctg][i], ptdf.contingency_limits_array[i]
        slack[cid] = base_mva * (limit - abs(flow))
    return slack


def fleet_participation(m, md, ptdf, names):
    """Bus-indexed shares of the incumbent fleet's all-on dispatch (the backdown weights).

    Candidate and load-shed units are excluded: they are not what backs down when a queue
    project comes online."""
    gens = md.data["elements"]["generator"]
    bus_index = {b: i for i, b in enumerate(ptdf.buses_keys)}
    weight = np.zeros(len(ptdf.buses_keys))
    skip = set(names)
    for name in m.pg:
        if name in skip or str(name).startswith("lostload_"):
            continue
        weight[bus_index[gens[name]["bus"]]] += max(m.pg[name].value, 0.0)
    return weight / weight.sum()


def candidate_injections(md, ptdf, cap_pu, names, gamma):
    """(nbus x ncand) net bus injection when each candidate comes online, per unit.

    Its derated capacity lands at its own bus and the incumbent fleet gives up the same
    amount in proportion to gamma."""
    gens = md.data["elements"]["generator"]
    bus_index = {b: i for i, b in enumerate(ptdf.buses_keys)}
    u = np.zeros((len(ptdf.buses_keys), len(names)))
    for j, name in enumerate(names):
        u[:, j] = -cap_pu[name] * gamma
        u[bus_index[gens[name]["bus"]], j] += cap_pu[name]
    return u


def snapshot_descriptors(snapshot, constraints, candidates, p, config):
    """Slack, swing sigma (MW), and mu for one snapshot at the base configuration."""
    m, md, solver, cap_pu, names, base_mva = build_persistent(snapshot, str(CASE), candidates)
    mus = solve_config(m, md, solver, config, cap_pu, names, base_mva)
    ptdf = m._PTDF
    u = candidate_injections(md, ptdf, cap_pu, names, fleet_participation(m, md, ptdf, names))
    slack = constraint_slack(ptdf, m, constraints, base_mva)
    rows = np.array([constraint_row(ptdf, cid) for cid in constraints])
    df = base_mva * (rows @ u)                                  # (nconstraint x ncand) MW swing
    sigma = np.sqrt((p * (1.0 - p) * df ** 2).sum(axis=1))
    return pd.DataFrame({"slack": [slack[c] for c in constraints], "sigma": sigma,
                         "mu_base": [mus.get(c, 0.0) for c in constraints]}, index=constraints)


def build(constraints, p, base):
    """Average the per-snapshot descriptors over the load collection; add z = slack/sigma."""
    candidates = json.loads(QUEUE.read_text())["candidates"]
    snapshots = json.loads(COLLECTION.read_text())["snapshots"]
    config = base_config(base, p)
    frames = [snapshot_descriptors(s, constraints, candidates, p, config) for s in snapshots]
    d = sum(frames) / len(frames)
    d["z"] = d.slack / np.maximum(d.sigma, 1e-9)
    return d[["slack", "sigma", "z", "mu_base"]]


def main():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tail_sampling.distributions import queue_p
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="off", choices=BASES)
    args = parser.parse_args()
    constraints = list(pd.read_parquet(RESULTS / "dual_cache.parquet")["mu"].columns)
    d = build(constraints, queue_p(), args.base)
    dest = RESULTS / f"static_descriptors_{args.base}.parquet"
    d.to_parquet(dest)
    print(d.round(2).to_string())
    print("wrote", dest)


if __name__ == "__main__":
    main()
