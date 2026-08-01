"""Static descriptors (slack, swing sigma) per congestion constraint, from one base solve.

These are the cheap single-configuration predictors Claim 2 tests: if sampling is
replaceable, they should rank the lines the sampled tail ranks.  `descriptors.study`
scores them; this file produces them.

Two sources, chosen by case family.  A `wide` case (the gen-per-bus testbeds) already has
per-line flows and dispatch in its dual_cache, so the descriptors read straight out of the
all-on row.  A security-constrained case stores only mu, and its constraints are N-1 pairs
("base:{line}" or "ctg:{outage}:{line}") rather than plain lines, so this rebuilds the same
two descriptors from ONE SCDCOPF solve per load snapshot at a chosen base configuration,
through `model.scdcopf` so the recipe (emergency factor, ORDC relief, soft caps) matches
the sweep exactly:

  - slack = limit - |flow|, on the base flow against the long-term rating for a "base:"
    constraint and on the post-outage flow against the emergency rating for a "ctg:" one;
  - swing sigma, the std of that same flow under the 20 queue candidates switching on and
    off at their calibrated Bernoulli p, where candidate g coming online injects its
    (capacity-factor derated) capacity at its bus and the existing fleet backs down in
    proportion to its dispatch at the base configuration.

An incumbent-fleet case has a real grid at the queue-empty configuration, so the base
configuration is a choice: "off" builds nothing, "on" builds every candidate, "modal"
builds the most probable set (each candidate with p > 0.5).  A queue is unlikely to build
out -- E[candidates online] is 5.4 of 20 on Texas -- so "off" and "modal" carry nominal
probability mass and "on" carries almost none.

Descriptors are averaged over the case's load snapshots, matching the load averaging of the
mu target, and written with that configuration's load-averaged mu to
results/<case>/static_descriptors_<base>.parquet.

Usage:
    uv run python -m descriptors.compute --case activsg2000_2025 --base off
"""

import argparse

import numpy as np
import pandas as pd
from egret.data.model_data import ModelData
from egret.model_library.defn import BasePointType
from egret.model_library.transmission.tx_calc import calculate_ptdf

from cases import registry
from model.scdcopf import build_persistent

BASES = ["off", "on", "modal"]
P_ON = 0.65   # nominal on-marginal for a case with no queue; a common scalar on sigma


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


def snapshot_descriptors(case, snapshot, constraints, p, config):
    """Slack, swing sigma (MW), and mu for one snapshot at the base configuration."""
    pm = build_persistent(case, snapshot)
    mus = pm.solve(config)
    m, md, ptdf = pm.m, pm.md, pm.m._PTDF
    names, cap_pu, base_mva = pm.names, pm.cap_pu, pm.base_mva
    u = candidate_injections(md, ptdf, cap_pu, names, fleet_participation(m, md, ptdf, names))
    slack = constraint_slack(ptdf, m, constraints, base_mva)
    rows = np.array([constraint_row(ptdf, cid) for cid in constraints])
    df = base_mva * (rows @ u)                                  # (nconstraint x ncand) MW swing
    sigma = np.sqrt((p * (1.0 - p) * df ** 2).sum(axis=1))
    return pd.DataFrame({"slack": [slack[c] for c in constraints], "sigma": sigma,
                         "mu_base": [mus.get(c, 0.0) for c in constraints]}, index=constraints)


def build_scdcopf(case, constraints, base):
    """Average the per-snapshot descriptors over the case's snapshots; add z = slack/sigma."""
    p = case.build_p()
    config = base_config(base, p)
    frames = [snapshot_descriptors(case, s, constraints, p, config) for s in case.snapshots()]
    d = sum(frames) / len(frames)
    d["z"] = d.slack / np.maximum(d.sigma, 1e-9)
    return d[["slack", "sigma", "z", "mu_base"]]


def build_wide(case):
    """Per-line slack, swing sigma, z, and base mu from a wide case's all-on dual_cache row.

    No solve: the enumerated cache already holds the all-on flows and dispatch, and the
    swing is the PTDF response to each generator dropping out with the rest of the fleet
    picking up its output in proportion to their own.
    """
    md = ModelData.read(str(case.case_file))
    els = md.data["elements"]
    buses, branches, gens = els["bus"], els["branch"], els["generator"]
    bus_keys, branch_keys = list(buses), list(branches)
    bi = {b: i for i, b in enumerate(bus_keys)}
    ptdf = calculate_ptdf(branches, buses, branch_keys, bus_keys,
                          md.data["system"]["reference_bus"],
                          base_point=BasePointType.FLATSTART)
    ratings = np.array([float(branches[b]["rating_long_term"]) for b in branch_keys])
    g_bus = np.array([bi[g["bus"]] for g in gens.values()])

    dc = pd.read_parquet(case.dual_cache)
    row = dc.loc[dc.index.max()]                                 # the all-on configuration
    flow = np.array([row[("flow", b)] for b in branch_keys])
    mu_base = np.abs([row[("mu", b)] for b in branch_keys])
    pg = np.array([row[("p_gen", bus_keys[b])] for b in g_bus])

    Hg = ptdf[:, g_bus]
    tot = pg.sum()
    swing = ((Hg @ pg)[:, None] - Hg * pg[None, :]) / np.maximum(tot - pg, 1e-9)[None, :]
    df = pg[None, :] * (swing - Hg)                              # (nline x ngen) MW flow swing
    sigma = np.sqrt(P_ON * (1 - P_ON) * (df ** 2).sum(axis=1))
    slack = ratings - np.abs(flow)
    return pd.DataFrame({"slack": slack, "sigma": sigma,
                         "z": slack / np.maximum(sigma, 1e-9), "mu_base": mu_base},
                        index=branch_keys)


def descriptors(case, base="off"):
    """The case's per-constraint descriptors, by whichever source its model calls for.

    A wide case is read straight out of its enumerated dual_cache, which is cheap.  A
    security-constrained case costs one SCDCOPF solve per snapshot, so it is read from the
    cache this module's `main` writes and errors rather than silently launching that job.
    """
    if case.model == "dcopf":
        return build_wide(case)
    cached = case.descriptor_cache(base)
    if not cached.exists():
        raise FileNotFoundError(
            f"{cached} not found; build it first with\n"
            f"    uv run python -m descriptors.compute --case {case.name} --base {base}")
    return pd.read_parquet(cached)


def main():
    parser = argparse.ArgumentParser()
    registry.add_case_arg(parser)
    parser.add_argument("--base", default="off", choices=BASES)
    args = parser.parse_args()
    case = registry.get(args.case)
    if case.model == "dcopf":
        d = build_wide(case)
    else:
        constraints = list(pd.read_parquet(case.dual_cache)["mu"].columns)
        d = build_scdcopf(case, constraints, args.base)
    dest = case.descriptor_cache(args.base)
    d.to_parquet(dest)
    print(d.round(2).to_string())
    print("wrote", dest)


if __name__ == "__main__":
    main()
