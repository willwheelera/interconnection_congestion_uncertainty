"""Sweep the capped congestion target S(x) across generator configurations.

The sampling target is S(x) = max over constraints of the load-averaged congestion
dual mu, with mu capped at the ERCOT Max Shadow Price (soft_limits). This evaluates
S(x) for a handful of configs -- all-off (certified worst), all-on, and Bernoulli
queue draws -- over the full snapshot collection, to check whether the target is a
usable sampler testbed: it must vary meaningfully over the config space rather than
collapse to all-zero or pin every config at the cap.

For each config it reports S(x), the argmax constraint, and cap_frac -- the fraction
of snapshots in which that top constraint sits exactly at the cap. cap_frac == 1.0
marks the flat plateau (S == cap, dS/dP == 0) where the sampler cannot rank configs;
0 < cap_frac < 1 is the informative regime where load-averaging grades the target.

Usage:
    uv run python src/texas/sweep_capped_target.py
    uv run python src/texas/sweep_capped_target.py --snapshots 2 --n-random 0
    uv run python src/texas/sweep_capped_target.py --emergency-factor 1.2 --n-random 6
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
from egret.data.model_data import ModelData
from egret.models.scopf import create_scopf_model

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "texas"))

from load_averaged_target import (add_queue_generators, add_lostload_generators_ordc,
                                   scale_loads_by_zone)
from soft_limits import enable_soft_limits
from check_relief_curve_mu import (set_emergency_factor, solve, snapshot_mu,
                                    C_MIN, C_MAX, SHADOW_PRICE_CAP)

CASE = REPO_ROOT / "data" / "activsg2000_screened.json"
QUEUE = REPO_ROOT / "data" / "activsg2000_queue20.json"
COLLECTION = REPO_ROOT / "data" / "activsg2000_load_collection.json"


def build_config(snapshot, candidates, config, emergency_factor, n_blocks, soft_cap):
    """Build one capped SCDCOPF snapshot with the given binary queue config."""
    md = ModelData.read(str(CASE))
    set_emergency_factor(md, emergency_factor)
    add_queue_generators(md, candidates, config)
    scale_loads_by_zone(md, snapshot["zone_scales"])
    add_lostload_generators_ordc(md, C_MIN, C_MAX, n_blocks)
    m, md_scaled = create_scopf_model(md, include_feasibility_slack=True)
    base_mva = md.data["system"]["baseMVA"]
    enable_soft_limits(m, soft_cap, base_mva)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solver = pyo.SolverFactory("gurobi_persistent")
    solver.options["OutputFlag"] = 0
    solver.set_instance(m)
    return m, md_scaled, solver, base_mva


def evaluate_config(snapshots, candidates, config, emergency_factor, n_blocks, soft_cap):
    """Load-average capped mu over snapshots; return (S, argmax constraint, cap_frac, n_binding)."""
    per_snap = []
    for s in snapshots:
        m, md, solver, base_mva = build_config(s, candidates, config, emergency_factor,
                                                n_blocks, soft_cap)
        solve(m, md, solver)
        per_snap.append(snapshot_mu(m, base_mva))
    cids = set().union(*per_snap) if per_snap else set()
    load_avg = {c: sum(ps.get(c, 0.0) for ps in per_snap) / len(per_snap) for c in cids}
    if not load_avg:
        return 0.0, None, 0.0, 0
    c_star, S = max(load_avg.items(), key=lambda kv: kv[1])
    cap_frac = sum(ps.get(c_star, 0.0) >= soft_cap - 1.0 for ps in per_snap) / len(per_snap)
    return S, c_star, cap_frac, len(load_avg)


def make_configs(k, n_random, p, seed):
    """The config set: all-off, all-on, then n_random Bernoulli(p) queue draws."""
    configs = [("all-off", [0] * k), ("all-on", [1] * k)]
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        cfg = (rng.random(k) < p).astype(int).tolist()
        configs.append((f"rand{i + 1}", cfg))
    return configs


def summarize(rows, n_snap, cap):
    """Print the spread of S(x) and how many configs land on the flat cap plateau."""
    svals = [r["S"] for r in rows]
    plateau = [r["label"] for r in rows if r["cap_frac"] >= 1.0 - 1e-9]
    print("\n" + "=" * 68)
    print(f"summary over {len(rows)} configs, {n_snap} snapshots, cap ${cap:.0f}")
    print(f"  S(x) range: {min(svals):8.1f} -> {max(svals):8.1f}  (spread {max(svals) - min(svals):8.1f})")
    print(f"  configs with S > 0:            {sum(s > 0 for s in svals)}/{len(rows)}")
    print(f"  configs on the cap plateau:    {len(plateau)}/{len(rows)}  {plateau}")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emergency-factor", type=float, default=1.2)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--snapshots", type=int, default=0, help="0 = all")
    parser.add_argument("--n-random", type=int, default=4)
    parser.add_argument("--p", type=float, default=0.5, help="Bernoulli online probability")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    candidates = json.loads(QUEUE.read_text())["candidates"]
    snapshots = json.loads(COLLECTION.read_text())["snapshots"]
    if args.snapshots:
        snapshots = snapshots[:args.snapshots]
    configs = make_configs(len(candidates), args.n_random, args.p, args.seed)

    print(f"sweep: factor {args.emergency_factor}, {len(snapshots)} snapshots, cap "
          f"${SHADOW_PRICE_CAP:.0f}, ORDC {C_MIN:.0f}->{C_MAX:.0f}")
    print(f"{'config':16s} {'n_on':>4s}  {'S($/MWh)':>9s}  {'cap_frac':>8s}  {'#bind':>5s}  constraint")
    rows = []
    for label, cfg in configs:
        S, c_star, cap_frac, n_bind = evaluate_config(snapshots, candidates, cfg,
                                                       args.emergency_factor, args.blocks,
                                                       SHADOW_PRICE_CAP)
        rows.append({"label": label, "S": S, "cap_frac": cap_frac})
        print(f"{label:16s} {sum(cfg):4d}  {S:9.1f}  {cap_frac:8.2f}  {n_bind:5d}  {c_star}",
              flush=True)
    summarize(rows, len(snapshots), SHADOW_PRICE_CAP)


if __name__ == "__main__":
    main()
