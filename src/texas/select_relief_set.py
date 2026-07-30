"""Select and validate a slim nodal-relief bus set for the full-sweep harness.

Only buses that ever shed load need an ORDC relief generator; the rest enlarge the LP
~5x (see cost decomposition) without changing any solution, because an inactive relief
generator is still a column the solver factorizes on every warm re-solve. This solves a
diverse config sample under FULL relief (every load bus), collects the union of buses
that actually shed, and checks that restricting relief to that set reproduces the scalar
target S on a fresh HELD-OUT config sample (the set is frozen before the held-out actives
are seen, so it is a real generalization test). The set is written for sample_harness,
which reads it from HARNESS_RELIEF_SET.

The full sweep additionally saves, per config, the worst line overload and the system-wide
shed: any config needing a bus outside the set stays feasible via overload / the system
slack and is flagged there for a full-relief recheck. This validation bounds how often
that will happen.

Usage:
    uv run python src/texas/select_relief_set.py
    uv run python src/texas/select_relief_set.py --collect 400 --validate 200
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
from egret.data.model_data import ModelData
from egret.models.scopf import create_scopf_model

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "texas"))

from load_averaged_target import (add_queue_generators, add_lostload_generators_ordc,
                                   scale_loads_by_zone, scale_renewables_by_snapshot)
from soft_limits import enable_soft_limits, thermal_violations
from sample_harness import (set_emergency_factor, solve_config, EMERGENCY_FACTOR,
                            ORDC_C_MIN, ORDC_C_MAX, ORDC_BLOCKS, SHADOW_PRICE_CAP,
                            CASE, QUEUE, COLLECTION, RESULTS_NAME)

RESULTS = REPO_ROOT / "results" / RESULTS_NAME
RELIEF_SET = RESULTS / "relief_set_slim.json"
SHED_TOL_MW = 0.05      # a relief generator counts as active above this shed
DS_TOL = 1.0            # $/MWh; |S_full - S_slim| below this is numerical noise


def build(snapshot, candidates, relief_buses):
    """One snapshot's capped SCDCOPF with relief at relief_buses (None = all), persistent.

    Renewables (CDR fleet and queue wind/solar) are derated to the snapshot's
    capacity factor, matching the sweep's operating conditions so the shedding
    buses reflect real CF-derated dispatch rather than nameplate injection.
    """
    cf = snapshot.get("renewable_cf") or {}
    md = ModelData.read(str(CASE))
    set_emergency_factor(md, EMERGENCY_FACTOR)
    add_queue_generators(md, candidates, [1] * len(candidates))
    scale_loads_by_zone(md, snapshot["zone_scales"])
    scale_renewables_by_snapshot(md, cf)
    add_lostload_generators_ordc(md, ORDC_C_MIN, ORDC_C_MAX, ORDC_BLOCKS, buses=relief_buses)
    m, mds = create_scopf_model(md, include_feasibility_slack=True)
    bm = md.data["system"]["baseMVA"]
    enable_soft_limits(m, SHADOW_PRICE_CAP, bm)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    s = pyo.SolverFactory("gurobi_persistent"); s.options["OutputFlag"] = 0; s.set_instance(m)
    cap_pu = {f"queue_{c['inr']}": cf.get(c["fuel"], 1.0) * c["capacity_mw"] / bm
              for c in candidates}
    return m, mds, s, cap_pu, list(cap_pu), bm


def snapshot_pass(snapshot, candidates, relief_buses, bits):
    """Warm-solve each config on one snapshot; yield (mus, active_buses, overload_mw, shed_mw)."""
    m, mds, s, cap_pu, names, bm = build(snapshot, candidates, relief_buses)
    gens = mds.data["elements"]["generator"]
    for cfg in bits:
        mus = solve_config(m, mds, s, cfg, cap_pu, names, bm)
        active = {gens[nm]["bus"] for nm, v in m.pg.items()
                  if nm.startswith("lostload_") and (v.value or 0.0) * bm > SHED_TOL_MW}
        ov = max(thermal_violations(m, bm).values(), default=0.0)
        shed = float(pyo.value(m.p_load_shed)) * bm
        yield mus, active, ov, shed


def config_targets(snapshots, candidates, relief_buses, bits):
    """Per config over all snapshots: (S, active_buses, max_overload_mw, max_shed_mw)."""
    n = len(snapshots)
    totals = [defaultdict(float) for _ in bits]
    active = [set() for _ in bits]
    maxov = np.zeros(len(bits)); maxshed = np.zeros(len(bits))
    for snap in snapshots:
        for i, (mus, a, ov, shed) in enumerate(snapshot_pass(snap, candidates, relief_buses, bits)):
            for k, v in mus.items():
                totals[i][k] += v
            active[i] |= a
            maxov[i] = max(maxov[i], ov); maxshed[i] = max(maxshed[i], shed)
    S = np.array([max((v / n for v in t.values()), default=0.0) for t in totals])
    return S, active, maxov, maxshed


def sample_configs(ng, n, rng):
    """A diverse config bit-matrix: all-off, all-on, and Bernoulli draws at mixed p."""
    rows = [[0] * ng, [1] * ng]
    for p in (0.2, 0.35, 0.5, 0.65, 0.8):
        for _ in range(max(1, n // 5)):
            rows.append((rng.random(ng) < p).astype(int).tolist())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=int, default=400)
    ap.add_argument("--validate", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    candidates = json.loads(QUEUE.read_text())["candidates"]
    snapshots = json.loads(COLLECTION.read_text())["snapshots"]
    ng = len(candidates)

    collect_bits = sample_configs(ng, args.collect, np.random.default_rng(args.seed))
    print(f"collecting active relief buses: {len(collect_bits)} configs x {len(snapshots)} snapshots")
    _, active_collect, _, shed_collect = config_targets(snapshots, candidates, None, collect_bits)
    relief_set = set().union(*active_collect)
    print(f"active relief buses in collection sample: {len(relief_set)}   "
          f"(full-relief system shed max {shed_collect.max():.2f} MW, should be ~0)")

    val_bits = sample_configs(ng, args.validate, np.random.default_rng(args.seed + 1))
    print(f"validating on {len(val_bits)} held-out configs (set frozen before their actives)")
    S_full, active_val, ov_full, _ = config_targets(snapshots, candidates, None, val_bits)
    S_slim, _, ov_slim, shed_slim = config_targets(snapshots, candidates, relief_set, val_bits)

    dS = np.abs(S_full - S_slim)
    miss = [i for i, a in enumerate(active_val) if not a <= relief_set]
    print(f"\nVALIDATION")
    print(f"  max|dS| {dS.max():8.3f}  mean|dS| {dS.mean():.4f}  configs |dS|>{DS_TOL}$: "
          f"{int((dS > DS_TOL).sum())}/{len(val_bits)}")
    print(f"  held-out configs needing a bus outside the set: {len(miss)}/{len(val_bits)}")
    print(f"  worst overload  full {ov_full.max():7.1f} MW   slim {ov_slim.max():7.1f} MW")
    print(f"  slim system shed max {shed_slim.max():.2f} MW (nonzero => slim relief fell short)")

    final = sorted(relief_set.union(*active_val))
    RESULTS.mkdir(parents=True, exist_ok=True)
    RELIEF_SET.write_text(json.dumps(final))
    verdict = "PASS" if dS.max() <= DS_TOL else "REVIEW (S mismatch on held-out)"
    print(f"\n{verdict}: wrote {len(final)} relief buses -> {RELIEF_SET}")


if __name__ == "__main__":
    main()
