"""Select and validate a slim nodal-relief bus set for the full-sweep harness.

Only buses that ever shed load need an ORDC relief generator; the rest enlarge the LP
~5x (see cost decomposition) without changing any solution, because an inactive relief
generator is still a column the solver factorizes on every warm re-solve. This solves a
diverse config sample under FULL relief (every load bus), collects the union of buses
that actually shed, and checks that restricting relief to that set reproduces the scalar
target S on a fresh HELD-OUT config sample (the set is frozen before the held-out actives
are seen, so it is a real generalization test). The set is written to the active case's
data/<case>_relief_set.json, which the registry loads and every solve picks up.

The full sweep additionally saves, per config, the worst line overload and the system-wide
shed: any config needing a bus outside the set stays feasible via overload / the system
slack and is flagged there for a full-relief recheck. This validation bounds how often
that will happen.

Usage:
    uv run python -m src.cases.texas.select_relief_set --case activsg2000_2025
    uv run python -m src.cases.texas.select_relief_set --case activsg2000_2025 --collect 400 --validate 200
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import pyomo.environ as pyo

from src.cases import registry
from src.model.scdcopf import build_persistent
from src.model.soft_limits import thermal_violations

SHED_TOL_MW = 0.05      # a relief generator counts as active above this shed
DS_TOL = 1.0            # $/MWh; |S_full - S_slim| below this is numerical noise


def snapshot_pass(case, snapshot, relief_buses, bits):
    """Warm-solve each config on one snapshot; yield (mus, active_buses, overload_mw, shed_mw)."""
    pm = build_persistent(case, snapshot, relief_buses=relief_buses)
    gens = pm.md.data["elements"]["generator"]
    for cfg in bits:
        mus = pm.solve(cfg)
        active = {gens[nm]["bus"] for nm, v in pm.m.pg.items()
                  if nm.startswith("lostload_") and (v.value or 0.0) * pm.base_mva > SHED_TOL_MW}
        ov = max(thermal_violations(pm.m, pm.base_mva).values(), default=0.0)
        yield mus, active, ov, pm.load_shed()


def config_targets(case, relief_buses, bits):
    """Per config over all snapshots: (S, active_buses, max_overload_mw, max_shed_mw)."""
    snapshots = case.snapshots()
    n = len(snapshots)
    totals = [defaultdict(float) for _ in bits]
    active = [set() for _ in bits]
    maxov = np.zeros(len(bits)); maxshed = np.zeros(len(bits))
    for snap in snapshots:
        for i, (mus, a, ov, shed) in enumerate(snapshot_pass(case, snap, relief_buses, bits)):
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
    registry.add_case_arg(ap)
    ap.add_argument("--collect", type=int, default=400)
    ap.add_argument("--validate", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    case = registry.get(args.case)
    ng, snapshots = case.ngen, case.snapshots()

    collect_bits = sample_configs(ng, args.collect, np.random.default_rng(args.seed))
    print(f"collecting active relief buses: {len(collect_bits)} configs x {len(snapshots)} snapshots")
    _, active_collect, _, shed_collect = config_targets(case, None, collect_bits)
    relief_set = set().union(*active_collect)
    print(f"active relief buses in collection sample: {len(relief_set)}   "
          f"(full-relief system shed max {shed_collect.max():.2f} MW, should be ~0)")

    val_bits = sample_configs(ng, args.validate, np.random.default_rng(args.seed + 1))
    print(f"validating on {len(val_bits)} held-out configs (set frozen before their actives)")
    S_full, active_val, ov_full, _ = config_targets(case, None, val_bits)
    S_slim, _, ov_slim, shed_slim = config_targets(case, relief_set, val_bits)

    dS = np.abs(S_full - S_slim)
    miss = [i for i, a in enumerate(active_val) if not a <= relief_set]
    print(f"\nVALIDATION")
    print(f"  max|dS| {dS.max():8.3f}  mean|dS| {dS.mean():.4f}  configs |dS|>{DS_TOL}$: "
          f"{int((dS > DS_TOL).sum())}/{len(val_bits)}")
    print(f"  held-out configs needing a bus outside the set: {len(miss)}/{len(val_bits)}")
    print(f"  worst overload  full {ov_full.max():7.1f} MW   slim {ov_slim.max():7.1f} MW")
    print(f"  slim system shed max {shed_slim.max():.2f} MW (nonzero => slim relief fell short)")

    # A sweep input, not a sweep output: written to data/ beside the case it belongs to,
    # under the name the registry expects to load it from.
    final = sorted(relief_set.union(*active_val))
    dest = registry.DATA / f"{case.name}_relief_set.json"
    dest.write_text(json.dumps(final))
    verdict = "PASS" if dS.max() <= DS_TOL else "REVIEW (S mismatch on held-out)"
    print(f"\n{verdict}: wrote {len(final)} relief buses -> {dest}")


if __name__ == "__main__":
    main()
