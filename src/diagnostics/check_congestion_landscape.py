"""Characterize the congestion landscape of a gen-per-bus case before enumerating it.

We need the new model to have the features the 14-bus model lacked: multiple distinct
congestion paths, and many separate ways to reach the high-mu tail (a multimodal tail,
not one dominant binding line).  This samples configs across several availability
levels, solves each DCOPF, and reports:

  - feasibility rate and the S = sum_l |mu_l| distribution;
  - how many distinct branches ever bind (how many corridors exist);
  - among the high-S tail, how many distinct binding-line signatures occur (modes),
    and which lines carry the peak mu (distinct dominant corridors = distinct paths).

Usage:
    uv run python src/check_congestion_landscape.py
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

from egret.data.model_data import ModelData
import model.dcopf as so

from paths import REPO_ROOT
CASE_PATH = REPO_ROOT / "data" / "case30_gen_per_bus.json"
SOLVER = "highs"
MU_TOL = 1e-3
LEVELS = [0.8, 0.6, 0.45, 0.35]     # nominal and tail-probing availability levels
N_PER_LEVEL = 800

_MD = None


def _init():
    global _MD
    _MD = ModelData.read(str(CASE_PATH))


def _solve(av):
    """Solve one config; return (feasible, S, sorted binding branch names, peak line)."""
    try:
        md_sol, m = so.solve_opf(so.create_case(_MD, av), solver=SOLVER)
        so.store_branch_duals(m, md_sol)
    except Exception:
        return False, 0.0, (), None
    branches = md_sol.data["elements"]["branch"]
    mu = {bn: b.get("mu_upper", 0.0) + b.get("mu_lower", 0.0) for bn, b in branches.items()}
    binding = tuple(sorted(bn for bn, v in mu.items() if v > MU_TOL))
    peak = max(mu, key=mu.get) if binding else None
    return True, float(sum(mu.values())), binding, peak


def sample_configs(ngen, rng):
    """Draw N_PER_LEVEL Bernoulli configs at each availability level."""
    out = []
    for p in LEVELS:
        out.append((rng.random((N_PER_LEVEL, ngen)) < p).astype(int))
    return np.concatenate(out)


def report(configs, results):
    """Print landscape diagnostics, focusing on tail multimodality."""
    feas = np.array([r[0] for r in results])
    S = np.array([r[1] for r in results])
    on = configs.sum(axis=1)
    print(f"solved {feas.sum()}/{len(feas)} feasible   "
          f"corr(S, #on) = {np.corrcoef(S[feas], on[feas])[0, 1]:+.2f}")
    qs = [50, 90, 99, 100]
    print("S quantiles " + "  ".join(f"p{q}={np.percentile(S[feas], q):.1f}" for q in qs))

    binding = [results[i][2] for i in np.where(feas)[0]]
    all_lines = set().union(*binding) if binding else set()
    print(f"distinct branches that ever bind: {len(all_lines)} of 41")

    thr = np.percentile(S[feas], 90)
    tail = [(results[i][2], results[i][3]) for i in np.where(feas)[0] if S[i] >= thr]
    sigs = Counter(b for b, _ in tail)
    peaks = Counter(p for _, p in tail)
    print(f"\nhigh-S tail (S >= p90 = {thr:.1f}): {len(tail)} configs")
    print(f"  distinct binding-line signatures (modes): {len(sigs)}")
    print(f"  distinct peak-mu lines (congestion paths): {len(peaks)}")
    print("  top peak-mu lines:  " +
          "  ".join(f"L{ln}:{c}" for ln, c in peaks.most_common(8)))
    print("  top signatures:")
    for sig, c in sigs.most_common(6):
        print(f"    x{c:<4d} lines {sig}")


def main():
    rng = np.random.default_rng(0)
    md = ModelData.read(str(CASE_PATH))
    ngen = len(md.data["elements"]["generator"])
    configs = sample_configs(ngen, rng)
    with ProcessPoolExecutor(initializer=_init) as pool:
        results = list(pool.map(_solve, list(configs), chunksize=16))
    report(configs, results)


if __name__ == "__main__":
    main()
