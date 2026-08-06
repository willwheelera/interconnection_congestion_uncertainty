"""Score the reusable tail samplers against the exact 2^20 Texas ground truth.

Every queue config's S = max load-averaged mu is already cached
(`results/activsg2000/S_cache.parquet`, the full 2^20 sweep), so the samplers run as
pure table lookups here --- no SCDCOPF solves --- and are scored against the exact
nominal-weighted exceedance. The sampler algorithms are unchanged from `tail_sampling/`
(they consume only the Problem interface); this file specializes just the two
problem-dependent pieces for this case: ground truth is the precomputed S-cache, and the
nominal law is the real per-generator Bernoulli (each queue candidate online with its
calibrated p), which is a zero-coupling MRF so `MarkovRandomField` is reused directly.

Usage:
    HARNESS_CASE=2025 uv run python -m src.texas.score_samplers
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO / "src"))                   # tail_sampling + utils packages
sys.path.insert(0, str(REPO / "src" / "texas"))

import pandas as pd

from sample_harness import RESULTS_NAME                 # results dir of the HARNESS_CASE sweep

from tail_sampling.problem_setup import Problem, write_parquet_meta
from tail_sampling.mrf import MarkovRandomField
from tail_sampling import compare_methods, evaluate
import tail_sampling.subset_simulation as ss
import tail_sampling.ascent_importance_sampling as ais
import tail_sampling.smc_tilting as smc
import tail_sampling.ascent_exact_mass as exact

S_CACHE = REPO / "results" / RESULTS_NAME / "S_cache.parquet"
QUEUE = REPO / "data" / "activsg2000_queue20.json"


def bernoulli_law(p):
    """Per-generator independent Bernoulli(p) as a zero-coupling MRF (h = logit p, J = 0)."""
    p = np.asarray(p, dtype=float)
    return MarkovRandomField(np.log(p / (1.0 - p)), np.zeros((len(p), len(p))))


class TexasProblem(Problem):
    """Problem backed by the precomputed S-cache and the real Bernoulli queue law."""

    def __init__(self, s_cache=S_CACHE, queue=QUEUE):
        self.results_dir = Path(s_cache).parent
        p = [c["p"] for c in json.loads(Path(queue).read_text())["candidates"]]
        self.ngen = len(p)
        self.nconf = 1 << self.ngen
        self._powers = (1 << np.arange(self.ngen - 1, -1, -1)).astype(np.int64)  # MSB-first
        self.dist = bernoulli_law(p)
        df = pd.read_parquet(s_cache)
        self.S = np.zeros(self.nconf)
        self.S[df.index.values] = df["S"].values
        self.w = self._nominal_weights()


def main(budget=1280, n_rep=60, seed0=0):
    problem = TexasProblem()
    samplers = {
        "direct": compare_methods.direct,
        "subset_sim": ss.sample,
        "ascent_is": ais.sample,
        "smc_tilt": smc.sample,
        "ascent_exact": exact.sample,
    }
    samples = compare_methods.run_sampling(problem, samplers, budget, n_rep, seed0)

    grid = problem.thresholds_for_probs(np.logspace(-1, -6, 40))
    fixed = problem.thresholds_for_probs(np.array([1e-2, 1e-3, 1e-4, 1e-5]))
    thresholds = np.union1d(grid, fixed)
    truth = problem.true_exceedance(thresholds)
    curves = evaluate.weighted_exceedance(samples, problem.S, thresholds)

    print(f"budget = {budget} solves ({budget / problem.nconf:.2%} of {problem.nconf} configs)"
          f"   replicates = {n_rep}   ngen = {problem.ngen}")
    print(f"nominal-law S: median {np.average(problem.S, weights=problem.w):.1f}  "
          f"max reachable {problem.S[problem.w > 0].max():.1f}")
    evaluate.report(curves, truth, thresholds, fixed)

    meta = dict(case="activsg2000", budget=budget, n_rep=n_rep,
                nconf=problem.nconf, methods=list(samplers))
    out = problem.results_dir / "tail_samples.parquet"
    write_parquet_meta(samples, meta, out)
    write_parquet_meta(evaluate.tidy_curves(curves, thresholds, truth),
                       meta, problem.results_dir / "tail_guiding_curves.parquet")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
