"""Direct Monte Carlo over a testbed's congestion-dual tail, scored against exact truth.

For each case, sweeps the cross-product of nominal law x target (importance function) x
snapshot mode x budget, drawing `budget` iid configurations from the nominal law and scoring
their weighted exceedance curve against the exact enumerated truth:

    columns: case, dist, target, snapshot, budget, method, ascent_move, rep,
             threshold, estimate      (method "truth" carries the exact curve)

Every case shares the common target_cache format (build_target_cache.py), so a case is
selected by name alone and everything runs as table lookups against the cached enumeration.

The swept laws default to every law build_laws.py stores, structured ones included; a case
with no queue has no technology groups, so its structured laws are skipped.  Sweeping a
subset of the laws rewrites only those rows in each budget shard, so a sweep can be extended
without redoing the rest.

A substring or `all` selection tolerates members that have not been built yet: a case with no
target_cache is skipped with a notice rather than aborting the batch (a single named case
still errors, since it was asked for by name).  `--skip-existing` additionally skips any case
that already carries a shard for every requested budget, so an interrupted sweep resumes
without recomputing the cases that finished.

Usage:
    uv run python -m src.sampling.run_experiments --case activsg2000_v12_lev3a3
    uv run python -m src.sampling.run_experiments --case all --n-rep 40 --budgets 640,2560
    uv run python -m src.sampling.run_experiments --case case14_queue
    uv run python -m src.sampling.run_experiments --case lev --skip-existing
"""

import argparse

import numpy as np
import pandas as pd

from src.cases import registry
from src.sampling.problem import TargetProblem, write_experiment_shard, experiment_dir
from src.sampling.distributions import DIST_NAMES, case_dist, case_law_names

PROBS = np.array([1e-2, 1e-3, 1e-4])                     # fixed tail points reported per cell


def direct(problem, evaluator, rng, budget):
    """Baseline sampler: budget iid nominal draws, each weighted 1/budget."""
    ids = problem.bits_to_id(problem.draw_nominal(budget, rng))
    evaluator(ids)
    return ids, np.full(len(ids), 1.0 / len(ids))


def target_specs(problem):
    """(snapshot, target_label, column) for every runnable target of a case.

    The averaged/nominal targets run under snapshot "multi"; the per-snapshot peak (Texas
    only, where snapmax was enumerated) runs the max_mu target under snapshot "single"."""
    specs = [("multi", t, t) for t in ("max_mu", "sum_mu", "one_mu", "one_lmp")
             if t in problem.available]
    if "max_mu_single" in problem.available:
        specs.append(("single", "max_mu", "max_mu_single"))
    return specs


def run_cell(problem, budget, n_rep, seed0):
    """Run the direct sampler n_rep times at the budget; tidy (method, move, rep, id, weight)."""
    rows = []
    for r in range(n_rep):
        ev = problem.evaluator()
        ids, w = direct(problem, ev, np.random.default_rng(seed0 + r), budget)
        rows.append(pd.DataFrame({"method": "direct", "move": "na", "rep": r,
                                  "id": np.asarray(ids), "weight": np.asarray(w)}))
    return pd.concat(rows, ignore_index=True)


def score_cell(problem, samples, dims):
    """Weighted exceedance curve per (method, move, rep) plus the exact truth row."""
    grid = problem.thresholds_for_probs(np.logspace(-1, -5, 30))
    thresholds = np.union1d(grid, problem.thresholds_for_probs(PROBS))
    truth = problem.true_exceedance(thresholds)
    hits = problem.S[:, None] >= thresholds[None, :]
    rows = [pd.DataFrame({**dims, "method": "truth", "ascent_move": "na", "rep": 0,
                          "threshold": thresholds, "estimate": truth})]
    for (method, move, rep), g in samples.groupby(["method", "move", "rep"], sort=False):
        est = (g["weight"].to_numpy()[:, None] * hits[g["id"].to_numpy()]).sum(0)
        rows.append(pd.DataFrame({**dims, "method": method, "ascent_move": move, "rep": rep,
                                  "threshold": thresholds, "estimate": est}))
    return pd.concat(rows, ignore_index=True)


def run_case(case, n_rep, budgets, dists, seed0=0):
    """Sweep the direct estimator over one case; write one parquet shard per budget.

    Each budget's curves and samples go to their own shard under experiment_curves/ and
    experiment_samples/, so rerunning over a different budget set adds shards instead of
    overwriting an earlier run, and rerunning a subset of the laws replaces only those rows.
    The plotting side recombines the shards on read."""
    name, tc = case.name, case.target_cache
    base = TargetProblem(tc, "max_mu", dist=case_dist(case, "bernoulli"))
    specs = target_specs(base)

    curves = {b: [] for b in budgets}
    samples = {b: [] for b in budgets}
    for dist_name in case_law_names(case, dists):
        dist = case_dist(case, dist_name)
        for snapshot, label, col in specs:
            problem = TargetProblem(tc, target=col, dist=dist)
            for budget in budgets:
                dims = dict(case=name, dist=dist_name, target=label,
                            snapshot=snapshot, budget=budget)
                raw = run_cell(problem, budget, n_rep, seed0)
                curves[budget].append(score_cell(problem, raw, dims))
                samples[budget].append(raw.rename(columns={"move": "ascent_move"}).assign(**dims))
                print(f"  {dist_name:9s} {label:8s}/{snapshot:5s} b={budget:<5d} "
                      f"(direct x {n_rep} reps)")

    meta = dict(case=name, n_rep=n_rep, budgets=list(budgets), nconf=base.nconf,
                one_mu_element=base.meta["one_mu_element"],
                one_lmp_element=base.meta["one_lmp_element"])
    case_dir = case.results_dir
    for budget in budgets:
        write_experiment_shard(pd.concat(curves[budget], ignore_index=True),
                               meta, case_dir, "curves", budget)
        write_experiment_shard(pd.concat(samples[budget], ignore_index=True),
                               meta, case_dir, "samples", budget)
    print(f"{name}: {len(budgets)} budget shards -> {experiment_dir(case_dir, 'curves')}/ "
          f"and {experiment_dir(case_dir, 'samples')}/\n")


def has_all_shards(case, budgets):
    """True if the case already carries a curves shard for every requested budget.

    run_case writes a case's shards only after all its cells finish, so a case is either
    fully present for a budget or absent; the presence of every budget's file is enough
    to call it done and skip it under --skip-existing.
    """
    d = experiment_dir(case.results_dir, "curves")
    return all((d / f"b{b}.parquet").exists() for b in budgets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_lev3a3",
                        help="a case name, a name substring, or 'all'")
    parser.add_argument("--n-rep", type=int, default=40)
    parser.add_argument("--budgets", default="640,2560,10240")
    parser.add_argument("--dists", default=",".join(DIST_NAMES))
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip cases that already have a shard for every requested budget")
    args = parser.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    dists = args.dists.split(",")

    batch = args.case not in registry.CASES     # a substring or 'all' tolerates un-built members
    for case in registry.get_cases(args.case):
        if batch and not case.target_cache.exists():
            print(f"=== {case.name}: no target_cache, skipping (build_target_cache first) ===")
            continue
        if args.skip_existing and has_all_shards(case, budgets):
            print(f"=== {case.name}: shards present for {budgets}, skipping (--skip-existing) ===")
            continue
        print(f"=== {case.name} ===")
        run_case(case, args.n_rep, budgets, dists)


if __name__ == "__main__":
    main()
