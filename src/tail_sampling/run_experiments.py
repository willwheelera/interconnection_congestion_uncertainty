"""One driver for the full rare-event experiment matrix over all four testbeds.

Sweeps, for each case, the cross-product of nominal law x sampler x ascent move x
target (importance function) x snapshot mode x budget, scores every cell's weighted
sample against the exact enumerated truth, and writes one tidy inverse-CDF frame:

    columns: case, dist, target, snapshot, budget, method, ascent_move, rep,
             threshold, estimate      (method "truth" carries the exact curve)

All four cases share the common target_cache format (build_target_cache.py), so a case
is selected by name alone.  Everything runs as table lookups against the cached
enumeration, except `--proxy live`, which drives the ascent proxy by live per-config
barrier solves (slower; run separately).  Proxy-ranked ascent moves (best / top5) run
only where a proxy is available (the generic gen-per-bus cases); elsewhere ascent uses
the all-neighbor move.

Usage:
    uv run python -m tail_sampling.run_experiments --case case14_gen_per_bus
    uv run python -m tail_sampling.run_experiments --case case30_gen_per_bus --proxy live
    uv run python -m tail_sampling.run_experiments --case all --n-rep 40 --budgets 640,2560
"""

import argparse
import functools
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import tail_sampling.subset_simulation as ss
import tail_sampling.smc_tilting as smc
import tail_sampling.ascent_importance_sampling as ais
import tail_sampling.ascent_exact_mass as exact
import tail_sampling.cross_entropy as ce
from tail_sampling.problem_setup import (TargetProblem, REPO_ROOT, CASES,
                                          write_experiment_shard, experiment_dir)
from tail_sampling.distributions import DIST_NAMES, case_p, make_dist

RESULTS = REPO_ROOT / "results"
PROBS = np.array([1e-2, 1e-3, 1e-4])                     # fixed tail points reported per cell


def direct(problem, ev, rng, budget):
    """Baseline sampler: budget iid nominal draws, each weighted 1/budget."""
    ids = problem.bits_to_id(problem.draw_nominal(budget, rng))
    ev(ids)
    return ids, np.full(len(ids), 1.0 / len(ids))


def build_samplers(dist_name, proxy):
    """The sampler set for one cell, keyed by (method, ascent_move).

    The three base samplers and the two ascent samplers always run; ascent gets the
    proxy moves (best/top5) only when a proxy exists; cross-entropy only under a
    Bernoulli law (it fits a product proposal).
    """
    samplers = {("direct", "na"): direct,
                ("subset_sim", "na"): ss.sample,
                ("smc_tilt", "na"): smc.sample}
    moves = ["all", "best", "top5"] if proxy is not None else ["all"]
    for name, fn in [("ascent_is", ais.sample), ("ascent_exact", exact.sample)]:
        for m in moves:
            lookup = proxy.score if m != "all" else None
            samplers[(name, m)] = functools.partial(fn, move=m, score_lookup=lookup)
    if dist_name == "bernoulli":
        samplers[("cross_entropy", "na")] = ce.sample
    return samplers


def target_specs(problem):
    """(snapshot, target_label, column) for every runnable target of a case.

    The averaged/nominal targets run under snapshot "multi"; the per-snapshot peak (Texas
    only, where snapmax was enumerated) runs the max_mu target under snapshot "single"."""
    specs = [("multi", t, t) for t in ("max_mu", "sum_mu", "one_mu", "one_lmp")
             if t in problem.available]
    if "max_mu_single" in problem.available:
        specs.append(("single", "max_mu", "max_mu_single"))
    return specs


def run_cell(problem, samplers, budget, n_rep, seed0):
    """Run every sampler n_rep times at the budget; tidy (method, move, rep, id, weight)."""
    rows = []
    for r in range(n_rep):
        for (method, move), fn in samplers.items():
            ev = problem.evaluator()
            ids, w = fn(problem, ev, np.random.default_rng(seed0 + r), budget)
            rows.append(pd.DataFrame({"method": method, "move": move, "rep": r,
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


def run_case(name, proxy_kind, n_rep, budgets, seed0=0):
    """Sweep the whole matrix for one case; write one parquet shard per budget.

    Each budget's curves and samples go to their own shard under experiment_curves/ and
    experiment_samples/, so rerunning over a different budget set adds shards instead of
    overwriting an earlier run.  The plotting side recombines the shards on read."""
    from tail_sampling.sensitivity_proxy import make_proxy   # defers egret/pyomo to run time
    spec, tc = CASES[name], RESULTS / name / "target_cache.parquet"
    ngen = int(json.loads(pq.read_schema(tc).metadata[b"run_meta"])["ngen"])
    p = case_p(spec)
    base = TargetProblem(tc, "max_mu", dist=make_dist("bernoulli", ngen, p))
    specs = target_specs(base)
    proxy = make_proxy(proxy_kind if spec["proxy"] else None, name, ngen)

    curves = {b: [] for b in budgets}
    samples = {b: [] for b in budgets}
    for dist_name in DIST_NAMES:
        dist = make_dist(dist_name, ngen, p)
        samplers = build_samplers(dist_name, proxy)
        for snapshot, label, col in specs:
            problem = TargetProblem(tc, target=col, dist=dist, sensitivity=proxy)
            for budget in budgets:
                dims = dict(case=name, dist=dist_name, target=label,
                            snapshot=snapshot, budget=budget)
                raw = run_cell(problem, samplers, budget, n_rep, seed0)
                curves[budget].append(score_cell(problem, raw, dims))
                samples[budget].append(raw.rename(columns={"move": "ascent_move"}).assign(**dims))
                print(f"  {dist_name:9s} {label:8s}/{snapshot:5s} b={budget:<5d} "
                      f"({len(samplers)} samplers x {n_rep} reps)")

    meta = dict(case=name, n_rep=n_rep, budgets=list(budgets), proxy=proxy_kind,
                nconf=base.nconf, proxy_available=proxy is not None,
                one_mu_element=base.meta["one_mu_element"],
                one_lmp_element=base.meta["one_lmp_element"])
    case_dir = RESULTS / name
    for budget in budgets:
        write_experiment_shard(pd.concat(curves[budget], ignore_index=True),
                               meta, case_dir, "curves", budget)
        write_experiment_shard(pd.concat(samples[budget], ignore_index=True),
                               meta, case_dir, "samples", budget)
    print(f"{name}: {len(budgets)} budget shards -> {experiment_dir(case_dir, 'curves')}/ "
          f"and {experiment_dir(case_dir, 'samples')}/\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14_gen_per_bus", choices=[*CASES, "all"])
    parser.add_argument("--proxy", default="cached", choices=["cached", "live"])
    parser.add_argument("--n-rep", type=int, default=40)
    parser.add_argument("--budgets", default="640,2560")
    args = parser.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    for name in (CASES if args.case == "all" else [args.case]):
        print(f"=== {name} (proxy={args.proxy}) ===")
        run_case(name, args.proxy, args.n_rep, budgets)


if __name__ == "__main__":
    main()
