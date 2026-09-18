"""Driver for the per-line panel figures: one parameter set, six figure modules.

Six modules in this package answer the same question -- what does the congestion dual of
each individual line do -- from different angles, and are read together:

    exact_mu_distribution   the exact distribution of |mu_l|, by enumeration
    top_config_mu           |mu_l| of the most probable configurations
    line_exceed_strip       every sampler's P[|mu_l| > c], per (target, budget) cell
    per_line_survivor       every sampler's survivor curve, per (target, budget) slice
    direct_box              direct sampling's P[|mu_l| > c] and its q-quantile, per budget
    direct_inverse_cdf      individual direct replicate survivor curves against truth

Each has its own CLI and its own `run()`; this fixes the parameters they share -- case,
nominal laws, exceedance threshold, budgets -- in one place and calls all six with them, so
a consistent set is produced by one command rather than six agreeing invocations.  Nothing
is computed here: every module writes its own PDFs to results/<case>/ as usual.

The last four read the sampling experiment and so cover only the laws it was run under; the
first two are pure enumeration and work for any law in the registry.

Usage:
    uv run python -m src.figures.tail.per_line_figures --case case30_gen_per_bus
    uv run python -m src.figures.tail.per_line_figures --case activsg2000_original_v12 \
        --dists bernoulli,mrf_c --threshold 200 --budget 2560
    uv run python -m src.figures.tail.per_line_figures --case activsg2000_original_v12 \
        --dists mrf_cobuild,mrf_substitute --only exact_mu_distribution,top_config_mu
"""

import argparse

import matplotlib.pyplot as plt

from src.cases.registry import add_case_arg
from src.sampling.problem import REPO_ROOT, experiment_meta
from src.figures.tail import (direct_box, direct_inverse_cdf, exact_mu_distribution,
                          line_exceed_strip, per_line_survivor, top_config_mu)


def parse_args():
    """The parameters every per-line figure is drawn with, with the lists already split."""
    parser = argparse.ArgumentParser()
    add_case_arg(parser, default="activsg2000_original_v12")
    parser.add_argument("--dists", default="mrf_cobuild,mrf_substitute")
    parser.add_argument("--targets", default="max_mu,sum_mu")
    parser.add_argument("--budgets", default=None,
                        help="budget cells on the x axis; default every budget in the run")
    parser.add_argument("--budget", type=int, default=None,
                        help="budget of the single-slice figures; default the largest")
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="exceedance level c of P[|mu_l| > c]")
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--n-lines", type=int, default=20)
    parser.add_argument("--n-configs", type=int, default=5)
    parser.add_argument("--n-reps", type=int, default=8)
    parser.add_argument("--kind", default="survivor", choices=["survivor", "qq"])
    parser.add_argument("--mu-floor", type=float, default=1e-2,
                        help="treat |mu| below this as uncongested (near-binding dust)")
    args = parser.parse_args()
    args.dists = args.dists.split(",")
    args.targets = args.targets.split(",")
    return args


def main():
    args = parse_args()
    meta = experiment_meta(REPO_ROOT / "results" / args.case, "samples")
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else meta["budgets"]
    budget = args.budget or max(budgets)
    v = vars(args)
    params = lambda l: {k: v[k] for k in l}
    a = [args.case, args.dists]

    exact_mu_distribution.run(*a, **params(["n_lines", "mu_floor"]))
    top_config_mu.run(*a, **params(["threshold", "n_lines", "n_configs"]))
    line_exceed_strip.run(*a, budgets=budgets, **params(["targets", "threshold"]))
    per_line_survivor.run(*a, budgets=[budget], **params(["targets", "kind"]))
    direct_box.run(*a, budgets=budgets, **params(["threshold", "quantile", "mu_floor"]))
    direct_inverse_cdf.run(*a, budget=budget, **params(["n_reps"]))

    plt.show()


if __name__ == "__main__":
    main()
