"""How the nominal law spreads its probability over the 2^ngen availability configs.

Every other figure in this package plots the *output* distribution of the congestion dual;
this one plots the *input* distribution the samplers draw from -- the nominal law's own pmf
over all 2^ngen generator-availability configs -- to show how concentrated or spread that law
is before any physics enters.  For each law the exact enumerated config weights are sorted
from most to least probable and drawn two ways:

    left    the sorted weight itself versus config rank (log-log): the height of the most
            probable configs and how steeply the weight falls off toward the unlikely ones
    right   the cumulative mass captured by the K most probable configs versus K (log x): the
            fraction of all probability the top configs already hold

A steep left curve and an early-saturating right curve are a concentrated law (a handful of
configs carry the mass, so direct sampling keeps redrawing them); a flat left curve and a
right curve that only reaches one near the full count are a spread law.  One curve per law,
colored as in the per-line enumeration figures.  Both are exact -- the case is fully
enumerated, so every config's nominal weight is known.

Usage:
    uv run python -m src.figures.mu_distributions.nominal_pmf_concentration --case activsg2000_v12_lev3a3
    uv run python -m src.figures.mu_distributions.nominal_pmf_concentration --case activsg2000_v12_lev3a3 \
        --dists bernoulli,mrf_a
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.cases.registry import get_case
from src.sampling.problem import case_tag, create_problem
from src.sampling.distributions import DIST_NAMES, DIST_LABEL
from src.figures.style import DIST_COLOR
from src.figures.mu_distributions._panel_layout import FIG_W, FONT

N_POINTS = 3000                   # log-spaced ranks plotted per curve (the full 2^ngen is decimated)


def sorted_weights(w):
    """The law's config weights sorted from most to least probable."""
    return np.sort(w)[::-1]


def log_ranks(nconf, n=N_POINTS):
    """Distinct 1-based ranks spaced logarithmically across [1, nconf], endpoints kept."""
    r = np.unique(np.round(np.logspace(0, np.log10(nconf), n)).astype(np.int64))
    return np.clip(r, 1, nconf)


def rank_curve_panel(ax, curves, ranks):
    """Left panel: sorted config weight versus rank on log-log axes, one line per law."""
    for d, sw in curves.items():
        ax.plot(ranks, sw[ranks - 1], color=DIST_COLOR[d], lw=1.2, label=DIST_LABEL[d])
    ax.set_xscale("log")
    #ax.set_yscale("log")
    #ax.set_xlabel("config rank (most to least probable)", fontsize=FONT["axlabel"])
    ax.set_ylabel("prob.", fontsize=FONT["axlabel"])
    ax.tick_params(labelsize=FONT["tick"])


def cumulative_panel(ax, curves, ranks):
    """Right panel: cumulative mass held by the K most probable configs versus K (log K)."""
    for d, sw in curves.items():
        cum = np.cumsum(sw)
        ax.plot(ranks, cum[ranks - 1], color=DIST_COLOR[d], lw=1.2, label=DIST_LABEL[d])
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    #ax.set_xlabel("number of most probable configs, K", fontsize=FONT["axlabel"])
    ax.set_xlabel("config rank (most to least probable)", fontsize=FONT["axlabel"])
    ax.set_ylabel("cumulative", fontsize=FONT["axlabel"])
    ax.tick_params(labelsize=FONT["tick"])


def build_figure(curves, nconf, title):
    """The two-panel concentration figure (sorted weight, cumulative mass) with a shared legend."""
    ranks = log_ranks(nconf)
    fig, (ax_left, ax_right) = plt.subplots(2, 1, figsize=(3, 3), sharex=True)
    rank_curve_panel(ax_left, curves, ranks)
    cumulative_panel(ax_right, curves, ranks)
    handles, labels = ax_left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=len(labels), frameon=True, fontsize=FONT["legend"], handletextpad=0.4,
               columnspacing=1.0, borderpad=0.3)
    fig.suptitle(title, y=0.995, fontsize=FONT["suptitle"])
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    return fig


def run(case, dists=DIST_NAMES):
    """Nominal-pmf concentration figure for one case; returns it."""
    c = get_case(case)
    problems = {d: create_problem(c, d) for d in dists}
    curves = {d: sorted_weights(pr.w) for d, pr in problems.items()}
    nconf = next(iter(problems.values())).nconf

    title = f"{c.name.split('_')[0]}: nominal law over {nconf:,} configs"
    fig = build_figure(curves, nconf, title)
    out = c.results_dir / f"nominal_pmf_concentration_{case_tag(c.name)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_lev3a3")
    parser.add_argument("--dists", default=",".join(DIST_NAMES))
    args = parser.parse_args()

    run(args.case, args.dists.split(","))
    plt.show()


if __name__ == "__main__":
    main()
