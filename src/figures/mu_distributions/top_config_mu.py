"""Per-line |mu| of the few most probable configurations, one panel per line.

Same per-line panel grid and dots as line_exceed_strip.py, but each panel holds single
configuration values instead of repeated sampling estimates: for the n most probable
configurations under the nominal law, one dot per configuration at its |mu_l| on that
line, so the y axis is the dual itself rather than an exceedance probability.  The dashed
line is the exact nominal-weighted mean of |mu_l|, giving the typical level the individual
configurations are read against.

Both the configuration ranking and the line ranking are exact -- the case is fully
enumerated, so every configuration's nominal weight is known and no sampling enters.
Lines are ordered by the exact P[|mu_l| > threshold] (the line_exceed_strip order), ties
broken by the exact mean, which matters here because the most congested elements of a
Texas case sit at probability 1.  For a snapshot-averaged case (texas family) the plotted
|mu_l| is the average over that case's load snapshots.

Usage:
    uv run python -m src.figures.tail.top_config_mu --case activsg2000_original_v12
    uv run python -m src.figures.tail.top_config_mu --case case30_gen_per_bus \
        --dists bernoulli --n-lines 15 --n-configs 5
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.sampling.problem import case_tag
from src.sampling.distributions import DIST_LABEL
from src.figures.tail.exact_mu_distribution import load_problem
from src.figures.tail.line_exceed_strip import (FONT, LABEL_IN, LEGEND_IN, TITLE_IN,
                                            margin_grid, one_sigfig_yaxis, truth_probabilities)


def config_colors(n):
    """One color per ranked configuration, dark (most probable) to light."""
    return plt.get_cmap("viridis")(np.linspace(0.1, 0.8, n))


def top_configs(problem, n):
    """The n most probable configuration ids and their exact nominal probabilities."""
    ids = np.argsort(-problem.w)[:n]
    return ids, problem.w[ids]


def line_ranking(mu, w, threshold, n_lines):
    """Line indices by descending exact P[|mu_l| > threshold], ties broken by exact mean."""
    prob = truth_probabilities(mu, w, threshold)
    mean = w @ mu
    return np.lexsort((-mean, -prob))[:n_lines], prob, mean


def config_panel(ax, values, mean, colors, name):
    """One line's panel: a dot per ranked configuration plus the exact mean."""
    ax.scatter(np.arange(len(values)), values, c=colors, s=8, linewidths=0, zorder=3)
    ax.axhline(mean, color="k", ls="--", lw=0.7)
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels([str(i + 1) for i in range(len(values))])
    ax.set_title(f"line {name}", fontsize=FONT["panel_title"], y=0.84)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=FONT["tick"], pad=1.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.25 * max(float(values.max()), float(mean)))
    one_sigfig_yaxis(ax)


def config_legend(ids, probs, colors):
    """Legend handles: one entry per ranked configuration, plus the exact-mean line."""
    handles = [Line2D([0], [0], marker="o", ls="", ms=2.5, color=colors[k],
                      label=f"{k + 1}: id {int(i)} (p {p:.3f})")
               for k, (i, p) in enumerate(zip(ids, probs))]
    return handles + [Line2D([0], [0], color="k", ls="--", label="exact mean")]


def plot_top_configs(case, problem, mu, line_names, dist, threshold, n_lines, n_configs,
                     case_dir):
    """One figure: |mu_l| of the most probable configurations, one panel per line."""
    ids, probs = top_configs(problem, n_configs)
    order, prob, mean = line_ranking(mu, problem.w, threshold, n_lines)
    colors = config_colors(n_configs)
    print(f"{dist}: configs {[int(i) for i in ids]} carry "
          f"{probs.sum():.3f} of the nominal mass, {(prob > 0).sum()} lines congested")

    ncol = min(5, len(order))
    nrow = int(np.ceil(len(order) / ncol))
    fig, axs = margin_grid(nrow, ncol)
    for l, ax in zip(order, axs.ravel()):
        config_panel(ax, mu[ids, l], mean[l], colors, line_names[l])
    for ax in axs.ravel()[len(order):]:
        ax.set_visible(False)

    w, h = fig.get_figwidth(), fig.get_figheight()
    fig.legend(handles=config_legend(ids, probs, colors), loc="upper center",
               bbox_to_anchor=(0.5, 1 - LEGEND_IN / h), ncol=n_configs + 1, frameon=True,
               fontsize=FONT["legend"], handletextpad=0.3, columnspacing=0.8,
               borderpad=0.3, handlelength=1.4)
    fig.suptitle(f"{case}: $|\\mu_l|$ of the {n_configs} most probable configurations  |  "
                 f"{DIST_LABEL[dist]}", y=1 - TITLE_IN / h,
                 fontsize=FONT["suptitle"])
    fig.supxlabel("configuration, most probable first", y=LABEL_IN / h,
                  fontsize=FONT["axlabel"])
    fig.supylabel(r"$|\mu_l|$", x=LABEL_IN / w, fontsize=FONT["axlabel"])

    out = case_dir / f"top_config_mu_{dist}_{case_tag(case)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def run(case, dists, threshold=100.0, n_lines=15, n_configs=5):
    """One most-probable-configuration figure per nominal law; returns them in order."""
    figs = []
    for dist in dists:
        case_dir, problem = load_problem(case, dist)
        mu, line_names = problem.mu_table()
        figs.append(plot_top_configs(case, problem, mu, line_names, dist, threshold,
                                     n_lines, n_configs, case_dir))
    return figs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_original_v12")
    parser.add_argument("--dists", default="bernoulli,mrf_a,mrf_b,mrf_c")
    parser.add_argument("--n-lines", type=int, default=15)
    parser.add_argument("--n-configs", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="exceedance level the exact line ranking is taken at")
    args = parser.parse_args()

    run(args.case, args.dists.split(","), args.threshold, args.n_lines, args.n_configs)
    plt.show()


if __name__ == "__main__":
    main()
