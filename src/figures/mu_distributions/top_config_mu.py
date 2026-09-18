"""Per-line |mu| of the few most probable configurations, all nominal laws in one panel.

A single row of per-line panels on the shared panel_layout.fraction_grid scaffold (figure-fraction
margins, a right-hand legend band), but each panel holds single configuration values: for the
n most probable configurations under a law, one dot per configuration at its |mu_l| on that
line, so the y axis is the dual itself rather than an exceedance probability.  Every law of
the case is drawn in the same panel, color coded by law and dodged along x, with the law's
exact nominal-weighted mean of |mu_l| as a thin dashed line in the same color.

The |mu_l| values are shared across laws (physics of the enumerated configs); a law enters
only through which configurations are most probable and in what order.  So a configuration
that tops several laws' rankings lands at one height under each, and the colors separate
exactly where a law promotes a different configuration.

Both the configuration ranking and the line ranking are exact -- the case is fully
enumerated, so every configuration's nominal weight is known and no sampling enters.  Lines
are ordered by the baseline law's exact P[|mu_l| > threshold] (the line_exceed_strip order),
ties broken by its exact mean.  For a snapshot-averaged case (texas family) the plotted
|mu_l| is the average over that case's load snapshots.

Usage:
    uv run python -m src.figures.mu_distributions.top_config_mu --case activsg2000_v12_lev3a3
    uv run python -m src.figures.mu_distributions.top_config_mu --case case30_gen_per_bus \
        --dists bernoulli,mrf_a --n-lines 15 --n-configs 5
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator

from src.cases.registry import get_case
from src.sampling.problem import case_tag, create_problem
from src.sampling.distributions import DIST_LABEL, DIST_NAMES, case_law_names
from src.figures.style import DIST_COLOR
from src.figures.mu_distributions._line_stats import exceedance_prob, mean_abs_mu, select_lines
from src.figures.mu_distributions._panel_layout import FIG_W, FONT, fraction_grid, panel_title


def top_configs(problem, n):
    """The n most probable configuration ids and their exact nominal probabilities."""
    ids = np.argsort(-problem.w)[:n]
    return ids, problem.w[ids]


def line_ranking(mu, w, threshold, n_lines):
    """Line indices by descending exact E[|mu_l|]; also returns exact P[|mu_l| > threshold]."""
    prob = exceedance_prob(mu, w, threshold)
    idx, _ = select_lines(mu, np.arange(mu.shape[1]), w, n=n_lines)
    return idx, prob


def fmt_sci_one_sigfig(v, _pos):
    """One-significant-figure tick label in scientific form: 1000 -> 1e3, 0 -> 0."""
    if v == 0:
        return "0"
    mantissa, exponent = f"{v:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def mu_yaxis(ax):
    """|mu_l| y axis: one-sig-fig scientific major ticks (1e3, not 1000) plus minor ticks."""
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_sci_one_sigfig))
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def config_panel(ax, mu_line, tops, means, dists, name, n_configs):
    """One line's panel: for each law, a dodged dot per ranked config and the law's mean."""
    ranks = np.arange(n_configs)
    offs = np.linspace(-0.01, 0.01, len(dists))
    for d, dx in zip(dists, offs):
        ids, _ = tops[d]
        ax.scatter(ranks + dx, mu_line[ids], c=[DIST_COLOR[d]], s=12, linewidths=0, zorder=3)
        ax.axhline(means[d], color=DIST_COLOR[d], ls="--", lw=0.6, alpha=0.75, zorder=1)
    ax.set_xlim(-0.6, n_configs - 0.4)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(k + 1) for k in ranks])
    panel_title(ax, f"line {name}")
    ax.tick_params(labelsize=FONT["tick"], pad=1.5)
    ax.tick_params(axis="x", length=0)
    mu_yaxis(ax)


def config_legend(dists):
    """Legend handles: one colored marker per law, plus the dashed exact-mean line."""
    handles = [Line2D([0], [0], marker="o", ls="", ms=3, color=DIST_COLOR[d],
                      label=DIST_LABEL[d]) for d in dists]
    return handles + [Line2D([0], [0], color="0.4", ls="--", lw=0.8, label="exact mean")]


def panel_ylim(mu, tops, means, order, dists):
    """Shared y top just above the tallest dot or mean across all laws and shown lines."""
    top = max(float(mu[np.asarray(tops[d][0])][:, order].max()) for d in dists)
    mid = max(float(means[d][order].max()) for d in dists)
    return 1.25 * max(top, mid)


def plot_top_configs(case, problems, mu, line_names, dists, threshold, n_lines, n_configs,
                     case_dir):
    """One figure: |mu_l| of each law's most probable configs, all laws in every panel."""
    baseline = dists[0]
    tops = {d: top_configs(problems[d], n_configs) for d in dists}
    means = {d: mean_abs_mu(mu, problems[d].w) for d in dists}
    order, prob = line_ranking(mu, problems[baseline].w, threshold, n_lines)
    for d in dists:
        ids, probs = tops[d]
        print(f"{d}: configs {[int(i) for i in ids]} carry {probs.sum():.3f} of the mass")
    print(f"{(prob > 0).sum()} lines congested at threshold, showing {len(order)}")

    fig, axs, figh = fraction_grid(1, len(order), lft=0.07, foot=0.15)
    for l, ax in zip(order, axs.ravel()):
        config_panel(ax, mu[:, l], tops, {d: means[d][l] for d in dists}, dists,
                     line_names[l], n_configs)
    axs.ravel()[0].set_ylim(0, panel_ylim(mu, tops, means, order, dists))

    fig.legend(handles=config_legend(dists), loc="center left",
               bbox_to_anchor=(1 - 0.14, 0.55), ncol=1, frameon=True,
               fontsize=FONT["legend"], handlelength=1.2, handletextpad=0.4, borderpad=0.3)
    fig.supxlabel("configuration, most probable first", y=0.04 / figh, fontsize=FONT["axlabel"])
    axs[0, 0].set_ylabel(r"$|\mu_l|$", fontsize=FONT["axlabel"])

    out = case_dir / f"top_config_mu_{case_tag(case)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def run(case, dists, threshold=100.0, n_lines=4, n_configs=5):
    """One most-probable-configuration figure per case, every law overlaid by color."""
    c = get_case(case)
    dists = case_law_names(c, dists)
    problems = {d: create_problem(c, d) for d in dists}
    mu, line_names = problems[dists[0]].mu_table()
    return plot_top_configs(c.name, problems, mu, line_names, dists, threshold,
                            n_lines, n_configs, c.results_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_lev3a3")
    parser.add_argument("--dists", default=",".join(DIST_NAMES))
    parser.add_argument("--n-lines", type=int, default=5)
    parser.add_argument("--n-configs", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="exceedance level the exact line ranking is taken at")
    args = parser.parse_args()

    run(args.case, args.dists.split(","), args.threshold, args.n_lines, args.n_configs)
    plt.show()


if __name__ == "__main__":
    main()
