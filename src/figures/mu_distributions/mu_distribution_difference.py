"""Per-line change in the |mu_l| distribution induced by coupling, relative to Bernoulli.

The four nominal laws (independent Bernoulli plus the structured MRFs) put almost the same
distribution on each line's congestion dual |mu_l|; overlaying the four histograms just
draws four curves on top of each other.  This figure separates shape from difference over a
two-row grid, one column per line:

  top row     the Bernoulli congested-mass histogram, mass_Bernoulli(bin) over |mu_l| > floor
              -- the shape all four laws share, drawn once as the baseline
  bottom row  each MRF as one colored step outline of mass_MRF(bin) - mass_Bernoulli(bin),
              on a zero-centered axis -- the coupling effect, magnified onto its own scale

Each row shares one y axis across lines (sharey by row), so a small effect on a busy line is
not flattened by a per-panel rescale, and columns share an x axis (top and bottom of a line
cover the same |mu_l| range).  The |mu| values are shared across laws (physics of the
enumerated configs); the four laws differ only in the config weights, so the difference is a
pure reweighting of the same values.

One figure per case (all laws in the one frame), one column per line, most-congested first.

Usage:
    uv run python -m src.figures.mu_distributions.mu_distribution_difference --case case30_gen_per_bus
    uv run python -m src.figures.mu_distributions.mu_distribution_difference --case case30_gen_per_bus \
        --baseline bernoulli --mrfs mrf_a,mrf_cobuild,mrf_substitute --n-lines 5 --bins 40
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from src.cases.registry import get_case
from src.sampling.problem import case_tag, create_problem
from src.sampling.distributions import DIST_LABEL
from src.figures.style import COLORS, DIST_COLOR
from src.figures.mu_distributions._line_stats import mean_abs_mu, select_lines, line_label
from src.figures.mu_distributions._panel_layout import FIG_W, FONT, one_sigfig_yaxis, fraction_grid, panel_title


def bin_mass(v, w, edges):
    """Weighted probability mass of |mu_l| in each bin (values below the first edge dropped)."""
    return np.histogram(v, bins=edges, weights=w)[0]


def line_column(top, bot, v, weights, baseline, mrfs, colors, name, floor, bins):
    """Fill one line's column: Bernoulli mass on top, each MRF's difference below.

    The dashed mean line is the unconditional exact E[|mu_l|] (mean_abs_mu), the same
    per-line mean used to rank lines and drawn in top_config_mu; binning from zero puts the
    uncongested mass in the first bar, so each column integrates to one."""
    edges = np.linspace(floor, float(v.max()), bins + 1)
    base = bin_mass(v, weights[baseline], edges)
    top.stairs(base, edges, fill=True, color=DIST_COLOR["bernoulli"], alpha=0.55, lw=0)
    top.axvline(mean_abs_mu(v, weights[baseline]), color=DIST_COLOR["bernoulli"],
                lw=0.5, ls="--", zorder=3)
    for mrf in mrfs:
        top.stairs(bin_mass(v, weights[mrf], edges), edges, color=colors[mrf], lw=0.6)
    #    bot.stairs(bin_mass(v, weights[mrf], edges) - base, edges, color=colors[mrf], lw=1.0)
        top.axvline(mean_abs_mu(v, weights[mrf]), color=colors[mrf],
                    lw=0.5, ls="--", zorder=3)
    #bot.axhline(0.0, color="k", lw=0.5, zorder=1)
    panel_title(top, f"{name}")
    for ax in (top, ):#bot):
        ax.tick_params(labelsize=5.5, length=2)


def grid_legend(baseline, mrfs, colors):
    """Legend handles: the red Bernoulli baseline, then one colored outline per MRF."""
    base = [Patch(facecolor=DIST_COLOR[baseline], alpha=0.55, label=DIST_LABEL[baseline])]
    return base+[Line2D([0], [0], color=colors[m], lw=1.4,
                          label=f"{DIST_LABEL[m]}") for m in mrfs]


def plot_differences(case, mu, line_names, baseline, mrfs, weights, floor, bins, n_lines, case_dir):
    """One two-row figure: Bernoulli shape on top, MRF-minus-baseline difference below."""
    idx, names = select_lines(mu, line_names, weights[baseline], n=n_lines)
    shown = list(zip(idx, [line_label(n) for n in names]))
    print(f"showing {len(shown)} lines")
    colors = {m: DIST_COLOR[m] for m in mrfs}
    fig, axs, figh = fraction_grid(1, len(shown), foot=0.2,
#head=0.055, foot=0.172, lft=0.113, rgt=0.133, wspace=0.049, hspace=0.095,
                                   )

    for col, (i, name) in enumerate(shown):
        line_column(axs[0, col], axs[0, col], mu[:, i], weights, baseline, mrfs, colors,
                    name, floor, bins)

    for j in range(len(shown)):
        axs[0, j].set_xlim(0.0, 1.02 * float(mu[:, shown[j][0]].max()))  # shared, from 0
        axs[0, j].xaxis.set_major_locator(MaxNLocator(nbins=4))
    axs[0, 0].set_ylabel(f"prob. mass", fontsize=FONT["axlabel"])
    #axs[0, 0].set_ylabel(f"{DIST_LABEL[baseline]}\nprob. mass", fontsize=FONT["axlabel"])
    #axs[1, 0].set_ylabel(r"$\Delta$ prob. mass", fontsize=FONT["axlabel"])

    y_range = 0.7
    axs[0, 0].set_ylim([0, y_range])
    #m = axs[1, 0].get_ylim()[0]
    #axs[1, 0].set_ylim([m, m+y_range])

    fig.legend(handles=grid_legend(baseline, mrfs, colors), loc="center left",
               bbox_to_anchor=(1 - 0.8 / FIG_W, 0.5), ncol=1, title="Differences", title_fontsize=FONT["legend"],
               frameon=True, fontsize=FONT["legend"], handlelength=1.6, borderpad=0.3)
    #fig.suptitle(f"{case}: |mu| distribution and its change under coupling  |  full enumeration",
    #             y=1 - 0.08 / figh, fontsize=8)
    fig.supxlabel(r"Congestion price $|\mu_l|$ (\$/MWh)", y=0.00 / figh, fontsize=FONT["axlabel"])
    out = case_dir / f"mu_distribution_difference_{case_tag(case)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    return fig


def run(case, baseline, mrfs, bins=30, n_lines=15, mu_floor=0.0):
    """One difference figure for a case; the |mu| table is shared, only the law weights change."""
    c = get_case(case)
    base = create_problem(c, baseline)
    mu, line_names = base.mu_table()
    weights = {baseline: base.w}
    for mrf in mrfs:
        weights[mrf] = create_problem(c, mrf).w
    return plot_differences(c.name, mu, line_names, baseline, mrfs, weights,
                            mu_floor, bins, n_lines, c.results_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_v12_lev3a3")
    parser.add_argument("--baseline", default="bernoulli")
    parser.add_argument("--mrfs", default="mrf_a,mrf_cobuild,mrf_substitute")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--n-lines", type=int, default=5)
    parser.add_argument("--mu-floor", type=float, default=0.0,
                        help="lower bin edge; 0 includes the uncongested mass in the first bar")
    args = parser.parse_args()

    run(args.case, args.baseline, args.mrfs.split(","), args.bins, args.n_lines, args.mu_floor)
    plt.show()


if __name__ == "__main__":
    main()
