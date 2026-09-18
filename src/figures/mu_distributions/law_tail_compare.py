"""Exact mu tail under each nominal law, at marginals held fixed.

Every law in `sampling.distributions` puts the same probability on each individual
candidate, so the curves separate only where the *correlation* between candidates
matters.  Two panels, both exact by enumeration:

  left    survivor curve of the target S = max_b |mu_b|
  right   survivor curve of the element the laws disagree on most, other than the one
          driving S -- the check that the separation is not confined to a single element

The unstructured law (mrf_a) sits on top of the Bernoulli curve; the two structured
laws pull the tail apart in opposite directions.

Usage:
    uv run python -m src.figures.mu_distributions.law_tail_compare --case activsg2000_original_v12
    uv run python -m src.figures.mu_distributions.law_tail_compare --case activsg2000_original_v12 \
        --dists bernoulli,mrf_cobuild,mrf_substitute
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.cases.registry import get_case
from src.sampling.problem import case_tag, create_problem
from src.sampling.distributions import DIST_NAMES, DIST_LABEL

LAW_STYLE = {"bernoulli": ("k", "-"), "mrf_a": ("0.55", "-"),
             "mrf_cobuild": ("tab:blue", "-"), "mrf_substitute": ("tab:red", "-")}


def survivor(values, w, levels):
    """Exact P[values > level] under weights w, at each level."""
    return np.array([float(w[values > b].sum()) for b in levels])


def levels_for(values, n):
    """Levels spanning the upper half of the target's range, where the laws separate."""
    return np.linspace(float(np.median(values)), float(values.max()), n)


def bernoulli_levels(mu, w, at):
    """Per element, the |mu_l| level it exceeds with probability `at` under Bernoulli."""
    levels = []
    for l in range(mu.shape[1]):
        order = np.argsort(-mu[:, l])
        i = min(np.searchsorted(np.cumsum(w[order]), at), mu.shape[0] - 1)
        levels.append(float(mu[order[i], l]))
    return np.array(levels)


def spread_element(mu, names, weights, at, exclude, floor_frac=0.1):
    """The element whose exceedance spreads widest across the laws, at its own `at` level.

    Each element is read at the level it exceeds with probability `at` under the Bernoulli
    law, so elements on different mu scales are compared on the same footing.  Elements
    whose level is under `floor_frac` of the largest are skipped: their spread is a spread
    in how *often* they congest at all, which is not what these curves are about."""
    levels = bernoulli_levels(mu, weights["bernoulli"], at)
    best, best_spread = None, 0.0
    for l, name in enumerate(names):
        if name == exclude or levels[l] < floor_frac * levels.max():
            continue
        ps = [float(w[mu[:, l] > levels[l]].sum()) for w in weights.values()]
        spread = max(ps) / min(ps) if min(ps) > 0 else 0.0
        if spread > best_spread:
            best, best_spread = name, spread
    return best, best_spread


def law_panel(ax, values, weights, title, floor, n_levels):
    """One target's survivor curve per law, log y, exact enumeration."""
    levels = levels_for(values, n_levels)
    for name, w in weights.items():
        color, style = LAW_STYLE[name]
        ax.semilogy(levels, np.maximum(survivor(values, w, levels), 1e-12), color=color,
                    ls=style, lw=1.0, label=DIST_LABEL[name])
    ax.set_ylim(floor, 1.5)
    ax.set_xlim(levels[0], levels[-1])
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)


def plot_laws(case, weights, problem, floor, n_levels, at=1e-3):
    """Both panels plus the shared legend; returns the figure."""
    mu, names = problem.mu_table()
    element, spread = spread_element(mu, names, weights, at, problem.meta["one_mu_element"])
    print(f"second panel: {element}, laws spread by {spread:.0f}x at its {at:g} level")
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.3))
    law_panel(axs[0], problem.targets["max_mu"], weights,
              r"$S = \max_b |\mu_b|$", floor, n_levels)
    law_panel(axs[1], mu[:, names.index(element)], weights,
              rf"element {element}", floor, n_levels)
    axs[0].set_ylabel("exact nominal exceedance", fontsize=7)
    for ax in axs:
        ax.set_xlabel(r"level  $b$  [\$/MWh]", fontsize=7)
    axs[1].legend(fontsize=5.5, frameon=False, labelspacing=0.25, handlelength=1.6)
    fig.suptitle(f"{case.name}: mu tail at identical marginals", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_original_v12",
                        help="a registered case name")
    parser.add_argument("--dists", default=",".join(DIST_NAMES))
    parser.add_argument("--floor", type=float, default=1e-6,
                        help="smallest exceedance probability the y axis shows")
    parser.add_argument("--n-levels", type=int, default=120)
    args = parser.parse_args()

    case = get_case(args.case)
    problems = {d: create_problem(case, d) for d in args.dists.split(",")}
    weights = {d: pr.w for d, pr in problems.items()}
    fig = plot_laws(case, weights, next(iter(problems.values())), args.floor, args.n_levels)

    out = case.results_dir / f"law_tail_compare_{case_tag(case.name)}.pdf"
    fig.savefig(out)
    print(f"saved -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
