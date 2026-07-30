"""Plot the estimated congestion-target tail from sample_tail's saved sample.

Draws the subset-simulation exceedance curve P[S >= b] on a log y-axis and marks the
conditioning thresholds b_k, giving the whole tail of the load-averaged congestion
target over the 2^20 queue-generator configuration space.

Usage:
    uv run python src/texas/plot_tail.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parents[2]
SAMPLE = REPO_ROOT / "results" / "activsg2000" / "tail_sample.npz"


def plot_exceedance(ax, thresholds, ccdf, levels) -> None:
    """Exceedance curve with conditioning thresholds marked."""
    pos = ccdf > 0
    ax.semilogy(thresholds[pos], ccdf[pos], color="C0", lw=2, label="subset simulation")
    for b in levels:
        ax.axvline(b, color="0.7", ls=":", lw=0.8)
    ax.set_xlabel("congestion target $b$  (max load-averaged $\\mu$, \\$/MWh)")
    ax.set_ylabel("$P[S \\geq b]$")
    ax.set_title("ACTIVSg2000 queue-config congestion tail")
    ax.legend()


def main():
    data = np.load(SAMPLE)
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_exceedance(ax, data["thresholds"], data["ccdf"], data["levels"])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
