"""Shared plotting parameters and helpers for the sampler-comparison figures.

Sampler colors and labels, plus the standard median + 10-90 replicate band, so every
plot in the package reads samplers with one consistent color, name, and band style.
"""

import numpy as np

COLORS = {
    "direct": "tab:red", "subset_sim": "tab:blue", "ascent_is": "tab:green",
    "smc_tilt": "tab:purple", "ascent_exact": "tab:orange",
}
LABELS = {
    "direct": "direct",
    "subset_sim": "subset sim",
    "ascent_is": "ascent-IS",
    "smc_tilt": "sequential tilt",
    "ascent_exact": "ascent-IS+exact",
    "cross_entropy": "cross-entropy",
}


def band(ax, b, est, color, label):
    """Median curve with a 10-90 percentile band over replicates."""
    med = np.median(est, axis=0)
    lo, hi = np.percentile(est, [10, 90], axis=0)
    ax.plot(b, med, color=color, lw=2, label=label)
    ax.fill_between(b, lo, hi, color=color, alpha=0.18)
