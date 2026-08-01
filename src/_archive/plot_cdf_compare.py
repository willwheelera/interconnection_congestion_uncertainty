"""Compare sampled inverse-CDFs of mu per line: p-only vs dmu-biased vs truth.

Mirrors ``plot_cdf_sampled`` but overlays the two weighted samplers from
``mcmc_fake_sensitivity`` against the exact weighted distribution under the
target MRF (exact enumeration over the cached configs).  Each panel shows the
cdf curve ``proportion mu_l > mu_hat`` for one line; a good sampler tracks
the black truth curve, especially in the congested (mu > 0) tail.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from informed_mcmc import MRF
from mcmc_fake_sensitivity import (
    NGEN,
    RESULTS_DIR,
    all_unnormalized_p,
    load_mu_table,
)

ALGO_COLORS = {"truth": "k", "p-only": "tab:blue", "dmu-biased": "tab:red"}


def weighted_cdf(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sorted values, proportion of weight with mu > value)``."""
    order = np.argsort(values)
    v = values[order]
    cumw = np.zeros(weights.shape)
    cumw[1:] = np.cumsum(weights[order])[:-1]
    return v, 1.0 - cumw / cumw[-1]


def plot_cdf(ax, frame: pd.DataFrame, line: str, label: str, color: str):
    """Draw one weighted cdf curve for ``line`` onto ``ax``."""
    v, cdf = weighted_cdf(frame[line].to_numpy(), frame["weight"].to_numpy())
    #cdf += 1e-8
    label = label.replace("biased", "bias")
    ax.plot(v, cdf, label=label, color=color, lw=1.3)
    ax.set_title(line, y=0.82, x=0.55)


def truth_frame(target: MRF, tol=1e-1) -> tuple[pd.DataFrame, list[str]]:
    """Exact weighted distribution: every valid config with weight ``p(x)``."""
    table, line_names = load_mu_table(RESULTS_DIR)
    p = all_unnormalized_p(target)
    present = np.nonzero(p)[0]
    df = pd.DataFrame(table[present], columns=line_names)
    zero_mu = df.sum(axis=1) < tol
    df["weight"] = p[present]
    # compress zeros
    newrow = df[zero_mu].sum(axis=0)
    df = df[~zero_mu]
    df.loc[-1] = newrow
    df["method"] = "truth"
    return df, line_names


def plot_compare(data: dict, line_names: list[str]):
    """Grid of per-line cdf curves overlaying truth and both samplers."""
    lines = sorted(line_names, key=int)
    fig, axs = plt.subplots(4, 5, figsize=(5 * 1.8, 4 * 1.8), sharex=True, sharey=True)
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylim([10**(-5), 1.5])
    for line, ax in zip(lines, axs.ravel()):
        for method in ("truth", "p-only", "dmu-biased"):
            plot_cdf(ax, data[method], line, method, ALGO_COLORS[method])
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.suptitle(r"Inverse CDF: proportion $\mu_l > \hat{\mu}$ (by line number)", y=0.99)
    fig.supxlabel(r"Congestion threshold $\hat{\mu}$")
    fig.supylabel(r"Probability of exceeding $\hat{\mu}$")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96),
               ncol=3, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def main():
    target = MRF.random(NGEN, seed=42, a_scale=0.8, b_scale=1.0)
    samples = pd.read_parquet(RESULTS_DIR / "fake_sensitivity_samples.parquet")
    truth, line_names = truth_frame(target)
    data = {a: df for a, df in samples.groupby("method")}
    data["truth"] = truth

    fig = plot_compare(data, line_names)
    out_path = RESULTS_DIR / "fake_sensitivity_cdf.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
