"""Inspect the random-MRF target used in informed_mcmc.main.

Computes:
- Per-generator marginal P(x_k = 1) under the MRF target (restricted to the
  cached 16383 feasible configs).
- The true distribution of sum_b mu_b under that target (concentrated at
  zero, heavy-tailed) and per-line marginals for a couple of lines.

Mirrors the target construction in informed_mcmc.main:
    target = MRF.random(NGEN, seed=42, a_scale=0.8, b_scale=1.0)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from informed_mcmc import MRF, NGEN, RESULTS_DIR, true_distribution
from utils.availability import id_to_avail


def marginals(p: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Per-bit marginals P(x_k = 1) given a probability over configurations."""
    X = np.stack([id_to_avail(int(i), NGEN) for i in ids])
    return X.T @ p


def main():
    df = pd.read_parquet(RESULTS_DIR / "dual_cache.parquet").sort_index()
    target = MRF.random(NGEN, seed=42, a_scale=0.8, b_scale=1.0)
    p_true, F_true = true_distribution(target, df)
    ids = df.index.to_numpy()

    pi = marginals(p_true, ids)
    print("MRF.random(NGEN, seed=42, a_scale=0.8, b_scale=1.0)")
    print(f"effective support (p > 1e-5): {(p_true > 1e-5).sum()} / {len(p_true)}")
    print(f"max p(x): {p_true.max():.4f}")
    print()
    print("Per-generator marginals P(x_k = 1):")
    for k, m in enumerate(pi):
        bar = "#" * int(round(m * 40))
        print(f"  gen {k + 1:2d}  {m:.3f}  {bar}")
    print(f"  mean marginal: {pi.mean():.3f}  (uniform-on-feasible baseline ~{0.5:.2f})")

    mu_cols = [i for i, c in enumerate(df.columns) if c[0] == "mu"]
    lmp_cols = [i for i, c in enumerate(df.columns) if c[0] == "lmp"]
    f_sum_mu = F_true[:, mu_cols].sum(axis=1)
    f_sum_lmp = F_true[:, lmp_cols].sum(axis=1)

    pct = np.percentile(f_sum_mu, [50, 90, 99, 100])
    mean = (p_true * f_sum_mu).sum()
    p_zero = p_true[f_sum_mu < 1e-6].sum()
    print()
    print(f"sum_b mu_b under MRF target:  mean={mean:.2f}  "
          f"median={pct[0]:.2f}  p90={pct[1]:.2f}  p99={pct[2]:.2f}  max={pct[3]:.2f}")
    print(f"P(sum_b mu_b ~ 0) = {p_zero:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].bar(np.arange(1, NGEN + 1), pi, color="tab:blue")
    axes[0, 0].axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.6, label="0.5")
    axes[0, 0].set_xlabel("generator k")
    axes[0, 0].set_ylabel(r"$P(x_k = 1)$")
    axes[0, 0].set_title("Per-generator marginal availability under MRF")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(fontsize=9)

    bins = np.linspace(0, max(f_sum_mu.max(), 1e-6), 80)
    h, _ = np.histogram(f_sum_mu, bins=bins, weights=p_true)
    centers = 0.5 * (bins[1:] + bins[:-1])
    width = bins[1] - bins[0]
    axes[0, 1].bar(centers, h / width, width=width, color="tab:purple", alpha=0.8)
    axes[0, 1].axvline(mean, color="k", ls="--", lw=1, label=f"mean = {mean:.1f}")
    axes[0, 1].set_xlabel(r"$\sum_b \mu_b$")
    axes[0, 1].set_ylabel("density")
    axes[0, 1].set_title(r"True distribution of $\sum_b \mu_b$ (linear)")
    axes[0, 1].legend(fontsize=9)

    axes[1, 0].bar(centers, h / width + 1e-12, width=width, color="tab:purple", alpha=0.8)
    axes[1, 0].axvline(mean, color="k", ls="--", lw=1, label=f"mean = {mean:.1f}")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel(r"$\sum_b \mu_b$")
    axes[1, 0].set_ylabel("density (log)")
    axes[1, 0].set_title(r"True distribution of $\sum_b \mu_b$ (log y, heavy tail)")
    axes[1, 0].legend(fontsize=9)

    line = 12
    vals = F_true[:, mu_cols[line]]
    bins_l = np.linspace(0, max(vals.max(), 1e-6), 60)
    hl, _ = np.histogram(vals, bins=bins_l, weights=p_true)
    centers_l = 0.5 * (bins_l[1:] + bins_l[:-1])
    width_l = bins_l[1] - bins_l[0]
    axes[1, 1].bar(centers_l, hl / width_l + 1e-12, width=width_l, color="tab:orange", alpha=0.8)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel(fr"$\mu$ on line {line + 1}")
    axes[1, 1].set_ylabel("density (log)")
    axes[1, 1].set_title(fr"Per-line $\mu_{{{line + 1}}}$ marginal (log y)")

    fig.tight_layout()
    out = RESULTS_DIR / "check_mrf_target.pdf"
    fig.savefig(out)
    print(f"\nfigure saved to {out}")


if __name__ == "__main__":
    main()
