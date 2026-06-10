import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from utils.solution_io import load_dual_cache

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR



def plot_single_mu(mu_S, ax, label=None):
    n = len(mu_S)
    ypts = np.linspace(1, 0, n) + 1e-5 # include endpoint to compare across sample size
    ax.plot(mu_S.sort_values(), ypts, label=label)
    ax.set_title(mu_S.name, y=0.82, x=0.55)


def plot_separate_mu_cdf(mu_df):
    frac = 0.1
    sdf = mu_df.sample(frac=frac, replace=True, random_state=0)
    fig, axs = plt.subplots(4, 5, figsize=(5*1.8, 4*1.8), sharex=True, sharey=True)
    axs[0, 0].set_yscale("log") 
    for line, ax in zip(mu_df.columns, axs.ravel()):
        plot_single_mu(mu_df[line], ax, label="full")
        plot_single_mu(sdf[line], ax, label=f"{frac*100:.1f}%")
    fig.suptitle(r"CDF proportion $\mu_l > \hat{\mu}$ (by line number)")#, y=0.98)
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    fig.tight_layout()
    return fig
    


def main(result_path):
    df = load_dual_cache(result_path / "dual_cache.parquet")
    lmp_all, mu_all = df["lmp"], df["mu"]
    mu_all.columns = mu_all.columns.astype(int)
    mu_all.sort_index(axis=1, inplace=True)

    fig = plot_separate_mu_cdf(mu_all)
    cdf_path = result_path / f"mu_cdf_{result_path.name}.pdf"
    fig.savefig(cdf_path, bbox_inches="tight")
    
    plt.show()


if __name__ == "__main__":
    subdir = "case14_gen_per_bus"
    main(RESULTS_DIR / subdir)
