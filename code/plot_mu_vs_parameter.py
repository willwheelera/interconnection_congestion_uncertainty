from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from utils.solution_io import load_dual_cache

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR


def plot(mu_all):
    n = mu_all.index.get_level_values(0).nunique()
    ncol_ax = int(np.ceil(n / 3))
    fig, axs = plt.subplots(3, ncol_ax, figsize=(10, 7), sharex=True, sharey=True)
    axs = axs.ravel()
    for i, (key, df) in enumerate(mu_all.groupby(level=0)):
        axs[i].plot(df.loc[key])
        axs[i].set_title(f"load{key}")
    plt.tight_layout()
    plt.show()


def plot_seaborn(mu_all, path, val="mu", var="line"):
    print(f"saving to {path}")
    df = mu_all.reset_index(names=["load", "scale"])
    df = df.melt(id_vars=["load", "scale"], value_vars=mu_all.columns, var_name=var, value_name=val)
    g = sns.FacetGrid(data=df, col="load", hue=var, col_wrap=4, height=2.5)
    g.map(plt.plot, "scale", val)
    congested = mu_all.sum(axis=0) > 1
    select = [[g._legend_data[k], k] for k in congested.index[congested]]
    g.axes[-1].legend(*list(zip(*select)), title=var, ncol=2, loc="center left", bbox_to_anchor=(1.2, 0.5))
    plt.savefig(path, bbox_inches="tight")


def main(result_path):
    lmp_all, mu_all = load_dual_cache(result_path, "*_load*")
    lmp_all = lmp_all.loc[540] # remove avail_id index level
    mu_all = mu_all.loc[540] # remove avail_id index level
    loads = [9, 13, 14]
    mu_all = mu_all.loc[loads]
    lmp_all = lmp_all.loc[loads]

    plot_seaborn(mu_all, result_path.parent / "mu_vs_load.pdf")
    plot_seaborn(lmp_all, result_path.parent / "lmp_vs_load.pdf", "lmp", "bus")

    plt.show()


if __name__ == "__main__":
    import sys
    #subdir = sys.argv[1] if len(sys.argv) > 1 else "test_scuc_masked"
    subdir = "case14_gen_per_bus"

    #build_dual_cache(RESULTS_DIR / subdir)
    main(RESULTS_DIR / subdir / "scale_load_cache.parquet")
