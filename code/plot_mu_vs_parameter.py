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
    g = sns.FacetGrid(data=df, col="load", hue=var, col_wrap=4, height=4.5)
    g.map(plt.plot, "scale", val)
    congested = mu_all.sum(axis=0).abs() > 1
    select = [[g._legend_data[k], k] for k in congested.index[congested]]
    g.axes[-1].legend(*list(zip(*select)), title=var, ncol=2, loc="center left", bbox_to_anchor=(1.2, 0.5))
    plt.savefig(path, bbox_inches="tight")


def plot_seaborn_combined(df_dict, path, var):
    """One figure with a row per variable in df_dict and a column per load; hue = `var`."""
    print(f"saving to {path}")
    frames = []
    for key, mu_all in df_dict.items():
        d = mu_all.reset_index(names=["load", "scale"])
        d = d.melt(id_vars=["load", "scale"], value_vars=mu_all.columns, var_name=var, value_name="value")
        d["variable"] = key
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    g = sns.FacetGrid(data=df, row="variable", col="load", hue=var,
                     height=3.0, sharey="row", row_order=list(df_dict))
    g.map(plt.plot, "scale", "value")
    g.add_legend(title=var)
    plt.savefig(path, bbox_inches="tight")


def main(result_path):
    df = load_dual_cache(result_path, scan="load")
    df = df.loc[540]
    loads = [9, 13, 14]
    df = df.loc[loads]

    bus_vars = ["lmp", "injection", "va"]
    line_vars = ["mu", "flow", "nu"]
    plot_seaborn_combined({k: df[k] for k in bus_vars},
                          result_path.parent / "bus_vars_vs_load.pdf", "bus")
    plot_seaborn_combined({k: df[k] for k in line_vars},
                          result_path.parent / "line_vars_vs_load.pdf", "line")

    plt.show()


if __name__ == "__main__":
    import sys
    #subdir = sys.argv[1] if len(sys.argv) > 1 else "test_scuc_masked"
    subdir = "case14_gen_per_bus"

    #build_dual_cache(RESULTS_DIR / subdir)
    main(RESULTS_DIR / subdir / "scale_load_cache.parquet")
