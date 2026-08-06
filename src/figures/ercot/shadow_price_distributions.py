"""
Plot distributions of ERCOT binding-constraint shadow prices, per constraint.

The shadow-price feed (np6-86) reports a row for every constraint SCED monitors
each interval, not only the binding ones. Non-binding intervals have
ShadowPrice == 0 (flow Value below Limit, ViolatedMW <= 0); we treat the nonzero
shadow prices as the binding events whose distribution we care about.

Identifiers: ConstraintName is the physical monitored element (a line, or an
interface/aggregate transfer limit like NE_LOB with NaN stations). ConstraintID
is NOT an element key -- it only ranges 1-59, a per-interval rank slot. A
distinct constraint is (ConstraintName, ContingencyName): an element monitored
under an assumed contingency. We group per physical element, i.e. ConstraintName.

Usage:
    uv run python -m src.figures.ercot.shadow_price_distributions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm

from src.ercot.data_paths import RESULTS as OUT_DIR
from src.ercot.data_paths import SHADOW_PRICES as DATA_PATH
from src.ercot.shadow_prices import (binding_summary, calendar_grid, dominant_frequency,
                                 hourly_max_constraint, hourly_price_max,
                                 load_shadow_prices)
TOP_N = 20
TOP_DOMINANT = 10  # constraints given distinct colors in the dominance calendar




def plot_top_constraint_distributions(ax: plt.Axes, df: pd.DataFrame,
                                      names: list[str]) -> None:
    """Box-style distribution of binding shadow prices for the top constraints."""
    binding = df[df["ShadowPrice"] > 0]
    data = [binding.loc[binding["ConstraintName"] == n, "ShadowPrice"].values
            for n in names]
    ax.boxplot(data, vert=False, tick_labels=names, showfliers=True,
               flierprops=dict(marker=".", markersize=3, alpha=0.3))
    ax.set_xscale("log")
    ax.set_xlabel("Shadow price ($/MWh, log scale)")
    ax.set_title(f"Binding shadow-price distribution, top {len(names)} constraints")
    ax.tick_params(axis="y", labelsize=8)


def plot_overall_distribution(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Binding shadow-price distributions, base-case vs all contingencies.

    Two step-histogram lines on shared log bins: BASE CASE (no assumed outage,
    mostly interface/aggregate limits) and all contingency constraints pooled.
    """
    binding = df[df["ShadowPrice"] > 0]
    is_base = binding["ContingencyName"].str.strip() == "BASE CASE"
    base = binding.loc[is_base, "ShadowPrice"].values
    cont = binding.loc[~is_base, "ShadowPrice"].values
    all_vals = binding["ShadowPrice"].values
    bins = np.logspace(np.log10(all_vals.min()), np.log10(all_vals.max()), 80)
    ax.hist(base, bins=bins, histtype="step", color="C0", linewidth=1.5,
            label=f"BASE CASE (n={len(base)})")
    ax.hist(cont, bins=bins, histtype="step", color="C3", linewidth=1.5,
            label=f"contingencies (n={len(cont)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Shadow price ($/MWh, log scale)")
    ax.set_ylabel("Count (binding intervals)")
    ax.set_title("Binding shadow prices, 2025")
    ax.legend(fontsize=8)


def plot_price_timeseries(ax: plt.Axes, df: pd.DataFrame,
                          names: list[str]) -> None:
    """Binding shadow price vs time, one color per top constraint."""
    binding = df[df["ShadowPrice"] > 0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(names)))
    for name, color in zip(names, colors):
        sub = binding[binding["ConstraintName"] == name]
        ax.scatter(sub["SCEDTimestamp"], sub["ShadowPrice"], s=4, color=color,
                   label=name, alpha=0.5, edgecolors="none")
    ax.set_yscale("log")
    ax.set_ylabel("Shadow price ($/MWh, log scale)")
    ax.set_title(f"Binding shadow price over time, top {len(names)} constraints")
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.legend(fontsize=5, markerscale=2, ncol=2, loc="lower left")



def plot_hourly_price_sum(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Hourly total shadow price vs time, base-case vs all contingencies."""
    is_base = df["ContingencyName"].str.strip() == "BASE CASE"
    base_max = hourly_price_max(df[is_base])
    cont_max = hourly_price_max(df[~is_base])
    ax.scatter(cont_max.index, cont_max.values, color="C3", #linewidth=0.6,
            alpha=0.2, s=2,
            label="contingencies")
    ax.scatter(base_max.index, base_max.values, color="C0", #linewidth=0.6,
            alpha=0.2, s=2,
            label="BASE CASE")
    ax.set_ylabel("Max of hourly-mean shadow prices ($/MWh)")
    ax.set_title("Hourly total shadow price, 2025")
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.legend(fontsize=8)





def plot_concentration_curve(ax: plt.Axes, hmax: pd.DataFrame) -> None:
    """Hourly max shadow price sorted descending vs its rank (a duration curve)."""
    vals = np.sort(hmax["max_price"].values)[::-1]
    vals = vals[vals > 0]
    pct = np.arange(1, len(vals) + 1) / len(vals) * 100
    ax.plot(pct, vals, color="C3", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Percent of hours (ranked most to least severe)")
    ax.set_ylabel("Hourly max shadow price ($/MWh, log scale)")
    ax.set_title("Concentration of hourly max congestion, 2025")


def plot_max_calendar(ax: plt.Axes, hmax: pd.DataFrame) -> None:
    """Calendar heatmap (hour-of-day x day-of-year) of the hourly max price."""
    grid = calendar_grid(hmax["max_price"])
    grid = np.where(grid > 0, grid, np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    pos = grid[np.isfinite(grid)]
    norm = LogNorm(vmin=pos.min(), vmax=pos.max())
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   extent=[1, grid.shape[1], 0, 24], cmap=cmap, norm=norm)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Hour of day")
    ax.set_title("Hourly max shadow price by hour and day")
    plt.colorbar(im, ax=ax, label="Max shadow price ($/MWh)")


def plot_dominant_frequency(ax: plt.Axes, freq: pd.Series,
                            names: list[str]) -> None:
    """Bar chart of how often each constraint is the system-wide hourly max."""
    ax.barh(names, freq.loc[names].values, color="C0")
    ax.invert_yaxis()
    ax.set_xlabel("Hours as the system max")
    ax.set_title(f"How often each constraint is the hourly max, top {len(names)}")
    ax.tick_params(axis="y", labelsize=7)


def plot_dominant_calendar(ax: plt.Axes, hmax: pd.DataFrame,
                           names: list[str]) -> None:
    """Calendar heatmap colored by which constraint was the hourly max.

    The top constraints in `names` get distinct colors; all others share one
    "other" color, and hours with no congestion are left blank.
    """
    code = {name: i for i, name in enumerate(names)}
    codes = hmax["max_constraint"].map(lambda name: code.get(name, len(names)))
    codes = codes.where(hmax["max_price"] > 0)
    grid = calendar_grid(codes)
    colors = list(plt.get_cmap("tab10")(np.arange(len(names))))
    colors.append((0.6, 0.6, 0.6, 1.0))
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(len(names) + 2) - 0.5, cmap.N)
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   extent=[1, grid.shape[1], 0, 24], cmap=cmap, norm=norm)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Hour of day")
    ax.set_title("Dominant (max) constraint by hour and day")
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(len(names) + 1))
    cbar.ax.set_yticklabels(names + ["other"], fontsize=6)


def report_inclusion(df: pd.DataFrame) -> None:
    """Print why zeros and NaN stations appear in the feed."""
    n = len(df)
    zero_frac = (df["ShadowPrice"] == 0).mean()
    nan_frac = df["FromStation"].isna().mean()
    base = df["ContingencyName"].str.strip().eq("BASE CASE")
    print(f"{n} rows; {zero_frac:.1%} have ShadowPrice == 0 (constraint monitored "
          "but not binding, ViolatedMW <= 0).")
    print(f"{nan_frac:.1%} have NaN FromStation/ToStation; "
          f"{base[df['FromStation'].isna()].mean():.1%} of those are BASE CASE "
          "constraints (interfaces/aggregate limits, not a single from->to branch).")


def build_distribution_figure(df: pd.DataFrame, summary: pd.DataFrame,
                              top_names: list[str]) -> plt.Figure:
    """Four-panel figure of binding shadow-price distributions over time."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_overall_distribution(axes[0, 0], df)
    plot_hourly_price_sum(axes[0, 1], df)
    plot_top_constraint_distributions(axes[1, 0], df, top_names)
    plot_price_timeseries(axes[1, 1], df, top_names)
    fig.tight_layout()
    return fig


def build_concentration_figure(df: pd.DataFrame) -> plt.Figure:
    """Four-panel figure on hourly-max congestion: concentration and dominance."""
    hmax = hourly_max_constraint(df)
    freq = dominant_frequency(hmax)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_concentration_curve(axes[0, 0], hmax)
    plot_max_calendar(axes[0, 1], hmax)
    plot_dominant_frequency(axes[1, 0], freq, freq.head(TOP_N).index.tolist())
    plot_dominant_calendar(axes[1, 1], hmax, freq.head(TOP_DOMINANT).index.tolist())
    fig.tight_layout()
    return fig


def main() -> None:
    df = load_shadow_prices(DATA_PATH)
    report_inclusion(df)
    summary = binding_summary(df)
    top_names = summary.head(TOP_N).index.tolist()

    OUT_DIR.mkdir(exist_ok=True)
    fig1 = build_distribution_figure(df, summary, top_names)
    fig1.savefig(OUT_DIR / "shadow_price_distributions.png", dpi=150,
                 bbox_inches="tight")

    fig2 = build_concentration_figure(df)
    fig2.suptitle("All constraints (base case + contingencies)", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "congestion_concentration.png", dpi=150,
                 bbox_inches="tight")

    base = df[df["ContingencyName"].str.strip() == "BASE CASE"]
    fig3 = build_concentration_figure(base)
    fig3.suptitle("Base case only (no assumed outage)", fontsize=12)
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "congestion_concentration_base_case.png", dpi=150,
                 bbox_inches="tight")

    print(f"saved figures to {OUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
