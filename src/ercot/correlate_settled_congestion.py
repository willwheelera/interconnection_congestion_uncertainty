"""
Correlate settled-price dispersion with base-case vs contingency congestion.

The research question (README Q2): do high congestion shadow prices (mu) actually
drive settled prices, and which regime -- base case or contingency -- is the
driver? Congestion shows up in prices as *spatial separation*: a binding limit
makes nodal LMPs diverge, whereas the energy/scarcity term lifts every node
together. So we use the per-interval cross-node LMP spread (max - min) as the
settled-price congestion footprint, and correlate it against the per-interval max
shadow price under each regime.

LMP (np6-788) shares the 5-min SCED clock with the shadow prices (np6-86), so the
two align in time after flooring to 5-min bins. The 15-min settlement-point price
(np6-905) is the looser "what actually settled" cross-check, left for follow-up.

Runs over the days in sample_days.csv, chosen to isolate the regimes.

Outputs (in results/):
    settled_vs_congestion.png   two-panel scatter, base vs contingency

Usage:
    uv run python src/grid_data/correlate_settled_congestion.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.ercot.shadow_prices import SHADOW_PRICES as DATA_PATH, load_shadow_prices

from src.ercot.data_paths import ERCOT_DATA as DATA_DIR
SAMPLE_DAYS_CSV = DATA_DIR / "sample_days.csv"
from src.ercot.data_paths import RESULTS as OUT_DIR
BIN = "5min"
REGIME_COLORS = {"base": "tab:blue", "mixed": "tab:purple", "contingency": "tab:red"}


def load_sample_days(path: Path) -> pd.DataFrame:
    """Read the sample delivery dates and their regime label."""
    return pd.read_csv(path)[["date", "regime"]]


def settled_dispersion(date: str) -> pd.DataFrame:
    """Per 5-min interval, the cross-node LMP spread for one day's np6-788 file.

    Columns (indexed by interval): lmp_range (max - min), lmp_std, lmp_max.
    """
    lmp = pd.read_parquet(DATA_DIR / f"lmp_node_zone_hub_{date}.parquet")
    lmp["t"] = pd.to_datetime(lmp["SCEDTimestamp"]).dt.floor(BIN)
    g = lmp.groupby("t")["LMP"]
    return pd.DataFrame(
        {"lmp_range": g.max() - g.min(), "lmp_std": g.std(), "lmp_max": g.max()}
    )


def congestion_by_regime(shadow_day: pd.DataFrame) -> pd.DataFrame:
    """Per 5-min interval, the max shadow price under each regime for one day.

    Columns (indexed by interval): base_shadow, cont_shadow. Non-binding
    constraints carry ShadowPrice 0, so an interval with no binding constraint in
    a regime reports 0 -- a real "no congestion" reading.
    """
    t = shadow_day["SCEDTimestamp"].dt.floor(BIN)
    is_base = shadow_day["ContingencyName"].str.strip() == "BASE CASE"
    base = shadow_day[is_base].groupby(t)["ShadowPrice"].max()
    cont = shadow_day[~is_base].groupby(t)["ShadowPrice"].max()
    return pd.DataFrame({"base_shadow": base, "cont_shadow": cont})


def build_interval_table(days: pd.DataFrame, shadow: pd.DataFrame) -> pd.DataFrame:
    """Join settled dispersion to regime congestion per interval, over all days."""
    shadow = shadow.assign(d=shadow["SCEDTimestamp"].dt.date)
    frames = []
    for date, regime in days.itertuples(index=False):
        day_shadow = shadow[shadow["d"] == pd.Timestamp(date).date()]
        tbl = settled_dispersion(date).join(congestion_by_regime(day_shadow))
        tbl[["base_shadow", "cont_shadow"]] = tbl[
            ["base_shadow", "cont_shadow"]
        ].fillna(0.0)
        tbl["date"], tbl["regime"] = date, regime
        frames.append(tbl)
    return pd.concat(frames).reset_index(names="interval")


def correlation_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Pearson and Spearman of LMP spread vs each regime's shadow price."""
    rows = []
    for col in ("base_shadow", "cont_shadow"):
        rows.append(
            {
                "signal": col,
                "pearson": table["lmp_range"].corr(table[col]),
                "spearman": table["lmp_range"].corr(table[col], method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def binned_median(x: pd.Series, y: pd.Series, n_bins: int = 12):
    """Median y within quantile bins of x; visualizes the monotonic trend."""
    bins = pd.qcut(x, n_bins, duplicates="drop")
    grouped = pd.DataFrame({"x": x, "y": y}).groupby(bins, observed=True)
    return grouped["x"].median(), grouped["y"].median()


def scatter_panel(ax: plt.Axes, table: pd.DataFrame, xcol: str, title: str) -> None:
    """Scatter LMP spread vs one regime's shadow price, colored by day regime.

    Symlog x absorbs the pile of zero-congestion intervals; log y compresses the
    extreme-spread tail. A binned-median line shows the underlying trend.
    """
    for regime, grp in table.groupby("regime"):
        ax.scatter(grp[xcol], grp["lmp_range"].clip(lower=1.0), s=8, alpha=0.3,
                   color=REGIME_COLORS[regime], label=regime)
    cx, cy = binned_median(table[xcol], table["lmp_range"].clip(lower=1.0))
    ax.plot(cx, cy, color="black", marker="o", ms=4, lw=1.5, label="binned median")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("log")
    r = table["lmp_range"].corr(table[xcol])
    rho = table["lmp_range"].corr(table[xcol], method="spearman")
    ax.set_title(f"{title}\nPearson r={r:.2f}   Spearman ρ={rho:.2f}")
    ax.set_xlabel("max shadow price in interval [$/MWh]")


def plot_correlation(table: pd.DataFrame, path: Path) -> None:
    """Two-panel scatter: settled-price dispersion vs base vs contingency mu."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    scatter_panel(axes[0], table, "base_shadow", "Base-case congestion")
    scatter_panel(axes[1], table, "cont_shadow", "Contingency congestion")
    axes[0].set_ylabel("cross-node LMP spread (max - min) [$/MWh]")
    axes[1].legend(title="series", markerscale=2)
    fig.suptitle("Settled-price dispersion vs congestion shadow price "
                 "(5-min SCED intervals, sample days)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    days = load_sample_days(SAMPLE_DAYS_CSV)
    shadow = load_shadow_prices(DATA_PATH)
    table = build_interval_table(days, shadow)
    print(correlation_summary(table).to_string(index=False))
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "settled_vs_congestion.png"
    plot_correlation(table, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
