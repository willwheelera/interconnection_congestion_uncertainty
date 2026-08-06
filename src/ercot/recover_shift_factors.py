"""
Recover empirical shift factors relating nodal LMPs to constraint shadow prices.

The DC price-formation identity is LMP_n = lambda - sum_k SF_{k,n} mu_k, where
SF_{k,n} is the shift factor of constraint k to node n. Differencing each node
from a reference hub cancels the energy term lambda:

    LMP_n - LMP_hub = - sum_k (SF_{k,n} - SF_{hub,n}) mu_k

so a least-squares fit of the hub-differenced LMP on the vector of constraint
shadow prices recovers the shift factors as the coefficients -- a node x
constraint matrix -- from price data alone, with no network model. A constraint
is keyed by (ConstraintName, ContingencyName): a line monitored under a
contingency sits on a different (post-outage) network than the same line in the
base case, so its shift-factor vector differs and is fit separately.

Shift factors are a property of the network impedances and topology, so they
should be stable across days unless a line is switched or on outage. We fit one
pooled regression over all sample days (more binding patterns, less collinearity).
Per-constraint coefficients are not separately identified when constraints co-bind
(only the spanned subspace is), so we test stability the valid way: apply the
pooled shift factors to each day's mu and measure out-of-sample reconstruction of
that day's prices. If the map generalizes, the topology is stable; a day fits
poorly only when its drivers fall outside the pooled set (low coverage) or its
price separation is too small to see over losses.

Outputs (in results/):
    shift_factors_pooled.csv      pooled node x constraint shift-factor matrix
    shift_factor_oos.csv          per-day out-of-sample R2 and constraint coverage
    shift_factor_oos.png          bar chart of out-of-sample R2 by day and regime

Usage:
    uv run python src/grid_data/recover_shift_factors.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ercot.shadow_prices import SHADOW_PRICES as DATA_PATH, load_shadow_prices

from src.ercot.data_paths import ERCOT_DATA as DATA_DIR
SAMPLE_DAYS_CSV = DATA_DIR / "sample_days.csv"
from src.ercot.data_paths import RESULTS as OUT_DIR
REF_HUB = "HB_HUBAVG"
BIN = "5min"
REGIME_COLORS = {"base": "tab:blue", "mixed": "tab:purple", "contingency": "tab:red"}
MIN_BIND_POOL = 30  # intervals a constraint must bind (mu>0) to be a pooled regressor
RCOND = 1e-8        # singular-value cutoff: truncate near-null (collinear) directions


def load_sample_days(path: Path) -> pd.DataFrame:
    """Read the sample delivery dates and their regime label."""
    return pd.read_csv(path)[["date", "regime"]]


def lmp_diff_matrix(date: str) -> pd.DataFrame:
    """Intervals x nodes matrix of LMP minus the reference hub, for one day."""
    lmp = pd.read_parquet(DATA_DIR / f"lmp_node_zone_hub_{date}.parquet")
    lmp["t"] = pd.to_datetime(lmp["SCEDTimestamp"]).dt.floor(BIN)
    wide = lmp.groupby(["t", "SettlementPoint"])["LMP"].mean().unstack()
    return wide.sub(wide[REF_HUB], axis=0)


def mu_matrix(shadow_day: pd.DataFrame) -> pd.DataFrame:
    """Intervals x (ConstraintName, ContingencyName) matrix of max shadow price."""
    shadow_day = shadow_day.copy()
    shadow_day["t"] = shadow_day["SCEDTimestamp"].dt.floor(BIN)
    return shadow_day.pivot_table(
        index="t",
        columns=["ConstraintName", "ContingencyName"],
        values="ShadowPrice",
        aggfunc="max",
    )


def day_panels(days: pd.DataFrame, shadow: pd.DataFrame) -> dict:
    """Per-day aligned (LMP-diff, mu) panels keyed by date."""
    shadow = shadow.assign(d=shadow["SCEDTimestamp"].dt.date)
    panels = {}
    for date in days["date"]:
        diff = lmp_diff_matrix(date)
        mu = mu_matrix(shadow[shadow["d"] == pd.Timestamp(date).date()])
        idx = diff.index.intersection(mu.index)
        panels[date] = (diff.loc[idx], mu.loc[idx])
    return panels


def regress(diff: pd.DataFrame, mu: pd.DataFrame):
    """Fit hub-differenced LMP on the mu vector.

    Returns (coef, r2): coef is a constraint x node DataFrame of shift factors;
    r2 is the per-node coefficient of determination. Nodes with any missing
    interval are dropped so the least-squares fit is dense. Constraints that
    co-bind every interval are collinear, so the minimum-norm (truncated-SVD)
    solution is used to keep coefficients bounded instead of blowing up.
    """
    diff = diff.dropna(axis=1)
    x = np.column_stack([np.ones(len(mu)), mu.values])
    beta, *_ = np.linalg.lstsq(x, diff.values, rcond=RCOND)
    pred = x @ beta
    ss_res = ((diff.values - pred) ** 2).sum(axis=0)
    ss_tot = ((diff.values - diff.values.mean(axis=0)) ** 2).sum(axis=0)
    r2 = pd.Series(1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot),
                   index=diff.columns)
    coef = pd.DataFrame(beta[1:], index=mu.columns, columns=diff.columns)
    return coef, r2


def pooled_fit(panels: dict):
    """Pool all days and fit, keeping constraints that bind on enough intervals."""
    diff = pd.concat([d for d, _ in panels.values()])
    mu = pd.concat([m for _, m in panels.values()]).fillna(0.0)
    mu = mu.loc[:, (mu > 0).sum() >= MIN_BIND_POOL]
    coef, r2 = regress(diff, mu)
    return coef, r2, mu


def regime_attribution(coef: pd.DataFrame, mu: pd.DataFrame) -> pd.DataFrame:
    """Base vs contingency shares of price-separation work, two ways.

    structural: sum of |shift factor| over nodes per constraint -- the magnitude
        of each constraint's price footprint, ignoring how often it binds.
    realized: structural times sum of mu over intervals -- the footprint weighted
        by how much the constraint actually bound over the sample.
    Also reports the constraint count per regime.
    """
    struct = coef.abs().sum(axis=1)
    realized = struct * mu.sum(axis=0)
    regime = np.where(
        coef.index.get_level_values("ContingencyName").str.strip() == "BASE CASE",
        "base", "contingency")
    out = pd.DataFrame({"structural": struct, "realized": realized}).groupby(
        regime).sum()
    out = out / out.sum()
    out["n_constraints"] = pd.Series(regime).value_counts()
    return out


def oos_reconstruction(panels: dict, coef: pd.DataFrame) -> pd.DataFrame:
    """Per-day out-of-sample fit of the pooled shift factors, plus coverage.

    Applies the pooled constraint x node matrix to each day's mu (no refitting)
    and reports the median per-node reconstruction R2. coverage is the share of
    that day's total binding mu carried by constraints in the pooled set -- a low
    R2 with low coverage means the day's drivers were simply out of scope, not a
    topology change.
    """
    nodes, cols = coef.columns, coef.index
    rows = {}
    for date, (diff, mu) in panels.items():
        d = diff.reindex(columns=nodes).dropna(axis=1)
        full = mu.fillna(0.0)
        pred = full.reindex(columns=cols, fill_value=0.0).values @ coef[d.columns].values
        res = ((d.values - pred) ** 2).sum(axis=0)
        tot = ((d.values - d.values.mean(axis=0)) ** 2).sum(axis=0)
        r2 = 1 - res / np.where(tot == 0, np.nan, tot)
        coverage = full[full.columns.intersection(cols)].sum().sum() / full.sum().sum()
        rows[date] = {"oos_r2": np.nanmedian(r2), "coverage": coverage}
    return pd.DataFrame(rows).T


def plot_oos(oos: pd.DataFrame, regime: pd.Series, path: Path) -> None:
    """Bar chart of per-day out-of-sample R2, colored by regime, coverage labeled."""
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [REGIME_COLORS[regime[d]] for d in oos.index]
    bars = ax.bar(range(len(oos)), oos["oos_r2"], color=colors)
    for bar, cov in zip(bars, oos["coverage"]):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                f"cov {cov:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(oos)), oos.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("out-of-sample median R²")
    ax.set_title("Pooled shift factors applied to each day (out-of-sample)\n"
                 "high R² = stable, generalizing price-formation map")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in REGIME_COLORS.values()]
    ax.legend(handles, REGIME_COLORS.keys(), title="day regime")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    days = load_sample_days(SAMPLE_DAYS_CSV)
    shadow = load_shadow_prices(DATA_PATH)
    panels = day_panels(days, shadow)

    coef, r2, mu = pooled_fit(panels)
    print(f"pooled fit: {coef.shape[0]} constraints x {coef.shape[1]} nodes")
    print(f"median per-node R2: {r2.median():.3f}  "
          f"(nodes R2>0.9: {(r2 > 0.9).mean():.2f})")

    print("\nprice-separation share by regime (pooled):")
    print(regime_attribution(coef, mu).round(3).to_string())

    oos = oos_reconstruction(panels, coef)
    print("\nout-of-sample reconstruction by day:")
    print(oos.round(3).to_string())

    OUT_DIR.mkdir(exist_ok=True)
    coef.T.to_csv(OUT_DIR / "shift_factors_pooled.csv")
    oos.to_csv(OUT_DIR / "shift_factor_oos.csv")
    plot_oos(oos, days.set_index("date")["regime"], OUT_DIR / "shift_factor_oos.png")
    print(f"\nsaved matrix, oos table, and bar chart to {OUT_DIR}")


if __name__ == "__main__":
    main()
