"""
Attribute settled-price separation to base-case vs contingency congestion, by time.

Settlement-point prices (np6-905, 15-min) are what actually settled. We align them
to the shadow prices (np6-86) by flooring both to 15-min bins, measure the settled
congestion footprint as the cross-point SPP spread (max - min), and tag each
interval by which regime is binding: base-case, contingency, both, or neither (a
constraint counts as binding when its max shadow price clears BIND). This answers
how often -- and at what hours -- the realized settled prices line up with base vs
contingency congestion. The 5-min node/zone/hub LMP (np6-788), aggregated to
15-min, is a cross-check that the settled price tracks the SCED price.

Runs over the days in sample_days.csv.

Outputs (in results/):
    settlement_regime_when.png   class frequency among congested intervals and
                                 their hour-of-day timing

Usage:
    uv run python src/grid_data/settlement_regime_split.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ercot.shadow_prices import SHADOW_PRICES as DATA_PATH, load_shadow_prices

from ercot.data_paths import ERCOT_DATA as DATA_DIR
SAMPLE_DAYS_CSV = DATA_DIR / "sample_days.csv"
from ercot.data_paths import RESULTS as OUT_DIR
BIN = "15min"
BIND = 50.0   # $/MWh shadow price above which a regime counts as binding
DISP = 50.0   # $/MWh settled spread above which an interval is "congested"
CLASSES = ["base-only", "both", "contingency-only", "neither"]
CLASS_COLORS = {"base-only": "tab:blue", "both": "tab:purple",
                "contingency-only": "tab:red", "neither": "lightgray"}


def load_sample_days(path: Path) -> pd.DataFrame:
    """Read the sample delivery dates and their regime label."""
    return pd.read_csv(path)[["date", "regime"]]


def spp_dispersion(date: str) -> pd.Series:
    """Cross-point settled-price spread (max - min) per 15-min interval, one day."""
    sp = pd.read_parquet(DATA_DIR / f"spp_node_zone_hub_{date}.parquet")
    t = (pd.to_datetime(sp["DeliveryDate"])
         + pd.to_timedelta(sp["DeliveryHour"] - 1, "h")
         + pd.to_timedelta((sp["DeliveryInterval"] - 1) * 15, "m"))
    g = sp.groupby(t)["SettlementPointPrice"]
    return (g.max() - g.min()).rename("spp_spread")


def lmp_dispersion(date: str) -> pd.Series:
    """Cross-node LMP spread per 15-min interval (5-min SCED aggregated), one day."""
    lmp = pd.read_parquet(DATA_DIR / f"lmp_node_zone_hub_{date}.parquet")
    lmp["t"] = pd.to_datetime(lmp["SCEDTimestamp"]).dt.floor(BIN)
    point_mean = lmp.groupby(["t", "SettlementPoint"])["LMP"].mean()
    g = point_mean.groupby(level="t")
    return (g.max() - g.min()).rename("lmp_spread")


def regime_mu(shadow_day: pd.DataFrame) -> pd.DataFrame:
    """Max base-case and contingency shadow price per 15-min interval, one day."""
    t = shadow_day["SCEDTimestamp"].dt.floor(BIN)
    is_base = shadow_day["ContingencyName"].str.strip() == "BASE CASE"
    base = shadow_day[is_base].groupby(t)["ShadowPrice"].max()
    cont = shadow_day[~is_base].groupby(t)["ShadowPrice"].max()
    return pd.DataFrame({"base_mu": base, "cont_mu": cont})


def build_table(days: pd.DataFrame, shadow: pd.DataFrame) -> pd.DataFrame:
    """Join settled spread, LMP spread, and regime mu per 15-min interval, all days."""
    shadow = shadow.assign(d=shadow["SCEDTimestamp"].dt.date)
    frames = []
    for date, regime in days.itertuples(index=False):
        day_shadow = shadow[shadow["d"] == pd.Timestamp(date).date()]
        tbl = pd.concat([spp_dispersion(date), lmp_dispersion(date),
                         regime_mu(day_shadow)], axis=1)
        tbl[["base_mu", "cont_mu"]] = tbl[["base_mu", "cont_mu"]].fillna(0.0)
        tbl["date"], tbl["regime"] = date, regime
        frames.append(tbl.reset_index(names="interval"))
    return pd.concat(frames, ignore_index=True)


def classify(table: pd.DataFrame) -> pd.DataFrame:
    """Label each interval base-only / both / contingency-only / neither."""
    b, c = table["base_mu"] >= BIND, table["cont_mu"] >= BIND
    table = table.copy()
    table["class"] = "neither"
    table.loc[b & ~c, "class"] = "base-only"
    table.loc[b & c, "class"] = "both"
    table.loc[~b & c, "class"] = "contingency-only"
    table["hour"] = pd.to_datetime(table["interval"]).dt.hour
    return table


def plot_when(congested: pd.DataFrame, path: Path) -> None:
    """Class frequency among congested intervals and their hour-of-day timing."""
    counts = congested["class"].value_counts().reindex(CLASSES).fillna(0)
    by_hour = (congested.groupby(["hour", "class"]).size()
               .unstack("class").reindex(columns=CLASSES).fillna(0))
    colors = [CLASS_COLORS[c] for c in CLASSES]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(CLASSES, counts.values, color=colors)
    axes[0].set_ylabel("congested 15-min intervals")
    axes[0].set_title(f"How often settled congestion is base vs contingency\n"
                      f"(spread > ${DISP:.0f}, binding > ${BIND:.0f})")
    axes[0].tick_params(axis="x", rotation=20)
    by_hour.plot.bar(stacked=True, ax=axes[1], color=colors, width=0.9, legend=False)
    axes[1].set_xlabel("hour of day")
    axes[1].set_ylabel("congested 15-min intervals")
    axes[1].set_title("When: hour-of-day timing by regime")
    axes[1].legend(title="class", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    days = load_sample_days(SAMPLE_DAYS_CSV)
    shadow = load_shadow_prices(DATA_PATH)
    table = classify(build_table(days, shadow))

    corr = table["spp_spread"].corr(table["lmp_spread"])
    print(f"settled (SPP) vs SCED (LMP) spread correlation: {corr:.3f}")

    congested = table[table["spp_spread"] > DISP]
    print(f"\ncongested intervals (SPP spread > ${DISP:.0f}): "
          f"{len(congested)} of {len(table)}")
    print("\nhow often each regime, among congested intervals:")
    freq = congested["class"].value_counts(normalize=True).reindex(CLASSES).fillna(0)
    print(freq.round(3).to_string())

    print("\ncongested-interval class counts by day:")
    print(congested.groupby(["regime", "date", "class"]).size()
          .unstack("class").reindex(columns=CLASSES).fillna(0).astype(int).to_string())

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "settlement_regime_when.png"
    plot_when(congested, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
