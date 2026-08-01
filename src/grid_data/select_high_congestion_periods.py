"""
Select the times and days when congestion is extreme, for Phase 2 drill-down.

The base-case hourly max shadow price (np6-86) clears THRESHOLD only ~1% of
hours in 2025; those tail events are the moments we want richer per-bus LMP,
load, and generation data for. This script flags the hours above THRESHOLD,
summarizes them by day, and writes the lists that decide which delivery dates to
download. It runs the same selection for base-case and contingency constraints
and reports how much their high-congestion days overlap.

Outputs (CSV, in ercot_data/):
    high_congestion_hours_{base,contingency}.csv  one row per flagged hour
    high_congestion_days_{base,contingency}.csv   one row per flagged day
    sample_days.csv                               days to download for Phase 2,
                                                  labeled by regime

Usage:
    uv run python src/grid_data/select_high_congestion_periods.py
"""

from pathlib import Path

import pandas as pd

from plot_shadow_price_distributions import (
    DATA_PATH,
    hourly_max_constraint,
    load_shadow_prices,
)

THRESHOLD = 300.0  # $/MWh; base-case hourly max clears this ~1% of hours
N_PER_REGIME = 4  # sample days drawn from each regime for the base-vs-contingency test
OUT_DIR = Path(__file__).parent / "ercot_data"


def select_subset(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Split rows into base-case vs contingency constraints."""
    is_base = df["ContingencyName"].str.strip() == "BASE CASE"
    return df[is_base] if kind == "base" else df[~is_base]


def high_congestion_hours(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Hours whose max shadow price clears `threshold`, sorted by severity.

    Columns: hour, date, hour_of_day, max_price, max_constraint.
    """
    hmax = hourly_max_constraint(df)
    hot = hmax[hmax["max_price"] > threshold].reset_index()
    hot["date"] = hot["hour"].dt.date
    hot["hour_of_day"] = hot["hour"].dt.hour
    return hot.sort_values("max_price", ascending=False)


def summarize_by_day(hours: pd.DataFrame) -> pd.DataFrame:
    """One row per flagged day: peak price, when it peaked, and what bound.

    Columns: peak_price, peak_hour, n_event_hours, constraints. Sorted by
    peak_price so the most extreme days (best drill-down candidates) lead.
    """
    grouped = hours.groupby("date")
    counts = grouped.size().rename("n_event_hours")
    constraints = grouped["max_constraint"].agg(
        lambda s: ", ".join(sorted(s.unique()))
    ).rename("constraints")
    peak = (
        hours.drop_duplicates("date")  # `hours` is pre-sorted by max_price desc
        .set_index("date")[["max_price", "hour_of_day"]]
        .rename(columns={"max_price": "peak_price", "hour_of_day": "peak_hour"})
    )
    summary = peak.join([counts, constraints])
    return summary.sort_values("peak_price", ascending=False)


def overlap_report(base_hours: pd.DataFrame, cont_hours: pd.DataFrame) -> None:
    """Print hour-level co-occurrence of base-case vs contingency extremes.

    Day-level overlap is uninformative (contingencies fire nearly every day);
    the question that matters for drill-down is whether the same *hour* is
    extreme under both regimes, or whether base-case extremes stand alone.
    """
    base, cont = set(base_hours["hour"]), set(cont_hours["hour"])
    both = base & cont
    print(f"Base-case high-congestion hours:    {len(base)}")
    print(f"Contingency high-congestion hours:  {len(cont)}")
    print(f"Same hour extreme under both:       {len(both)} "
          f"({len(both) / len(base):.0%} of base hours)")
    print(f"Base-only hours (contingency quiet that hour): {len(base - both)}")


def top_days_by_peak(hours: pd.DataFrame, n: int, regime: str) -> pd.DataFrame:
    """Top `n` days within `hours`, one row per day at its peak hour.

    Columns: regime, peak_price, peak_hour, peak_constraint; indexed by date.
    """
    peak = (
        hours.sort_values("max_price", ascending=False)
        .drop_duplicates("date")
        .head(n)
        .set_index("date")
    )
    out = peak[["max_price", "hour_of_day", "max_constraint"]].rename(
        columns={
            "max_price": "peak_price",
            "hour_of_day": "peak_hour",
            "max_constraint": "peak_constraint",
        }
    )
    out.insert(0, "regime", regime)
    return out


def suggest_sample_days(base_hours: pd.DataFrame, cont_hours: pd.DataFrame,
                        n: int = N_PER_REGIME) -> pd.DataFrame:
    """Delivery dates that isolate base-case vs contingency congestion.

    To test which regime drives settled prices, pick days where one regime is
    extreme and the other is quiet, plus a few mixed days as controls. Each
    flagged hour is labeled at the hour level:
        base         base-case extreme, no contingency extreme that hour
        mixed        base-case and contingency extreme the same hour
        contingency  contingency extreme on a day with no base-case extreme
    Returns up to 3*n rows (top n per regime by peak price), indexed by date.
    """
    cont_hour_set = set(cont_hours["hour"])
    base_only = base_hours[~base_hours["hour"].isin(cont_hour_set)]
    both = base_hours[base_hours["hour"].isin(cont_hour_set)]
    base_dates = set(base_hours["date"])
    cont_only = cont_hours[~cont_hours["date"].isin(base_dates)]
    mixed = both[~both["date"].isin(set(base_only["date"]))]
    return pd.concat([
        top_days_by_peak(base_only, n, "base"),
        top_days_by_peak(mixed, n, "mixed"),
        top_days_by_peak(cont_only, n, "contingency"),
    ])


def save_table(table: pd.DataFrame, name: str) -> None:
    """Write a selection table to OUT_DIR as CSV."""
    path = OUT_DIR / f"{name}.csv"
    table.to_csv(path)
    print(f"  saved {len(table)} rows -> {path}")


def main() -> None:
    df = load_shadow_prices(DATA_PATH)
    hours_by_kind = {}
    for kind in ("base", "contingency"):
        hours = high_congestion_hours(select_subset(df, kind), THRESHOLD)
        days = summarize_by_day(hours)
        save_table(hours.set_index("hour"), f"high_congestion_hours_{kind}")
        save_table(days, f"high_congestion_days_{kind}")
        print(f"[{kind}] {len(hours)} hours over ${THRESHOLD:.0f} "
              f"on {len(days)} days; top days:")
        print(days.head(15).to_string(), "\n")
        hours_by_kind[kind] = hours
    overlap_report(hours_by_kind["base"], hours_by_kind["contingency"])

    sample = suggest_sample_days(hours_by_kind["base"], hours_by_kind["contingency"])
    save_table(sample, "sample_days")
    print("\nSuggested sample days for the base-vs-contingency test:")
    print(sample.to_string())


if __name__ == "__main__":
    main()
