"""Read and reduce the ERCOT binding-constraint shadow-price feed (np6-86).

The feed reports a row for every constraint SCED monitors each interval, not only the
binding ones: a non-binding interval has `ShadowPrice == 0` (flow below limit,
`ViolatedMW <= 0`), so the nonzero shadow prices are the binding events.

Identifiers are easy to get wrong.  `ConstraintName` is the physical monitored element --
a line, or an interface / aggregate transfer limit like `NE_LOB`, which has NaN stations.
`ConstraintID` is *not* an element key: it only ranges 1-59 and is a per-interval rank
slot.  A distinct constraint is the pair (`ConstraintName`, `ContingencyName`) -- an
element monitored under an assumed contingency -- and these reductions group by the
physical element, `ConstraintName`.

Every module in this package that touches shadow prices loads them from here; the
figures built on top live in `figures/ercot/shadow_price_distributions.py`.
"""

import numpy as np
import pandas as pd

from ercot.data_paths import SHADOW_PRICES  # noqa: F401  (re-exported as the default input)


def load_shadow_prices(path=SHADOW_PRICES) -> pd.DataFrame:
    """Read the shadow-price parquet and parse the SCED timestamp."""
    df = pd.read_parquet(path)
    df["SCEDTimestamp"] = pd.to_datetime(df["SCEDTimestamp"])
    return df


def binding_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-constraint stats over binding intervals (ShadowPrice > 0)."""
    binding = df[df["ShadowPrice"] > 0]
    grouped = binding.groupby("ConstraintName")["ShadowPrice"]
    summary = grouped.agg(
        binding_count="size", mean_price="mean", max_price="max", total_price="sum"
    )
    return summary.sort_values("binding_count", ascending=False)


def hourly_price_max(df: pd.DataFrame) -> pd.Series:
    """Per hour, the max over constraints of each constraint's hourly-mean price.

    A constraint can appear in several SCED intervals within an hour; average those to
    one value per constraint, then max across constraints.
    """
    hourly = df.assign(hour=df["SCEDTimestamp"].dt.floor("h"))
    per_constraint = hourly.groupby(["hour", "ConstraintName"])["ShadowPrice"].mean()
    return per_constraint.groupby(level="hour").max()


def hourly_max_constraint(df: pd.DataFrame) -> pd.DataFrame:
    """Per hour, the max shadow price over all constraints and which one set it.

    Same hourly averaging as hourly_price_max; returns columns max_price and
    max_constraint indexed by hour.
    """
    hourly = df.assign(hour=df["SCEDTimestamp"].dt.floor("h"))
    per_constraint = hourly.groupby(["hour", "ConstraintName"])["ShadowPrice"].mean()
    by_hour = per_constraint.groupby(level="hour")
    return pd.DataFrame({"max_price": by_hour.max(),
                         "max_constraint": by_hour.idxmax().map(lambda key: key[1])})


def dominant_frequency(hmax: pd.DataFrame) -> pd.Series:
    """Count how often each constraint is the hourly max (hours with max > 0)."""
    binding = hmax[hmax["max_price"] > 0]
    return binding["max_constraint"].value_counts()


def calendar_grid(series: pd.Series) -> np.ndarray:
    """Reshape an hour-indexed Series to a (24, n_days) grid: hour x day-of-year."""
    idx = series.index
    n_days = int(idx.dayofyear.max())
    grid = np.full((24, n_days), np.nan)
    grid[idx.hour, idx.dayofyear - 1] = series.values
    return grid
