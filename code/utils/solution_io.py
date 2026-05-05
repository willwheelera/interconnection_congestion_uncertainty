"""Helpers for reading dispatch, metadata, and duals from solved ModelData / JSON."""

import pandas as pd

FUEL_LABELS = {
    "C": "Coal",
    "G": "Gas CC",
    "N": "Nuclear",
    "O": "Oil / CT",
    "H": "Hydro",
    "S": "Solar",
    "W": "Wind",
}


def get_lmps(md_sol) -> dict[str, float]:
    """Extract {bus_name: lmp} from a solved ModelData."""
    return {
        name: bus_data["lmp"]
        for name, bus_data in md_sol.elements(element_type="bus")
        if "lmp" in bus_data
    }


def load_pg(raw: dict) -> pd.DataFrame:
    """Return (n_hours x n_generators) DataFrame of pg dispatch in MW."""
    gens = raw["elements"]["generator"]
    n_hours = len(raw["system"]["time_keys"])
    data = {}
    for name, g in gens.items():
        pg = g.get("pg", 0.0)
        data[name] = pg["values"] if isinstance(pg, dict) else [float(pg)] * n_hours
    return pd.DataFrame(data, index=range(n_hours))


def load_generator_metadata(raw: dict) -> pd.DataFrame:
    """Return DataFrame with 'fuel' and 'type' columns indexed by generator name."""
    gens = raw["elements"]["generator"]
    return pd.DataFrame(
        {name: {"fuel": g.get("fuel", "?"), "type": g.get("generator_type", "?")}
         for name, g in gens.items()}
    ).T


def pg_by_fuel(pg_df: pd.DataFrame, meta: pd.DataFrame) -> dict[str, pd.Series]:
    """Aggregate pg by fuel type label, returning {label: hourly_total_MW}."""
    result = {}
    for fuel_code, label in FUEL_LABELS.items():
        gens = meta.index[meta["fuel"] == fuel_code].tolist()
        cols = [g for g in gens if g in pg_df.columns]
        if cols:
            result[label] = pg_df[cols].sum(axis=1)
    return result
