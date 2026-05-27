"""Helpers for reading dispatch, metadata, and duals from solved ModelData / JSON."""

import json
import gzip
import pandas as pd
import numpy as np
from pathlib import Path

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


def mu_sensitivity_df(result: dict) -> pd.DataFrame:
    """Stack all mu sensitivity blocks into one DataFrame.

    Rows are a (constraint_type, name) MultiIndex; columns are the injection
    bus names.  Empty blocks are skipped.
    """
    frames = []
    for ctype, entry in result.get("mu_sensitivity", {}).items():
        if not entry["index"]:
            continue
        df = pd.DataFrame(entry["values"], index=entry["index"], columns=entry["columns"])
        df.index = pd.MultiIndex.from_product([[ctype], df.index], names=["type", "name"])
        frames.append(df)
    return pd.concat(frames) if frames else pd.DataFrame()


def load_sensitivity_result(id: int, path: Path) -> dict:
    """Load the sensitivity JSON file with the given availability id from *path*.

    Matches files of the form ``*avail<id>.json`` in the directory *path*.
    """
    path = Path(path)
    matches = list(path.glob(f"*avail{id}.json"))
    assert len(matches) == 1, \
        f"Expected exactly 1 match for avail{id} in {path}, found {len(matches)}: {matches}"
    return json.loads(matches[0].read_text())


def load_dual_cache(cache: Path, search="*") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached dual sums from Parquet, rebuilding if missing or stale."""
    #cache = result_path / "dual_cache.parquet"
    stale = not cache.exists() or any(
        p.stat().st_mtime > cache.stat().st_mtime
        for p in cache.parent.glob(f"{search}.json.gz")
    )
    if stale:
        build_dual_cache(cache, search)
    df = pd.read_parquet(cache)
    lmp_all = df["lmp"]
    mu_all = df["mu"]
    lmp_all.index.name = "avail_id"
    mu_all.index.name = "avail_id"
    return lmp_all, mu_all


def load_all_duals(cache_path, search="*") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load LMP and mu from every solution JSON, summed over hours.

    Each file is read once; generation total, load total, LMP, and μ are
    all extracted from the same parsed dict. Works for both multi-period UC
    results (fields are dicts of hourly values) and single-period OPF
    results (fields are scalars) — in the OPF case "sum over hours" is
    just the single value.

    Returns:
        lmp_all — DataFrame with columns = bus names, one row per scenario
        mu_all  — DataFrame with columns = branch names, one row per scenario
    """
    lmp_rows = {}
    mu_rows = {}
    skipped = 0
    for path in sorted(cache_path.parent.glob(f"{search}.json.gz")):
        stem = path.stem.replace(".json", "")
        load_id = None
        if "_load" in stem:
            s = stem.replace("load_", "load")
            for word in s.split("_"):
                if "avail" in word:
                    avail_id = int(word.replace("avail", ""))
                if "load" in word:
                    load_id = int(word.replace("load", ""))
                if "scale" in word:
                    scale_id = int(word.replace("scale", ""))
        elif "avail" in stem:
            avail_id = int(stem.split("avail")[1])
        else:
            avail_id = stem
        with gzip.open(path, "rt") as f:
            raw = json.load(f)
        time_keys = raw["system"].get("time_keys")
        n_hours = len(time_keys) if time_keys else 1
        if isinstance(avail_id, int) and time_keys and sum(map(int, np.binary_repr(avail_id))) < 3:
            continue

        total_gen = np.zeros(n_hours)
        for g in raw["elements"]["generator"].values():
            total_gen += np.array(_hourly(g.get("pg", 0.0), n_hours))

        total_load = np.zeros(n_hours)
        for load in raw["elements"]["load"].values():
            total_load += np.array(_hourly(load["p_load"], n_hours))

        #if not np.allclose(total_gen, total_load, rtol=1e-6, atol=1e-6):
        #    skipped += 1
        #    print(f"Skipping {avail_id}")
        #    continue

        lmp_sum = {}
        for name, b in raw["elements"]["bus"].items():
            lmp = b.get("lmp")
            if lmp is None:
                continue
            lmp_sum[name] = float(np.sum(_hourly(lmp, n_hours)))

        mu_sum = {}
        for name, br in raw["elements"]["branch"].items():
            mu_u = br.get("mu_upper")
            mu_l = br.get("mu_lower")
            if mu_u is None or mu_l is None:
                continue
            upper = np.array(_hourly(mu_u, n_hours))
            lower = np.array(_hourly(mu_l, n_hours))
            mu_sum[name] = float((upper + lower).sum()) # sum abs instead of positives/negatives

        row_id = avail_id # default calculation, scan through availabilities.
        if "_load" in stem: # scaling load experiment
            row_id = (avail_id, load_id, scale_id)
        lmp_rows[row_id] = pd.Series(lmp_sum)
        mu_rows[row_id] = pd.Series(mu_sum)
    if skipped:
        print(f"Skipped {skipped} scenarios with load/generation mismatch")
    lmp_all = pd.DataFrame(lmp_rows).T
    mu_all = pd.DataFrame(mu_rows).T
    name = "load_scale" if load_id else "avail_id"
    lmp_all.index.name = name
    mu_all.index.name = name
    return lmp_all, mu_all


def build_dual_cache(cache_path: Path, search) -> Path:
    """Aggregate per-scenario dual sums into a single Parquet file.

    Reads every uc_*.json.gz in *cache_path.parent*, computes hourly sums of LMP
    and mu, and writes two column groups into one Parquet file:
      - lmp/<bus_name>  for each bus
      - mu/<branch_name> for each branch
    Rows are indexed by avail_id.  Scenarios that fail the gen/load balance
    check are excluded.

    Returns the path to the written Parquet file.
    """
    lmp_all, mu_all = load_all_duals(cache_path, search)
    combined = pd.concat(
        {"lmp": lmp_all, "mu": mu_all}, axis=1,
    )
    combined.to_parquet(cache_path)
    print(f"Wrote {cache_path} ({len(combined)} scenarios)")
    return cache_path

def _hourly(field, n_hours):
    """Expand a scalar-or-time-series field into a length-n_hours list."""
    return field["values"] if isinstance(field, dict) else [float(field)] * n_hours


