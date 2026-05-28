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


def load_dual_cache(cache: Path, scan="availability") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached dual sums from Parquet, rebuilding if missing or stale.

    *scan* selects which experiment's files feed the cache: "availability"
    (vary availability, fixed parameters) or "load" (scan load, fixed
    availability).  See `_solution_files`.
    """
    stale = not cache.exists() or any(
        p.stat().st_mtime > cache.stat().st_mtime
        for p in _solution_files(cache.parent, scan)
    )
    if stale:
        build_dual_cache(cache, scan)
    df = pd.read_parquet(cache)
    df = df.sort_index()
    
    name = "load_scale" if scan == "load" else "avail_id"
    df.index.name = name
    return df


def load_all_duals(cache_path, scan="availability") -> dict[str, pd.DataFrame]:
    """Load duals and primal variables from every solution JSON, summed over hours.

    *scan* selects which experiment's files to read (see `_solution_files`):
    "availability" indexes rows by avail_id; "load" indexes them by the
    (avail_id, load_id, scale_id) tuple.

    Each file is read once; generation total, load total, and all dual and
    primal quantities are extracted from the same parsed dict. Works for both
    multi-period UC results (fields are dicts of hourly values) and
    single-period OPF results (fields are scalars) — in the OPF case "sum over
    hours" is just the single value.

    Returns a dict of DataFrames, one per quantity, each indexed by scenario:
        lmp       — columns = bus names (dual λ of nodal power balance)
        mu        — columns = branch names (|mu_upper| + |mu_lower| on flow limit)
        nu        — columns = branch names (dual of pf = B(θ_f − θ_t))
        injection — columns = bus names (nodal power injection, gen minus load)
        flow      — columns = branch names (line power flow pf)
        va        — columns = bus names (nodal phase angle)
    """
    blocks = {k: {} for k in ("lmp", "mu", "nu", "injection", "flow", "va")}
    skipped = 0
    for path in _solution_files(cache_path.parent, scan):
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
        try:
            with gzip.open(path, "rt") as f:
                raw = json.load(f)
        except:
            print(path)
            raise
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
        nu_sum = {}
        for name, br in raw["elements"]["branch"].items():
            mu_u = br.get("mu_upper")
            mu_l = br.get("mu_lower")
            if mu_u is None or mu_l is None:
                continue
            upper = np.array(_hourly(mu_u, n_hours))
            lower = np.array(_hourly(mu_l, n_hours))
            mu_sum[name] = float((upper + lower).sum()) # sum abs instead of positives/negatives
            nu = br.get("nu")
            if nu is not None:
                nu_sum[name] = float(np.sum(_hourly(nu, n_hours)))

        row_id = avail_id # default calculation, scan through availabilities.
        if "_load" in stem: # scaling load experiment
            row_id = (avail_id, load_id, scale_id)
        blocks["lmp"][row_id] = pd.Series(lmp_sum)
        blocks["mu"][row_id] = pd.Series(mu_sum)
        blocks["nu"][row_id] = pd.Series(nu_sum)
        blocks["injection"][row_id] = pd.Series(_nodal_injection(raw, n_hours))
        blocks["flow"][row_id] = pd.Series(_line_flow(raw, n_hours))
        blocks["va"][row_id] = pd.Series(_phase_angle(raw, n_hours))
    if skipped:
        print(f"Skipped {skipped} scenarios with load/generation mismatch")
    name = "load_scale" if scan == "load" else "avail_id"
    result = {k: pd.DataFrame(rows).T for k, rows in blocks.items()}
    for df in result.values():
        df.index.name = name
    return result


def build_dual_cache(cache_path: Path, scan="availability") -> Path:
    """Aggregate per-scenario dual and primal sums into a single Parquet file.

    Reads the *scan* experiment's files in *cache_path.parent* (see
    `_solution_files`), computes hourly sums of the duals and primal variables,
    and writes one column group per quantity into a single Parquet file:
      - lmp/<bus_name>        nodal LMP (dual λ of power balance)
      - mu/<branch_name>      flow-limit dual (|mu_upper| + |mu_lower|)
      - nu/<branch_name>      dual of the pf = B(θ_f − θ_t) equality
      - injection/<bus_name>  nodal power injection (generation minus load)
      - flow/<branch_name>    line power flow (pf)
      - va/<bus_name>         nodal phase angle
    Rows are indexed by avail_id ("availability" scan) or by the
    (avail_id, load_id, scale_id) tuple ("load" scan).

    Returns the path to the written Parquet file.
    """
    blocks = load_all_duals(cache_path, scan)
    combined = pd.concat(blocks, axis=1)
    combined.to_parquet(cache_path)
    print(f"Wrote {cache_path} ({len(combined)} scenarios)")
    return cache_path

def _solution_files(parent: Path, scan: str) -> list[Path]:
    """Return the solution files for the named scan experiment, sorted by name.

    Two experiments write files into *parent*:
      "availability" — vary generator availability at fixed parameters; the plain
                       per-availability files, with no load/scale suffix.
      "load"         — vary load scaling at fixed availability; the files written
                       with a ``_load<n>_scale<n>`` suffix.
    glob/fnmatch cannot negate a substring, so the "availability" set is filtered
    in Python to drop the load-scan files.
    """
    if scan == "load":
        return sorted(parent.glob("*_load*.json.gz"))
    if scan == "availability":
        return [p for p in sorted(parent.glob("*.json.gz"))
                if "_load" not in p.name and "_scale" not in p.name]
    raise ValueError(f"unknown scan {scan!r}; expected 'availability' or 'load'")


def _hourly(field, n_hours):
    """Expand a scalar-or-time-series field into a length-n_hours list."""
    return field["values"] if isinstance(field, dict) else [float(field)] * n_hours


def _nodal_injection(raw, n_hours) -> dict[str, float]:
    """Sum over hours of net injection (generation minus load) at each bus."""
    inj = {name: np.zeros(n_hours) for name in raw["elements"]["bus"]}
    for g in raw["elements"]["generator"].values():
        inj[g["bus"]] += np.array(_hourly(g.get("pg", 0.0), n_hours))
    for load in raw["elements"]["load"].values():
        inj[load["bus"]] -= np.array(_hourly(load["p_load"], n_hours))
    return {name: float(v.sum()) for name, v in inj.items()}


def _line_flow(raw, n_hours) -> dict[str, float]:
    """Sum over hours of the power flow (pf) on each branch."""
    return {
        name: float(np.sum(_hourly(br["pf"], n_hours)))
        for name, br in raw["elements"]["branch"].items()
        if "pf" in br
    }


def _phase_angle(raw, n_hours) -> dict[str, float]:
    """Sum over hours of the phase angle (va) at each bus."""
    return {
        name: float(np.sum(_hourly(b["va"], n_hours)))
        for name, b in raw["elements"]["bus"].items()
        if "va" in b
    }


