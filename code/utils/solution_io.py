"""Helpers for reading dispatch, metadata, and duals from solved single-period
OPF ModelData / JSON. All scalar fields are plain floats — no hourly expansion.

For multi-period unit-commitment results (with ``time_keys`` and ``{"values":
[...]}`` time-series fields), use ``multiperiod_solution_io`` instead.
"""

import json
import gzip
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
import time

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


def load_pg(raw: dict) -> pd.Series:
    """Return a Series of pg dispatch in MW, indexed by generator name."""
    return pd.Series({
        name: float(g.get("pg", 0.0))
        for name, g in raw["elements"]["generator"].items()
    })


def load_generator_metadata(raw: dict) -> pd.DataFrame:
    """Return DataFrame with 'fuel' and 'type' columns indexed by generator name."""
    return pd.DataFrame(
        {name: {"fuel": g.get("fuel", "?"), "type": g.get("generator_type", "?")}
         for name, g in raw["elements"]["generator"].items()}
    ).T


def pg_by_fuel(pg: pd.Series, meta: pd.DataFrame) -> dict[str, float]:
    """Aggregate pg by fuel type label, returning {label: total_MW}."""
    result = {}
    for fuel_code, label in FUEL_LABELS.items():
        gens = meta.index[meta["fuel"] == fuel_code].tolist()
        cols = [g for g in gens if g in pg.index]
        if cols:
            result[label] = float(pg[cols].sum())
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


def load_dual_cache(cache: Path, scan="availability") -> pd.DataFrame:
    """Load cached dual values from Parquet, rebuilding if missing or stale.

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


def load_all_duals(cache_path, scan="availability", max_workers=None) -> pd.DataFrame:
    """Load duals and primal variables from every single-period OPF solution.

    *scan* selects which experiment's files to read (see `_solution_files`):
    "availability" indexes rows by avail_id; "load" indexes them by the
    (avail_id, load_id, scale_id) tuple.

    Files are read in parallel via a ThreadPoolExecutor (gzip + JSON parsing
    are the bottleneck and gzip releases the GIL); *max_workers* is forwarded
    to the pool (default: ``min(32, os.cpu_count() + 4)``).

    Returns a DataFrame with a two-level column index, one block per quantity:
        lmp       — columns = bus names (dual λ of nodal power balance)
        mu        — columns = branch names (mu_upper + mu_lower on flow limit)
        nu        — columns = branch names (dual of pf = B(θ_f − θ_t))
        p_gen     — columns = bus names (nodal power gen)
        injection — columns = bus names (nodal power injection, gen minus load)
        flow      — columns = branch names (line power flow pf)
        va        — columns = bus names (nodal phase angle)
    """
    blocks = {k: {} for k in ("lmp", "mu", "nu", "p_gen", "injection", "flow", "va")}
    paths = _solution_files(cache_path.parent, scan)
    t0 = time.perf_counter()
    blocks = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # Workers write a (unique) row_id slot into the shared `blocks` dict
        # CPython dict insertion is atomic, and no key collides -> no GIL
        # `list(...)` forces exceptions from workers to surface here
        futures = [ex.submit(_load_one_solution, path) for path in paths]
        for i, ft in enumerate(as_completed(futures)):
            blocks.append(ft.result())
            if (i+1)%500 == 0:
                print(f"{i+1}/{len(paths)}  {time.perf_counter()-t0:.2f}", flush=True)
        #list(ex.map(f, paths))

    name = "load_scale" if scan == "load" else "avail_id"
    #result = {k: pd.DataFrame(rows).T for k, rows in blocks.items()}
    #df = pd.concat(result, axis=1)
    df = pd.DataFrame(blocks)
    df.set_index("row_id", inplace=True)
    df.index.name = name
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df.sort_index().sort_index(axis=1, level=0)


def _load_one_solution(path: Path) -> None:
    """Read one gzipped solution file and write its row into *blocks* in-place.

    Each worker writes to a unique ``blocks[<quantity>][row_id]`` slot — 
    so concurrent calls are safe under the GIL without explicit locking.
    """
    row_id = _row_id_from_stem(path.stem.replace(".json", ""))
    with gzip.open(path, "rt") as f:
        raw = json.load(f)

    block = dict(row_id=row_id)
           
    for bus_name, bus in raw["elements"]["bus"].items():
        if "lmp" in bus.keys():
            block[("lmp", bus_name)] = float(bus["lmp"])
        if "va" in bus.keys():
            block[("va", bus_name)] = float(bus["va"])

    for br_name, br in raw["elements"]["branch"].items():
        mu_u = br.get("mu_upper")
        mu_l = br.get("mu_lower")
        if not (mu_u is None or mu_l is None):
            block[("mu", br_name)] = float(mu_u) + float(mu_l)
        if "nu" in br:
            block[("nu", br_name)] = float(br["nu"])
        if "pf" in br:
            block[("flow", br_name)] = float(br["pf"])

    block.update(_nodal_gen(raw))
    block.update(_nodal_load(raw))
    return block


def _row_id_from_stem(stem: str):
    """Parse the scenario id (or (avail, load, scale) tuple) from a file stem."""
    if "_load" in stem:
        s = stem.replace("load_", "load")
        avail_id = load_id = scale_id = None
        for word in s.split("_"):
            if "avail" in word:
                avail_id = int(word.replace("avail", ""))
            if "load" in word:
                load_id = int(word.replace("load", ""))
            if "scale" in word:
                scale_id = int(word.replace("scale", ""))
        return (avail_id, load_id, scale_id)
    if "avail" in stem:
        return int(stem.split("avail")[1])
    return stem


def build_dual_cache(cache_path: Path, scan="availability") -> Path:
    """Aggregate per-scenario duals and primal variables into a single Parquet file.

    Reads the *scan* experiment's files in *cache_path.parent* (see
    `_solution_files`), extracts the duals and primal variables from each
    single-period OPF solution, and writes one column group per quantity into a
    single Parquet file:
      - lmp/<bus_name>        nodal LMP (dual λ of power balance)
      - mu/<branch_name>      flow-limit dual (mu_upper + mu_lower)
      - nu/<branch_name>      dual of the pf = B(θ_f − θ_t) equality
      - injection/<bus_name>  nodal power injection (generation minus load)
      - flow/<branch_name>    line power flow (pf)
      - va/<bus_name>         nodal phase angle
    Rows are indexed by avail_id ("availability" scan) or by the
    (avail_id, load_id, scale_id) tuple ("load" scan).

    Returns the path to the written Parquet file.
    """
    df = load_all_duals(cache_path, scan)
    df.to_parquet(cache_path)
    print(f"Wrote {cache_path} ({len(df)} scenarios)")
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


def _nodal_gen(raw) -> dict[str, float]:
    """Net injection (generation minus load) at each bus."""
    key = "p_gen"
    inj = {(key, name): 0.0 for name in raw["elements"]["bus"]}
    for g in raw["elements"]["generator"].values():
        inj[(key, g["bus"])] += float(g.get("pg", 0.0))
    return inj


def _nodal_load(raw) -> dict[str, float]:
    key = "load"
    inj = {(key, name): 0.0 for name in raw["elements"]["bus"]}
    for load in raw["elements"]["load"].values():
        inj[(key, load["bus"])] -= float(load["p_load"])
    return inj


