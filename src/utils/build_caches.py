"""Joint builder for dual_cache.parquet and sensitivity_cache.parquet.

For each cached OPF solution under *case_dir* this opens the gzipped JSON
exactly once, walks the parsed dict to produce the dual-cache row, and --
when ``with_sensitivity`` is set -- wraps that same dict as a ModelData for
the barrier-regularised KKT sensitivity solve.

The dual extraction is cheap; the sensitivity solve is ~2x its cost and
nothing in the tail harness reads the sensitivity cache, so it is optional.
Build only the dual cache (``with_sensitivity=False``) when that is all the
tail comparison needs.

Results stream to disk in row batches: every ``batch_rows`` configs are
flushed to a numbered shard under ``_parts`` so the full run never holds the
whole cache in memory.  Each shard is written atomically, and shards are
flushed strictly in input order, so a run that is interrupted resumes by
skipping the already-persisted prefix (re-run with the same ``batch_rows``
and ``with_sensitivity``).  A clean finish consolidates the shards into the
two single-file caches and removes ``_parts``.

Output paths (under *out_dir*, default *case_dir*):
    dual_cache.parquet         columns: 2-level (group, name)
        groups: lmp, va, mu, nu, flow, p_gen, load, mu_pmax, mu_pmin
        - mu_pmax / mu_pmin are keyed by generator name and come from the
          KKT stationarity identity  mu_pmax - mu_pmin = LMP - dC/dPg
          with complementary slackness against the pg bounds.
    sensitivity_cache.parquet  columns: 2-level (branch, bus)
        values: d(mu_l) / d(load_j) at the OPF solution of each row

Both caches are written in the same row order, so positional row-axis
stacking downstream is safe.

Usage:
    uv run python src/utils/build_caches.py
"""

from __future__ import annotations

import functools
import gzip
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from egret.data.model_data import ModelData

from sensitivity_analysis import compute_sensitivity_barrier, _natkey
from utils.solution_io import _row_id_from_stem, _solution_files

EPS_MU, EPS_X = 1.0, 1.0
REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DIR = REPO_ROOT / "results" / "case30_gen_per_bus"
BATCH_ROWS = 5000
MAP_CHUNK = 256


def _marginal_cost(p_cost: dict, pg: float) -> float:
    """Evaluate dC/dPg at pg for an EGRET p_cost dict.

    Handles piecewise-linear curves (slope of the active segment) and
    polynomial curves (analytic derivative).  JSON deserialization turns
    polynomial coefficient keys into strings, so cast before using them.
    """
    curve_type = p_cost["cost_curve_type"]
    values = p_cost["values"]
    if curve_type == "piecewise":
        for (p0, c0), (p1, c1) in zip(values, values[1:]):
            if pg <= p1:
                return (c1 - c0) / (p1 - p0)
        (p0, c0), (p1, c1) = values[-2], values[-1]
        return (c1 - c0) / (p1 - p0)
    if curve_type == "polynomial":
        return sum(
            int(k) * float(a) * pg ** (int(k) - 1)
            for k, a in values.items()
            if int(k) >= 1
        )
    raise ValueError(f"Unsupported cost_curve_type: {curve_type}")


def _generator_bound_duals(
    raw: dict,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute mu_pmax and mu_pmin per generator from KKT stationarity.

    For a generator with cost C(pg), KKT gives mu_pmax - mu_pmin = LMP - dC/dPg,
    and complementary slackness makes only one side active depending on which
    bound (if any) pg is on.  Interior pg leaves both at zero.
    """
    buses = raw["elements"]["bus"]
    mu_pmax: dict[str, float] = {}
    mu_pmin: dict[str, float] = {}
    for g_name, g in raw["elements"]["generator"].items():
        pg = float(g["pg"])
        p_max = float(g["p_max"])
        p_min = float(g["p_min"])
        lmp = float(buses[g["bus"]]["lmp"])
        slack = lmp - _marginal_cost(g["p_cost"], pg)
        atol = 1e-4 * max(1.0, abs(p_max))
        at_max = abs(pg - p_max) < atol
        at_min = abs(pg - p_min) < atol
        mu_pmax[g_name] = max(0.0,  slack) if at_max else 0.0
        mu_pmin[g_name] = max(0.0, -slack) if at_min else 0.0
    return mu_pmax, mu_pmin


def _extract_blocks(
    raw: dict, row_id,
) -> tuple[dict, dict[str, float], dict[str, tuple[float, float]]]:
    """Walk the parsed solution dict once, producing everything both sides need.

    Returns
    -------
    dual_block
        Dict keyed by ("group", name) plus "row_id"; becomes one row of
        dual_cache.parquet.  Groups: lmp, va, mu, nu, flow, p_gen, load.
    lmp_base
        ``{bus: lmp}`` input for ``compute_sensitivity_barrier``.
    pf_dual_dict
        ``{branch: (mu_upper, mu_lower)}`` input for ``compute_sensitivity_barrier``.
    """
    dual_block = {"row_id": row_id}
    lmp_base: dict[str, float] = {}
    pf_dual_dict: dict[str, tuple[float, float]] = {}

    for name, bus in raw["elements"]["bus"].items():
        if "lmp" in bus:
            lmp = float(bus["lmp"])
            dual_block[("lmp", name)] = lmp
            lmp_base[name] = lmp
        if "va" in bus:
            dual_block[("va", name)] = float(bus["va"])
        dual_block[("p_gen", name)] = 0.0
        dual_block[("load", name)] = 0.0

    for name, br in raw["elements"]["branch"].items():
        mu_u, mu_l = br.get("mu_upper"), br.get("mu_lower")
        pf_dual_dict[name] = (float(mu_u or 0.0), float(mu_l or 0.0))
        if mu_u is not None and mu_l is not None:
            dual_block[("mu", name)] = float(mu_u) + float(mu_l)
        if "nu" in br:
            dual_block[("nu", name)] = float(br["nu"])
        if "pf" in br:
            dual_block[("flow", name)] = float(br["pf"])

    for g in raw["elements"]["generator"].values():
        dual_block[("p_gen", g["bus"])] += float(g.get("pg", 0.0))
    for ld in raw["elements"]["load"].values():
        dual_block[("load", ld["bus"])] += float(ld["p_load"])

    mu_pmax, mu_pmin = _generator_bound_duals(raw)
    for g_name, val in mu_pmax.items():
        dual_block[("mu_pmax", g_name)] = val
    for g_name, val in mu_pmin.items():
        dual_block[("mu_pmin", g_name)] = val

    return dual_block, lmp_base, pf_dual_dict


def _worker(path: Path, with_sensitivity: bool):
    """Per-file work: one disk read, one JSON parse, optional KKT solve.

    Returns ``(dual_block, row_id, S)`` where S is the flattened branch×bus
    sensitivity row, or None when sensitivity is disabled.  The branch/bus
    labels are fixed by the topology and built once by the driver, so they
    are not shipped back per file.
    """
    row_id = _row_id_from_stem(path.stem.replace(".json", ""))
    with gzip.open(path, "rt") as f:
        raw = json.load(f)

    dual_block, lmp_base, pf_dual_dict = _extract_blocks(raw, row_id)
    if not with_sensitivity:
        return dual_block, row_id, None

    md = ModelData(source=raw)
    _, mu_sens = compute_sensitivity_barrier(
        md, md, lmp_base, pf_dual_dict, eps_mu=EPS_MU, eps_x=EPS_X,
    )
    bu, bl = mu_sens["branch_upper"], mu_sens["branch_lower"]
    S = (bu.values + bl.values).reshape(-1)
    return dual_block, row_id, S


def _dual_columns(md: ModelData) -> pd.MultiIndex:
    """Full fixed dual-cache column set derived from the base-case topology.

    Fixing the schema up front lets every shard share it, so missing
    generator bound-dual columns (unavailable gens) fill with NaN and the
    shards stream into one file without a column-union step.
    """
    els = md.data["elements"]
    cols: list[tuple[str, str]] = []
    for b in els["bus"]:
        cols += [("lmp", b), ("va", b), ("p_gen", b), ("load", b)]
    for br in els["branch"]:
        cols += [("mu", br), ("nu", br), ("flow", br)]
    for g in els["generator"]:
        cols += [("mu_pmax", g), ("mu_pmin", g)]
    return pd.MultiIndex.from_tuples(sorted(cols))


def _sens_columns(md: ModelData) -> pd.MultiIndex:
    """branch×bus sensitivity columns in the natsorted order S is flattened in."""
    els = md.data["elements"]
    branches = sorted(els["branch"], key=_natkey)
    buses = sorted(els["bus"], key=_natkey)
    return pd.MultiIndex.from_product([branches, buses], names=["branch", "bus"])


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet via a temp file + rename so shards are never partial.

    ``index=True`` forces the row index to a physical column; otherwise
    pyarrow collapses a contiguous integer index to range-only metadata,
    leaving shard schemas inconsistent and breaking the streamed consolidate.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=True)
    os.replace(tmp, path)


def _flush(blocks, sens_rows, k, parts_dir, dual_cols, sens_cols, index_name):
    """Write batch *k*'s dual (and sensitivity) shard, reindexed to fixed schema."""
    dual = pd.DataFrame(blocks).set_index("row_id")
    dual.index.name = index_name
    dual.columns = pd.MultiIndex.from_tuples(dual.columns)
    dual = dual.reindex(columns=dual_cols)
    _write_atomic(dual, parts_dir / f"dual-{k:06d}.parquet")

    if sens_rows is not None:
        ids = [r for r, _ in sens_rows]
        mat = np.vstack([s for _, s in sens_rows])
        sens = pd.DataFrame(mat, index=pd.Index(ids, name=index_name), columns=sens_cols)
        _write_atomic(sens, parts_dir / f"sens-{k:06d}.parquet")


def _n_done_rows(parts_dir: Path) -> int:
    """Rows already persisted = sum of completed dual-shard row counts."""
    return sum(
        pq.read_metadata(p).num_rows for p in parts_dir.glob("dual-*.parquet")
    )


def _consolidate(parts_dir: Path, prefix: str, out_path: Path) -> int:
    """Stream shards (numeric order) into one parquet; return total row count."""
    writer, total = None, 0
    for s in sorted(parts_dir.glob(f"{prefix}-*.parquet")):
        t = pq.read_table(s)
        if writer is None:
            writer = pq.ParquetWriter(out_path, t.schema)
        writer.write_table(t)
        total += t.num_rows
    if writer is not None:
        writer.close()
    return total


def build_both_caches(
    case_dir: Path = DEFAULT_DIR,
    case_path: Path | None = None,
    scan: str = "availability",
    with_sensitivity: bool = True,
    max_workers: int | None = None,
    out_dir: Path | None = None,
    batch_rows: int = BATCH_ROWS,
) -> tuple[Path, Path | None]:
    """Build the dual cache (and optionally the sensitivity cache) from *case_dir*.

    Streams results to numbered shards under ``out_dir/_parts`` and resumes an
    interrupted run by skipping the already-persisted prefix.  *case_path* is
    the base-case JSON used to fix the column schemas; it defaults to
    ``data/<case_dir name>.json``.
    """
    case_path = case_path or REPO_ROOT / "data" / f"{case_dir.name}.json"
    md = ModelData.read(str(case_path))
    dual_cols = _dual_columns(md)
    sens_cols = _sens_columns(md) if with_sensitivity else None

    out_dir = out_dir or case_dir
    parts_dir = out_dir / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    index_name = "load_scale" if scan == "load" else "avail_id"

    paths = sorted(
        _solution_files(case_dir, scan),
        key=lambda p: _row_id_from_stem(p.stem.replace(".json", "")),
    )
    n_done = _n_done_rows(parts_dir)
    if n_done % batch_rows:
        raise RuntimeError(
            f"{n_done} persisted rows is not a multiple of batch_rows={batch_rows}; "
            "resume requires the same batch_rows as the interrupted run"
        )
    todo, k = paths[n_done:], n_done // batch_rows
    workers = max_workers or os.cpu_count() or 1
    print(
        f"{len(paths)} files in {case_dir}; {n_done} done, {len(todo)} to do; "
        f"{workers} workers; sensitivity={with_sensitivity}"
    )

    blocks: list[dict] = []
    sens_rows: list | None = [] if with_sensitivity else None
    t0 = time.perf_counter()
    fn = functools.partial(_worker, with_sensitivity=with_sensitivity)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for dual_block, row_id, S in pool.map(fn, todo, chunksize=MAP_CHUNK):
            blocks.append(dual_block)
            if sens_rows is not None:
                sens_rows.append((row_id, S))
            if len(blocks) == batch_rows:
                _flush(blocks, sens_rows, k, parts_dir, dual_cols, sens_cols, index_name)
                done = (k + 1) * batch_rows
                dt = time.perf_counter() - t0
                rate = (done - n_done) / dt
                print(f"  {done}/{len(paths)}  {dt:.1f}s ({rate:.0f}/s)", flush=True)
                blocks, k = [], k + 1
                sens_rows = [] if with_sensitivity else None
    if blocks:
        _flush(blocks, sens_rows, k, parts_dir, dual_cols, sens_cols, index_name)

    dual_path = out_dir / "dual_cache.parquet"
    nd = _consolidate(parts_dir, "dual", dual_path)
    print(f"Wrote {dual_path}  rows={nd}")
    sens_path: Path | None = None
    if with_sensitivity:
        sens_path = out_dir / "sensitivity_cache.parquet"
        ns = _consolidate(parts_dir, "sens", sens_path)
        assert nd == ns, "dual and sens row counts disagree -- joint-build invariant violated"
        print(f"Wrote {sens_path}  rows={ns}")

    shutil.rmtree(parts_dir)
    print(f"Total {time.perf_counter() - t0:.1f}s")
    return dual_path, sens_path


if __name__ == "__main__":
    build_both_caches(with_sensitivity=False)
