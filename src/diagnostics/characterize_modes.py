"""Characterize active-set modes: prices, probability, hit count.

A mode is the set of binding constraints (dual > tol) at a solution; an active
set. Within a mode the LP duals are generically constant, so a mode carries a
single price vector. This groups the enumerated configs by mode and reports,
per mode, the hit count, empirical probability (uniform over the enumerated
configs), and the mean/spread of the dual "prices". A within-mode spread near 0
confirms the duals are pinned by the mode; a large spread means the binding-set
signature does not fully determine the price (e.g. a marginal generator that is
not itself at a bound still sets lambda).

Applied to two enumerations: case30 (single load, full active set incl. gen
bounds) and ACTIVSg2000 Texas (load-averaged mu, congestion constraints only).
"""
import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

from paths import REPO_ROOT
RESULTS = REPO_ROOT / "results"

CASE30 = RESULTS / "case30_gen_per_bus/dual_cache.parquet"
TEXAS_MU = RESULTS / "activsg2000/mu_cache"
TEXAS_LMP = RESULTS / "activsg2000/lmp_cache.parquet"
TOL = 1e-3


def signature_codes(binding):
    """Map each row's binding-constraint boolean pattern to an integer mode code."""
    packed = np.packbits(binding, axis=1)
    view = np.ascontiguousarray(packed).view(
        np.dtype((np.void, packed.shape[1] * packed.dtype.itemsize)))
    _, codes, counts = np.unique(view.ravel(), return_inverse=True, return_counts=True)
    return codes.astype(np.int64), len(counts)


def summarize_modes(codes, prices):
    """Per-mode table: hits, probability, and mean/std of each price array."""
    df = pd.DataFrame({"code": codes, **prices})
    g = df.groupby("code")
    out = pd.DataFrame({"n_hits": g.size()})
    out["prob"] = out["n_hits"] / out["n_hits"].sum()
    for key in prices:
        out[f"{key}_mean"] = g[key].mean()
        out[f"{key}_std"] = g[key].std().fillna(0.0)
    return out.sort_values("prob", ascending=False)


def report(modes, name, price_key, prices):
    """Print headline mode statistics and the heaviest-price modes."""
    n = len(modes)
    cover95 = int((modes["prob"].cumsum() < 0.95).sum()) + 1
    print(f"\n===== {name} =====")
    print(f"distinct modes: {n}   modes covering 95% mass: {cover95}")
    for key in prices:
        w = np.average(modes[f"{key}_std"], weights=modes["n_hits"])
        print(f"  within-mode {key} std (0 => mode pins it): "
              f"hit-weighted mean {w:.4g}, max {modes[f'{key}_std'].max():.4g}")
    print(f"\ntop 8 modes by probability:")
    _print_rows(modes.head(8), price_key)
    print(f"\ntop 8 modes by {price_key} (rarity check):")
    _print_rows(modes.sort_values(f"{price_key}_mean", ascending=False).head(8), price_key)


def _print_rows(rows, price_key):
    print(f"  {'n_hits':>9} {'prob':>10} {price_key + '_mean':>14} {price_key + '_std':>12}")
    for _, r in rows.iterrows():
        print(f"  {int(r.n_hits):>9} {r.prob:>10.3e} "
              f"{r[price_key + '_mean']:>14.4g} {r[price_key + '_std']:>12.4g}")


def load_case30(sig_kinds):
    """Mode signature from the given dual kinds; prices = max mu and LMP range.

    sig_kinds selects the granularity: ("mu", "nu") = congestion (line) only;
    add ("mu_pmax", "mu_pmin") = full active set incl. generator bounds.
    """
    t = pq.read_table(CASE30)
    kinds = {n: n.split("'")[1] for n in t.column_names if n.startswith("(")}
    cons = [n for n, k in kinds.items() if k in sig_kinds]
    mus = [n for n, k in kinds.items() if k in ("mu", "nu")]
    lmps = [n for n, k in kinds.items() if k == "lmp"]
    M = np.column_stack([t.column(n).to_numpy() for n in cons])
    codes, _ = signature_codes(M > TOL)
    max_mu = np.column_stack([t.column(n).to_numpy() for n in mus]).max(axis=1)
    lmp = np.column_stack([t.column(n).to_numpy() for n in lmps])
    return codes, {"max_mu": max_mu, "lmp_range": lmp.max(axis=1) - lmp.min(axis=1)}


def load_texas():
    """Congestion mode = set of load-averaged binding constraints per config; join LMP."""
    parts = sorted(glob.glob(f"{TEXAS_MU}/part_*.parquet"))
    mu = pd.concat([pq.read_table(p, columns=["id", "constraint", "mu"]).to_pandas()
                    for p in parts], ignore_index=True)
    mu = mu[mu.mu > TOL]
    ccode = pd.factorize(mu.constraint)[0]
    mu = mu.assign(ccode=ccode).sort_values(["id", "ccode"])
    sig = mu.groupby("id")["ccode"].agg(lambda c: c.to_numpy().tobytes())
    codes = pd.factorize(sig)[0].astype(np.int64)
    lmp = pq.read_table(TEXAS_LMP, columns=["S", "lmp_range", "lmp_maxabs"]).to_pandas()
    lmp = lmp.loc[sig.index]
    return codes, {"S": lmp.S.to_numpy(), "lmp_range": lmp.lmp_range.to_numpy(),
                   "lmp_maxabs": lmp.lmp_maxabs.to_numpy()}


def main():
    codes, prices = load_case30(("mu", "nu"))
    report(summarize_modes(codes, prices), "case30 (congestion / line only)", "max_mu", prices)

    codes, prices = load_case30(("mu", "nu", "mu_pmax", "mu_pmin"))
    report(summarize_modes(codes, prices), "case30 (full active set)", "max_mu", prices)

    codes, prices = load_texas()
    report(summarize_modes(codes, prices), "ACTIVSg2000 Texas (congestion)", "S", prices)


if __name__ == "__main__":
    main()
