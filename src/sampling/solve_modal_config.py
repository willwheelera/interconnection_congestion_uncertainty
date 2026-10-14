"""Solve the single most-probable Bernoulli configuration for its per-line |mu|.

The Bernoulli nominal law has no coupling (J = 0), so its most-probable configuration is the
per-generator marginal mode: generator i on iff its field h_i > 0 (P(on) = sigmoid(h_i) > 1/2).
The MCMC sweep never draws this one fixed ngen-bit config -- its probability is the product of
all the marginal modes, effectively never hit -- so its per-line |mu| is absent from
dual_samples.parquet.  This solves that one config and writes its per-line |mu| as a one-row
wide table: the single-point companion to the sampled draws, in the same ("mu", constraint)
format and snapshot-averaged the same way.

By default it solves from scratch against the full N-1 set, with every contingency enforced
lazily, so no screened file is needed and its mu equals the screened samples' exactly (the
screened and unscreened solves converge to the same duals).  `--screened` instead reuses the
law's already-grown screened set, a faster build when that file is present.

Usage:
    uv run python -m src.sampling.solve_modal_config --case activsg2000_v12_levfull --law bernoulli
    uv run python -m src.sampling.solve_modal_config --case activsg2000_v12_levfull \
        --law bernoulli --screened
"""

import argparse
import functools

import numpy as np
import pandas as pd

from src.cases import registry
from src.model import evaluator
from src.sampling.distributions import case_dist
from src.sampling.mcmc_sweep import make_rows, per_law_case
from src.sampling.problem import write_parquet_meta
from src.sweeps.enumerate_configs import solve_chunk


def modal_config(mrf):
    """The most-probable Bernoulli config as a (1, ngen) row: generator on iff h_i > 0 (J = 0)."""
    return (mrf.h > 0).astype(np.int8).reshape(1, -1)


def solve_one(case, law, screened):
    """Per-constraint mu of the modal config, solved through the sweep's screened evaluator."""
    mrf = case_dist(case, law)
    X = modal_config(mrf)
    with evaluator.for_case(case, case.relief_buses(), screened=screened) as target:
        duals, _snaps, _cfgs = solve_chunk(case, target, X, functools.partial(make_rows, 0),
                                           None, 0)
    return duals, int(X.sum())


def wide_point(duals):
    """The modal config's binding |mu| as a one-row ("mu", constraint) frame (drops the sentinel)."""
    mu = {con: abs(m) for _r, _c, _n, con, m in duals if con != "none"}
    df = pd.DataFrame([list(mu.values())], columns=pd.MultiIndex.from_product([["mu"], list(mu)]))
    df.index.name = "id"
    return df


def run(case, law, screened):
    """Solve the modal Bernoulli config and write results/<case>/mcmc/<law>/modal_point.parquet."""
    out_dir = case.results_dir / "mcmc" / law
    if screened:
        case = per_law_case(case, out_dir)
    duals, n_on = solve_one(case, law, screened)
    df = wide_point(duals)
    meta = dict(case=case.name, law=law, n_on=n_on, config="bernoulli_mode", screened=screened)
    dest = out_dir / "modal_point.parquet"
    write_parquet_meta(df, meta, dest)
    top = df["mu"].iloc[0].sort_values(ascending=False)
    print(f"{case.name}/{law}  modal config n_on={n_on}  binding={df.shape[1]}  -> {dest}")
    print("  " + ", ".join(f"{c}={v:.0f}" for c, v in top.head(10).items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="a registered sample_only case name")
    parser.add_argument("--law", default="bernoulli", help="the Bernoulli nominal law name")
    parser.add_argument("--screened", action="store_true",
                        help="reuse the law's grown screened N-1 set (faster; needs screened.json)")
    args = parser.parse_args()
    case = registry.get(args.case)
    if not case.sample_only:
        raise SystemExit(f"{case.name} is not sample_only; use the enumerated pipeline")
    run(case, args.law, args.screened)


if __name__ == "__main__":
    main()
