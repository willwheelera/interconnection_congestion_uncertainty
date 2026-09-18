"""Build and store the coupled MRF nominal laws for a sample-only case, without enumeration.

The enumerated cases build their laws in `build_laws.py`, whose `calibrate_field` and
marginal check both iterate over all 2^ngen configurations.  A sample-only case (the full
leveraged queue, ngen ~ 195) cannot do that, so its laws are built here instead:

  - the within-block coupling is scaled by block size (`distributions.block_c`) so a large
    fuel block does not saturate to all-on/all-off,
  - the field is calibrated by MCMC moment matching (`mrf.calibrate_field_mcmc`) so the law
    still reproduces the queue's build probabilities, and
  - the realized marginals and within-block bit correlations are verified by an independent
    Gibbs run (`mrf.gibbs_marginals`, `mrf.gibbs_sample`) rather than by enumeration.

The stored file is the same `results/<case>/laws/<name>.json` the enumerated cases write, so
`mcmc_sweep` and any figure read a law the same way whatever built it.  `ref_block` (the
block size the per-node coupling is normalized to) and `coupling` are recorded so the scaling
a law was built under is recoverable.

Usage:
    uv run python -m src.sampling.build_laws_mcmc --case activsg2000_v12_levfull
    uv run python -m src.sampling.build_laws_mcmc --case activsg2000_v12_levfull \
        --coupling 2.0 --ref-block 5
"""

import argparse

import numpy as np

from src.cases.registry import get_cases
from src.sampling.distributions import GROUP_COUPLING, cobuild_coupling, substitute_coupling
from src.sampling.law_store import save_law
from src.sampling.mrf import (MarkovRandomField, calibrate_field_mcmc, gibbs_marginals,
                              gibbs_sample)

LAWS = ["bernoulli", "mrf_cobuild", "mrf_substitute"]


def coupling_matrix(name, groups, p, ngen, c, ref_block):
    """The pair coefficients J for one law: zero for bernoulli, block-scaled otherwise."""
    if name == "bernoulli":
        return np.zeros((ngen, ngen))
    if name == "mrf_cobuild":
        return cobuild_coupling(groups, ngen, c, ref_block)
    return substitute_coupling(groups, p, ngen, c, ref_block)


def within_block_corr(X, groups):
    """(mean, max) |correlation| over bit pairs sharing a fuel block, from a Gibbs sample."""
    corr = np.corrcoef(X, rowvar=False)
    vals = [abs(corr[i, j]) for g in groups for a, i in enumerate(g) for j in g[a + 1:]]
    vals = np.array([v for v in vals if np.isfinite(v)])
    return (float(vals.mean()), float(vals.max())) if len(vals) else (0.0, 0.0)


def build_law(name, p, groups, ngen, c, ref_block, seed):
    """Calibrate one law's field to the marginals p under its block-scaled coupling.

    Bernoulli is exact (h = logit p, no coupling); the coupled laws calibrate by MCMC.
    Returns the MRF, its stored record, and the max marginal error the check found.
    """
    J = coupling_matrix(name, groups, p, ngen, c, ref_block)
    if name == "bernoulli":
        h = np.log(p / (1 - p))
    else:
        h, _ = calibrate_field_mcmc(J, p, seed=seed)
    realized = gibbs_marginals(h, J, seed=seed + 1)
    mean_corr, max_corr = within_block_corr(gibbs_sample(h, J, seed=seed + 2), groups)
    record = {"p_target": p.tolist(), "p_exact": realized.tolist(),
              "groups": groups, "coupling": (None if name == "bernoulli" else c),
              "ref_block": ref_block, "mean_abs_corr": mean_corr, "max_abs_corr": max_corr}
    return MarkovRandomField(h, J), record, float(np.abs(realized - p).max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="a sample-only case name")
    parser.add_argument("--dists", default=",".join(LAWS))
    parser.add_argument("--coupling", type=float, default=GROUP_COUPLING,
                        help="within-block pair coefficient at the reference block size")
    parser.add_argument("--ref-block", type=int, default=5,
                        help="block size the per-node coupling is normalized to")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for case in get_cases(args.case):
        if not case.sample_only:
            print(f"{case.name}: not sample_only; use build_laws.py")
            continue
        p, groups = case.build_p(), case.candidate_groups()
        print(f"=== {case.name}: ngen {case.ngen}, blocks {[len(g) for g in groups]} ===")
        for name in args.dists.split(","):
            mrf, record, err = build_law(name, p, groups, case.ngen, args.coupling,
                                         args.ref_block, args.seed)
            path = save_law(case, name, mrf, record)
            print(f"{name:16s} marginal error {err:8.1e}  within-block |corr| "
                  f"mean {record['mean_abs_corr']:.3f} max {record['max_abs_corr']:.3f}  "
                  f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
