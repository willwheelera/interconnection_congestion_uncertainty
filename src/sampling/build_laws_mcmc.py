"""Build and store the coupled MRF nominal laws for a sample-only case, without enumeration.

The enumerated cases build their laws in `build_laws.py`, whose `calibrate_field` and
marginal check both iterate over all 2^ngen configurations.  A sample-only case (the full
leveraged queue, ngen ~ 195) cannot do that, so its laws are built here instead:

  - the within-block coupling is scaled by block size (`distributions.block_c`) so a large
    fuel block does not saturate to all-on/all-off,
  - the field is calibrated exactly, without enumeration, by the count collapse the uniform
    within-block coupling permits (`structured_calibration.calibrate_structured_field`), so
    the law reproduces the queue's build probabilities to machine precision, and
  - the realized marginals are the analytic count-collapse marginals of the calibrated field
    (`structured_calibration.structured_marginals`, exact to machine precision), and the
    within-block bit correlations are measured from an exact iid draw
    (`structured_sampling.sample_structured`).  Gibbs is not used: at the block coupling
    `block_c` leaves, a fuel block has a metastable all-on mode a Gibbs sweep cannot cross, so
    a Gibbs run does not reproduce the calibrated marginals -- the same reason the cluster
    sweep draws these laws by the exact count collapse rather than by Gibbs.

The stored file is the same tracked `data/<case>/laws/<name>.json` the enumerated cases write,
so `mcmc_sweep` and any figure read a law the same way whatever built it.  `ref_block` (the
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
from src.sampling.mrf import MarkovRandomField
from src.sampling.structured_calibration import calibrate_structured_field, structured_marginals
from src.sampling.structured_sampling import sample_structured

LAWS = ["bernoulli", "mrf_cobuild", "mrf_substitute"]
CHECK_DRAWS = 20000                 # exact iid draws the within-block correlation is measured on


def coupling_matrix(name, groups, p, ngen, c, ref_block):
    """The pair coefficients J for one law: zero for bernoulli, block-scaled otherwise."""
    if name == "bernoulli":
        return np.zeros((ngen, ngen))
    if name == "mrf_cobuild":
        return cobuild_coupling(groups, ngen, c, ref_block)
    return substitute_coupling(groups, p, ngen, c, ref_block)


def within_block_corr(X, groups):
    """(mean, max) |correlation| over bit pairs sharing a fuel block, from an exact iid sample."""
    corr = np.corrcoef(X, rowvar=False)
    vals = [abs(corr[i, j]) for g in groups for a, i in enumerate(g) for j in g[a + 1:]]
    vals = np.array([v for v in vals if np.isfinite(v)])
    return (float(vals.mean()), float(vals.max())) if len(vals) else (0.0, 0.0)


def build_law(name, p, groups, ngen, c, ref_block, seed):
    """Calibrate one law's field to the marginals p under its block-scaled coupling.

    Bernoulli is exact (h = logit p, no coupling); the structured laws calibrate exactly by the
    count collapse of their uniform within-block coupling.  The realized marginals are the
    analytic count-collapse marginals of the calibrated field (machine-precise, no sampling),
    and the within-block correlations are measured on an exact iid draw.  Returns the MRF, its
    stored record, and the max marginal error.
    """
    J = coupling_matrix(name, groups, p, ngen, c, ref_block)
    if name == "bernoulli":
        h, realized = np.log(p / (1 - p)), p.copy()
    else:
        h = calibrate_structured_field(name, groups, p, ngen, c, ref_block)
        realized = structured_marginals(name, groups, h, p, c, ref_block)
    X = sample_structured(h, J, groups, CHECK_DRAWS, np.random.default_rng(seed + 1))
    mean_corr, max_corr = within_block_corr(X, groups)
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
