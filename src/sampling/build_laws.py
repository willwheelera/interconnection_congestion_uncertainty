"""Build and store a case's nominal laws -- run once, after which nothing recomputes them.

Each named law is constructed at the case's own build probabilities and technology
groups, verified against those marginals by exact enumeration, and written to
`results/<case>/laws/<name>.json`.  Every coupled law solves its field over all 2^ngen
configurations, which at ngen = 20 costs seconds for a weak coupling and tens of seconds
for the structured ones; that cost is paid here and nowhere else.

Usage:
    uv run python -m src.sampling.build_laws --case activsg2000_original_v12
    uv run python -m src.sampling.build_laws --case all --dists mrf_a
"""

import argparse

import numpy as np

from src.cases.registry import get_cases
from src.sampling.distributions import (COUPLINGS, DIST_NAMES, GROUP_COUPLING, P_STD,
                                    STRUCTURED_NAMES, case_law_names, make_dist,
                                    marginal_vector)
from src.sampling.law_store import save_law
from src.sampling.mrf import all_configs


def exact_marginals(mrf):
    """P[x_i = 1] under the law, by enumerating every configuration."""
    bits = all_configs(mrf.ngen)
    lw = mrf.log_weight(bits)
    w = np.exp(lw - lw.max())
    return (w / w.sum()) @ bits


def build_law(case, name, p_std):
    """One law at the case's marginals and groups, with the record stored beside it."""
    groups = case.candidate_groups()
    p = marginal_vector(case.ngen, case.nominal_p(), p_std ** 2)
    mrf = make_dist(name, case.ngen, case.nominal_p(), p_std, groups)
    exact = exact_marginals(mrf)
    record = {"p_target": p.tolist(), "p_exact": exact.tolist(), "p_std": p_std,
              "groups": groups,
              "coupling": (GROUP_COUPLING if name in STRUCTURED_NAMES
                           else COUPLINGS.get(name))}
    return mrf, record, float(np.abs(exact - p).max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="a case name, a name substring, or 'all'")
    parser.add_argument("--dists", default=",".join(DIST_NAMES))
    parser.add_argument("--p-std", type=float, default=P_STD,
                        help="marginal spread for the cases without a queue")
    args = parser.parse_args()

    requested = args.dists.split(",")
    for case in get_cases(args.case):
        buildable = case_law_names(case, requested)
        for name in requested:
            if name not in buildable:
                print(f"{case.name:26s} {name:16s} skipped: no queue, so no groups")
                continue
            mrf, record, err = build_law(case, name, args.p_std)
            path = save_law(case, name, mrf, record)
            print(f"{case.name:26s} {name:16s} marginal error {err:8.1e}  -> {path}",
                  flush=True)


if __name__ == "__main__":
    main()
