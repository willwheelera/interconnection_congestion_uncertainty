"""Prove a screened sweep was exact, by checking the contingencies screening dropped.

`cases.screen_contingencies` keeps only the contingencies that bind in a spread of stress
configurations -- on ACTIVSg2000 that is 33 of 2756, which is what makes a 2^20
enumeration affordable.  The kept set is a union over stress configurations, so it is not
exact by construction.

A sweep run with `--screened` now closes that gap as it goes, checking every solve against
the dropped contingencies and growing the set when one is violated, so this is a check on
a finished cache rather than the step that makes it exact.  It should find nothing.  A
violation means the cache was written before that loop existed -- both Texas caches were.

For each audited configuration this re-solves the screened SCDCOPF and then recomputes
every *dropped* contingency's post-outage flows at the solved dispatch.  A configuration
with no violated dropped contingency is proved exact: adding constraints that the solution
already satisfies cannot move the optimum, so the screened solution is also the full
problem's solution and its duals are the full problem's duals.  A configuration with a
violation is not wrong by a little -- it is a configuration whose sweep entry must be
discarded, and a signal to re-screen with those contingencies included.

Auditing every configuration would cost more than the sweep did (the check touches all
2723 dropped contingencies), so this audits where a violation is most likely and most
consequential -- the largest max_mu -- plus a uniform sample that stands for the rest.
Both counts are arguments, and the audited ids are recorded, so a run can be extended.

Writes `results/<case>/screening_audit.json`: the ids audited, how they were chosen, and
every violation found.

Usage:
    uv run python -m src.sweeps.audit_screening --case activsg2000_original_v12
    uv run python -m src.sweeps.audit_screening --case case14_queue --top 200 --random 800
"""

import argparse
import json

import numpy as np
import pandas as pd

from src.cases import registry
from src.model import evaluator


def config_bits(ids, ngen):
    """MSB-first bit rows for config ids -- the encoding every cache in the project uses."""
    return ((np.asarray(ids)[:, None] & (1 << np.arange(ngen - 1, -1, -1))) != 0).astype(int)


def select(case, n_top, n_random, seed):
    """(ids to audit, how many of them are top-max_mu) from the swept target cache."""
    max_mu = pd.read_parquet(case.target_cache, columns=["max_mu"])["max_mu"]
    top = max_mu.sort_values(ascending=False).index[:n_top].to_numpy()
    rest = np.setdiff1d(max_mu.index.to_numpy(), top)
    rng = np.random.default_rng(seed)
    sample = rng.choice(rest, size=min(n_random, len(rest)), replace=False)
    return np.concatenate([top, sample]), len(top)


def report(case, ids, n_top, violations, max_mu):
    """Print the verdict and return the record to write."""
    bad = {int(i): v for i, v in zip(ids, violations) if v}
    print(f"{case.name}: audited {len(ids)} configurations "
          f"({n_top} highest max_mu, {len(ids) - n_top} uniform), "
          f"{len(bad)} violate a dropped contingency")
    for cid, v in list(bad.items())[:10]:
        cn, (bn, flow, lim) = max(v.items(), key=lambda kv: kv[1][1] / kv[1][2])
        print(f"  config {cid} (max_mu {max_mu.get(cid, float('nan')):8.1f}): "
              f"{len(v)} violated, worst ctg {cn} branch {bn} "
              f"at {100 * flow / lim:.2f}% of limit")
    if not bad:
        print("  screened solution proved exact on every audited configuration")
    return {"case": case.name, "n_audited": len(ids), "n_top": n_top,
            "n_violating": len(bad), "audited_ids": [int(i) for i in ids],
            "violations": {str(cid): {cn: list(t) for cn, t in v.items()}
                           for cid, v in bad.items()}}


def main():
    parser = argparse.ArgumentParser()
    registry.add_case_arg(parser)
    parser.add_argument("--top", type=int, default=100,
                        help="audit this many configurations of largest max_mu")
    parser.add_argument("--random", type=int, default=400,
                        help="audit this many uniformly drawn configurations")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    case = registry.get(args.case)
    ids, n_top = select(case, args.top, args.random, args.seed)
    max_mu = pd.read_parquet(case.target_cache, columns=["max_mu"])["max_mu"]
    print(f"{case.name}: auditing {len(ids)} configurations against the contingencies "
          f"dropped from {case.case_file.name}", flush=True)

    with evaluator.for_case(case, n_workers=args.workers, screened=True) as ev:
        violations = ev.audit(config_bits(ids, case.ngen).tolist())

    out = case.results_dir / "screening_audit.json"
    out.write_text(json.dumps(report(case, ids, n_top, violations, max_mu), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
