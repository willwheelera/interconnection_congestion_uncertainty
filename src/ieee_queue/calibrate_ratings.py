"""Relax branch ratings until the incumbent-only grid is N-1 secure.

build_queue_case sets provisional ratings from unconstrained flows, which is not a
security criterion: on a meshed network, rating one line tight pushes flow onto the next,
so the incumbent-only (all-off) configuration can end up congested or overloaded before a
single queue project is built.  A testbed like that inverts the question -- the worst
congestion sits at the empty queue and buildout relieves it.

This drives the case toward the opposite fixed point.  Each round solves the all-off N-1
SCDCOPF under the sampling recipe (sample_harness) and multiplies by RELAX_STEP the
long-term rating of every line that is overloaded or that carries a limit priced above
SECURE_MU, base case or post-contingency.  Rounds stop once the incumbent-only grid needs
no overload or load shed and prices nothing above SECURE_MU.

The threshold matters: relaxing every binding line instead would inflate exactly the
corridors that are supposed to stay tight, and the finished case would show no congestion
under buildout either.  Mild congestion on the incumbent grid is realistic and is left
alone; what the pass removes is severe, near-cap pricing that would saturate the case
before a single project is built.

Ratings are rewritten in place in data/<case>.json, which makes the pass idempotent: a
case that is already secure is left untouched.

Usage:
    uv run python -m ieee_queue.calibrate_ratings
    uv run python -m ieee_queue.calibrate_ratings --case case118_queue --rounds 40
"""

import argparse
import json
from pathlib import Path

from ieee_queue.build_queue_case import NETWORKS
from ieee_queue.sample_harness import DATA, SHADOW_PRICE_CAP, ParallelTarget, scalar_target

RELAX_STEP = 1.10                       # rating multiplier applied to a line each round
SECURE_MU = 0.05 * SHADOW_PRICE_CAP     # all-off congestion allowed to remain ($/MWh)


def binding_lines(mus, overload):
    """Monitored lines behind the severely-priced limits and the overloads of one solve.

    Constraint ids are "base:{line}" or "ctg:{outage}:{line}"; the monitored line is the
    last field in both, and it is the line whose rating has to grow."""
    severe = [cid for cid, mu in mus.items() if mu > SECURE_MU]
    return {cid.split(":")[-1] for cid in (*severe, *overload)}


def relax(case_path, lines):
    """Multiply the long-term rating of each named line by RELAX_STEP, in place."""
    case = json.loads(Path(case_path).read_text())
    branches = case["elements"]["branch"]
    for name in lines:
        branches[name]["rating_long_term"] *= RELAX_STEP
    Path(case_path).write_text(json.dumps(case))


def secure(case, rounds, workers):
    """Relax ratings round by round until the all-off configuration prices no congestion.

    Each round rebuilds the workers, because the ratings they were built from changed."""
    case_path = DATA / f"{case}.json"
    for r in range(rounds):
        target = ParallelTarget(case, n_workers=workers, screened=False)
        target.start()
        mus, overload, shed = target.evaluate_detailed([[0] * len(target.candidates)])[0]
        target.close()
        peak = scalar_target(mus)
        lines = binding_lines(mus, overload)
        print(f"  round {r:>2d}  all-off max mu {peak:9.2f} $/MWh   "
              f"binding {len(mus)}  overload {max(overload.values(), default=0.0):6.2f} MW   "
              f"shed {shed:6.2f} MW   relaxing {len(lines)} lines")
        if peak <= SECURE_MU and not overload and shed <= 0.0:
            return r, peak
        relax(case_path, lines)
    return rounds, peak


def report(case):
    """Print the case's rating spread after calibration."""
    case_data = json.loads((DATA / f"{case}.json").read_text())
    ratings = [b["rating_long_term"] for b in case_data["elements"]["branch"].values()]
    print(f"  ratings {min(ratings):.1f}-{max(ratings):.1f} MW  "
          f"total {sum(ratings):.0f} MW")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", choices=[*NETWORKS, "all"])
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    for case in (NETWORKS if args.case == "all" else [args.case]):
        print(f"=== {case} ===")
        used, peak = secure(case, args.rounds, args.workers)
        print(f"  secure after {used} rounds (all-off max mu {peak:.4g} $/MWh)")
        report(case)


if __name__ == "__main__":
    main()
