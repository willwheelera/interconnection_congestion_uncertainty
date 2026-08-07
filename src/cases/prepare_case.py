"""Build the two inputs a sweep needs: the relief bus set, then the screened contingencies.

Both depend on where the case's queue sits, so a new queue gets its own pair rather than
inheriting another case's, and they must run in this order -- screening solves at the
case's recipe, and the recipe includes its relief set.

They run as one job because neither is long.  Relief selection is 806 configurations x the
case's snapshots, screening is 34, and both are spread over the whole node by the same
worker pool the sweep uses: together minutes to a couple of hours against a sweep measured
in tens.  A step whose output already exists is skipped, so re-running this after adding
only a queue costs only what is actually missing.

Usage:
    uv run python -m src.cases.prepare_case --case activsg2000_v12_lev0
    uv run python -m src.cases.prepare_case --case activsg2000_v12_lev0 --collect 200 --n-stress 16
"""

import argparse

from src.cases import registry, screen_contingencies
from src.cases.texas import select_relief_set


def prepare(case, collect, validate, n_stress, workers, seed):
    """Select the relief set and screen the contingencies, skipping whichever exists."""
    if case.relief_file is None:
        select_relief_set.select(case, collect, validate, seed, workers)
    else:
        print(f"{case.name}: relief set already at {case.relief_file.name}, skipping")

    if not case.screened_file.exists():
        screen_contingencies.screen(case, n_stress, workers, seed)
    else:
        print(f"{case.name}: contingencies already screened to "
              f"{case.screened_file.name}, skipping")


def main():
    parser = argparse.ArgumentParser()
    registry.add_case_arg(parser)
    parser.add_argument("--collect", type=int, default=400)
    parser.add_argument("--validate", type=int, default=200)
    parser.add_argument("--n-stress", type=int, default=32)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare(registry.get(args.case), args.collect, args.validate, args.n_stress,
            args.workers, args.seed)


if __name__ == "__main__":
    main()
