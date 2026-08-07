"""Rank buses by how much power they can push onto the constraints that price congestion.

Two solves per snapshot -- every queue candidate off, then every one on -- give the
constraints that carry a nonzero mu.  One PTDF row per such constraint gives every bus
its shift factor onto that constraint, so a prospective site's leverage becomes a table
lookup instead of a solve per site.

Writes `bus_leverage.csv` (|shift factor|, buses x constraints) and `bus_leverage_mu.json`
(the largest mu each constraint reached) into the case results directory.

Usage:
    uv run python src/cases/texas/screen_bus_leverage.py --case activsg2000_original_v12
"""

import argparse
import json
import logging

import pandas as pd

from src.cases.registry import CASES
from src.model.scdcopf import build_persistent, build_audit_ptdf


def active_constraints(case) -> tuple[dict, dict]:
    """Return ({constraint: largest mu}, {constraint: {bus: shift factor}}).

    Both configuration extremes are solved at every snapshot: the all-off solve gives the
    constraints the incumbent fleet prices on its own, the all-on solve adds any the queue
    itself creates.  Constraints outside this union are not screened for.
    """
    weight, rows = {}, {}
    for snapshot in case.snapshots():
        pm = build_persistent(case, snapshot, relief_buses=case.relief_buses())
        ptdf = build_audit_ptdf(pm.md, case.case_file)
        for config in ([0] * case.ngen, [1] * case.ngen):
            for key, mu in pm.solve(config).items():
                weight[key] = max(weight.get(key, 0.0), mu)
                if key in rows:
                    continue
                kind, *rest = key.split(":")
                rows[key] = dict(ptdf.get_contingency_branch_ptdf_iterator(*rest)
                                 if kind == "ctg" else
                                 ptdf.get_branch_ptdf_iterator(rest[0]))
    return weight, rows


def leverage_frame(weight: dict, rows: dict) -> pd.DataFrame:
    """Buses x constraints table of |shift factor|, columns ordered by descending mu."""
    order = sorted(weight, key=weight.get, reverse=True)
    frame = pd.DataFrame({k: pd.Series(rows[k]) for k in order}).abs()
    frame.index.name = "bus"
    return frame


def print_summary(weight: dict, frame: pd.DataFrame, ntop: int) -> None:
    """Print the priced constraints and the buses with the most leverage on them."""
    print(f"{len(weight)} constraints carry mu")
    for key in frame.columns:
        best = frame[key].idxmax()
        print(f"  {key:16s} mu={weight[key]:9.1f}   best bus {best:>6} "
              f"sf={frame.loc[best, key]:.3f}")
    weighted = frame.mul(pd.Series(weight)[frame.columns], axis=1).sum(axis=1)
    top = weighted.sort_values(ascending=False).head(ntop)
    print(f"\ntop {ntop} buses by mu-weighted leverage ($/MWh per MW injected):")
    for bus, value in top.items():
        print(f"  bus {bus:>6}  {value:10.1f}   sum|sf| {frame.loc[bus].sum():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_original_v12")
    parser.add_argument("--ntop", type=int, default=15)
    args = parser.parse_args()

    logging.getLogger("egret").setLevel(logging.WARNING)
    case = CASES[args.case]
    weight, rows = active_constraints(case)
    frame = leverage_frame(weight, rows)

    case.results_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(case.results_dir / "bus_leverage.csv")
    (case.results_dir / "bus_leverage_mu.json").write_text(
        json.dumps({k: weight[k] for k in frame.columns}, indent=2))
    print_summary(weight, frame, args.ntop)
    print(f"\nwrote {case.results_dir / 'bus_leverage.csv'}")


if __name__ == "__main__":
    main()
