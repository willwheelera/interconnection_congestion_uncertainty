"""Screen a Texas case's N-1 contingencies at the recipe the sweep actually solves.

The stock screener (screen_contingencies.py) screens through snapshot_mu, which uses the
case JSON's native 1.5 emergency rating with no ORDC relief and no shadow-price cap. Every
sweep instead solves the harness recipe: emergency factor EMERGENCY_FACTOR x long-term,
ORDC-shaped nodal relief on the case's relief bus set, and congestion duals capped at
SHADOW_PRICE_CAP by soft thermal limits. At the tightened factor contingencies bind that
stay slack at 1.5, so a set screened at 1.5 can omit binders the sweep needs.

This screens by driving the stress configs through the harness solve itself, so the kept
set matches the recipe the sweep uses. The harness reads the full case, so all of its
contingencies are candidates. HARNESS_RELIEF_SET must name the active case's relief bus
set (the one its sweep runs with) -- screening under more relief than the sweep gets would
understate congestion and drop binders.

Outputs (names from the harness CASE_REGISTRY entry for HARNESS_CASE):
  data/<case>_contingencies_screened.json  -- kept names + diagnostics.
  data/<case>_screened.json  -- full case with the contingency set filtered to the kept
      names; the SCDCOPF the sweep solves.

Usage:
    HARNESS_CASE=2025 HARNESS_RELIEF_SET=results/activsg2000_2025/relief_set_slim.json \
      uv run python src/texas/screen_harness_recipe.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "texas"))

from sample_harness import (ACTIVE_CASE, CASE, EMERGENCY_FACTOR, FULL_CASE, ORDC_BLOCKS,
                            ORDC_C_MAX, ORDC_C_MIN, ParallelTarget, RELIEF_BUSES,
                            RESULTS_NAME, SHADOW_PRICE_CAP)
from screen_contingencies import stress_configs

QUEUE = REPO_ROOT / "data" / "activsg2000_queue20.json"
RELIEF_FILE = REPO_ROOT / "results" / RESULTS_NAME / "relief_set_slim.json"
OUT_CASE = CASE
OUT_LIST = CASE.parent / CASE.name.replace("_screened.json", "_contingencies_screened.json")


def check_relief_matches_sweep() -> None:
    """Require the resolved relief buses to be the active case's sweep relief set."""
    want = set(json.loads(RELIEF_FILE.read_text()))
    assert RELIEF_BUSES == want, (
        f"HARNESS_RELIEF_SET must resolve to {RELIEF_FILE} ({len(want)} buses); "
        f"got {'unset (all load buses)' if RELIEF_BUSES is None else f'{len(RELIEF_BUSES)} buses'}")


def contingency_of(constraint_id: str) -> str:
    """The contingency name from a 'ctg:<name>:<line>' constraint id."""
    return constraint_id.split(":")[1]


def screen(target: ParallelTarget, configs: dict) -> dict[str, list]:
    """Union of contingencies binding (load-averaged) across the stress configs.

    Returns {contingency_name: [n_configs_bound, max_avg_mu]}. A contingency that binds in
    any snapshot of any stress config leaves a nonzero load-averaged mu, so the union of
    keys is the inclusive kept set.
    """
    agg: dict[str, list] = {}
    averages, _snaps = target.evaluate_detailed(list(configs.values()))
    for avg in averages:
        for cid, mu in avg.items():
            if cid.startswith("ctg:"):
                rec = agg.setdefault(contingency_of(cid), [0, 0.0])
                rec[0] += 1
                rec[1] = max(rec[1], mu)
    return agg


def write_screened_case(kept: list[str]) -> None:
    """Write the full case with its contingency set filtered to kept names."""
    case = json.loads(FULL_CASE.read_text())
    full = case["elements"]["contingency"]
    case["elements"]["contingency"] = {k: full[k] for k in kept}
    OUT_CASE.write_text(json.dumps(case))


def previous_kept() -> set[str]:
    """Contingency names in the existing screened list, empty if there is none."""
    if not OUT_LIST.exists():
        return set()
    return set(json.loads(OUT_LIST.read_text())["contingencies"])


def report(agg: dict[str, list], kept: list[str], n_total: int, before: set[str]) -> None:
    """Print the kept count, the strongest binders, and the change vs the old set."""
    print(f"kept {len(kept)} of {n_total} contingencies ({100 * len(kept) / n_total:.1f}%)")
    print("top by max load-averaged mu:")
    for cn in kept[:10]:
        print(f"  ctg {cn:<8} max_mu {agg[cn][1]:7.2f}  bound in {agg[cn][0]} configs")
    added, dropped = sorted(set(kept) - before), sorted(before - set(kept))
    print(f"\nvs previous set ({len(before)} kept): +{len(added)} new, -{len(dropped)} gone")
    print(f"  new:     {added}")
    print(f"  no longer binding: {dropped}")


def main():
    check_relief_matches_sweep()
    candidates = json.loads(QUEUE.read_text())["candidates"]
    configs = stress_configs(candidates)
    n_total = len(json.loads(FULL_CASE.read_text())["elements"]["contingency"])
    print(f"case {ACTIVE_CASE}: screening {n_total} contingencies of {FULL_CASE.name} at "
          f"factor {EMERGENCY_FACTOR}, ORDC {ORDC_C_MIN}->{ORDC_C_MAX} over {ORDC_BLOCKS} "
          f"blocks on {len(RELIEF_BUSES)} relief buses, cap ${SHADOW_PRICE_CAP}, "
          f"{len(configs)} stress configs", flush=True)
    before = previous_kept()

    target = ParallelTarget(case=str(FULL_CASE))
    target.start()
    try:
        agg = screen(target, configs)
    finally:
        target.close()

    kept = sorted(agg, key=lambda cn: agg[cn][1], reverse=True)
    OUT_LIST.write_text(json.dumps({
        "case": ACTIVE_CASE, "emergency_factor": EMERGENCY_FACTOR,
        "shadow_price_cap": SHADOW_PRICE_CAP, "relief_buses": sorted(RELIEF_BUSES),
        "n_total": n_total, "n_kept": len(kept), "stress_configs": list(configs),
        "contingencies": {cn: {"n_configs_bound": agg[cn][0], "max_mu": agg[cn][1]}
                          for cn in kept},
    }, indent=2))
    write_screened_case(kept)
    report(agg, kept, n_total, before)
    print(f"\nwrote {OUT_LIST}\nwrote {OUT_CASE}")


if __name__ == "__main__":
    main()
