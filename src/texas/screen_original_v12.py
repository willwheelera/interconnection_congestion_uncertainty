"""Screen the original ACTIVSg2000 contingencies at the factor-1.2 soft-cap recipe.

SUPERSEDED by screen_harness_recipe.py, which does this for any case in the harness
CASE_REGISTRY. Kept for provenance: the committed activsg2000_original_v12 set came from
this script, which screens with relief at every load bus (it predates the v12
relief_set_slim.json). That gives the same kept set as the 9-bus sweep relief, since
select_relief_set chose the 9 as the only buses that ever shed under full relief and
relief elsewhere is an inactive column.

The stock screener (screen_contingencies.py) screens through snapshot_mu, which uses
the case's native 1.5 emergency rating and no shadow-price cap. The
activsg2000_original_v12 case instead solves the ORIGINAL fleet under the 2025 recipe
(emergency factor 1.2, ORDC relief, $3500 soft cap): at 1.2 several contingencies bind
that stay slack at 1.5, and an uncapped 1.2 solve is thermally infeasible. This
re-screens by driving the stress configs through the harness solve itself (which
applies factor 1.2 + the soft cap), so the kept set matches the recipe the full sweep
uses. The harness reads the full original case, so all 2756 contingencies are candidates.

Outputs:
  data/activsg2000_original_v12_contingencies_screened.json  -- kept names + diagnostics.
  data/activsg2000_original_v12_screened.json  -- original full case with the contingency
      set filtered to the kept names; the SCDCOPF the v12 sweep solves.

Usage:
    uv run python src/texas/screen_original_v12.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "texas"))

from sample_harness import ParallelTarget
from screen_contingencies import stress_configs

FULL_CASE = REPO_ROOT / "data" / "activsg2000.json"
QUEUE = REPO_ROOT / "data" / "activsg2000_queue20.json"
OUT_LIST = REPO_ROOT / "data" / "activsg2000_original_v12_contingencies_screened.json"
OUT_CASE = REPO_ROOT / "data" / "activsg2000_original_v12_screened.json"


def contingency_of(constraint_id: str) -> str:
    """The contingency name from a 'ctg:<name>:<line>' constraint id."""
    return constraint_id.split(":")[1]


def screen(target: ParallelTarget, configs: dict) -> dict[str, list]:
    """Union of contingencies binding (load-averaged) across the stress configs.

    Returns {contingency_name: [n_configs_bound, max_avg_mu]}. A contingency that
    binds in any snapshot of any stress config leaves a nonzero load-averaged mu, so
    the union of keys is the inclusive kept set.
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
    """Write the original full case with contingency set filtered to kept names."""
    case = json.loads(FULL_CASE.read_text())
    full = case["elements"]["contingency"]
    case["elements"]["contingency"] = {k: full[k] for k in kept}
    OUT_CASE.write_text(json.dumps(case))


def main():
    candidates = json.loads(QUEUE.read_text())["candidates"]
    configs = stress_configs(candidates)
    target = ParallelTarget(case=str(FULL_CASE))
    target.start()
    try:
        agg = screen(target, configs)
    finally:
        target.close()

    kept = sorted(agg, key=lambda cn: agg[cn][1], reverse=True)
    n_total = len(json.loads(FULL_CASE.read_text())["elements"]["contingency"])
    OUT_LIST.write_text(json.dumps({
        "n_total": n_total, "n_kept": len(kept), "stress_configs": list(configs),
        "contingencies": {cn: {"n_configs_bound": agg[cn][0], "max_mu": agg[cn][1]}
                          for cn in kept},
    }, indent=2))
    write_screened_case(kept)

    print(f"kept {len(kept)} of {n_total} contingencies "
          f"({100 * len(kept) / n_total:.1f}%)")
    print("top by max load-averaged mu:")
    for cn in kept[:10]:
        print(f"  ctg {cn:<8} max_mu {agg[cn][1]:7.2f}  bound in {agg[cn][0]} configs")
    print(f"\nwrote {OUT_LIST}\nwrote {OUT_CASE}")


if __name__ == "__main__":
    main()
