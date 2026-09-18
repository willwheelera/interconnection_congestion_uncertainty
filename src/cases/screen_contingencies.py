"""Screen a case's N-1 contingency set down to the ones that ever bind.

A security-constrained DCOPF pays for every contingency it carries through EGRET's lazy
LODF loop, and on these networks most single-branch outages never produce a binding
post-contingency limit under any buildout.  This solves a spread of stress configurations
against the *full* set and keeps the union of contingencies that bind in at least one.

**Screen at the recipe you solve at.**  Which contingencies bind depends on the emergency
factor, the relief the model has available, and the shadow-price cap, so the stress solves
run through the same `model.scdcopf` path a sweep uses, at the case's own recipe.  A set
screened at a case JSON's native rating (typically a looser 1.5 factor) omits binders that
the tightened factor needs.

**The kept set is a starting point, not a final answer.**  It is a union over the stress
configurations, so a configuration outside them can bind a dropped contingency -- on
case14_queue one configuration in 62 does, and its max mu comes out 2% low.  A screened
sweep therefore grows the set as it runs: every solve checks the dropped contingencies at
its own dispatch, and a chunk that turns one up is redone after `augment_screened` adds it
here.  Screening is what keeps a 2^20 enumeration affordable; the growth is what keeps it
exact.  The better this screen is, the fewer times a sweep has to stop and rebuild.

Writes `data/<case>_screened.json` (the full case with its contingency set filtered) and
`data/<case>_contingencies_screened.json` (the kept names, the recipe screened under, and
the stress set, so a screen is reproducible from its own record).

A screened record with no `emergency_factor` predates this file and was screened at the
case JSON's native rating, not at the recipe.  `activsg2000_original_v12`'s is one, and its
33-contingency set does not survive the audit; `activsg2000_2025`, screened here, keeps 104
on the same network.  Re-screening replaces a set that a completed sweep depends on, so do
it deliberately -- but a record without a recipe is not to be trusted.

Usage:
    uv run python -m src.cases.screen_contingencies --case case118_queue --n-stress 32
    uv run python -m src.cases.screen_contingencies --case all --workers 16
"""

import argparse
import json

import numpy as np

from src.cases import registry
from src.model import evaluator
from src.model.relief import ALL_LOAD_BUSES


def stress_configs(ngen, n_stress, seed):
    """Nothing built, everything built, and n_stress half-built configurations.

    The extremes bracket the space and the half-built draws probe its middle, where a
    dense build perturbs the network far more than a sparse one does.
    """
    rng = np.random.default_rng(seed)
    configs = [[0] * ngen, [1] * ngen]
    for _ in range(n_stress):
        on = set(rng.choice(ngen, ngen // 2, replace=False).tolist())
        configs.append([1 if j in on else 0 for j in range(ngen)])
    return configs


def binding_contingencies(per_config_mu):
    """{contingency: [configs bound in, max mu]} over every stress configuration's duals."""
    found = {}
    for mus in per_config_mu:
        for cid, mu in mus.items():
            if not cid.startswith("ctg:"):
                continue
            rec = found.setdefault(cid.split(":")[1], [0, 0.0])
            rec[0] += 1
            rec[1] = max(rec[1], mu)
    return found


def write_screened(case, kept, found, configs, n_total):
    """Write the filtered case and the screening record."""
    data = json.loads(case.case_file.read_text())
    full = data["elements"]["contingency"]
    data["elements"]["contingency"] = {k: full[k] for k in kept}
    case.screened_file.write_text(json.dumps(data))

    relief = case.relief_buses()
    case.contingency_file.write_text(json.dumps({
        "case": case.name,
        "emergency_factor": case.recipe.emergency_factor,
        "shadow_price_cap": case.recipe.shadow_price_cap,
        "relief_buses": ALL_LOAD_BUSES if relief == ALL_LOAD_BUSES else sorted(relief),
        "n_total": n_total, "n_kept": len(kept),
        "stress_configs": configs,
        "contingencies": {c: {"n_configs_bound": found[c][0], "max_mu": found[c][1]}
                          for c in kept},
    }, indent=2))


def augment_screened(case, names, chunk):
    """Add contingencies to an existing screened set in place; return the new kept count.

    Called by the sweep when a solve violates a contingency screening dropped: the set the
    workers read has to grow before that configuration can be solved correctly.  The
    filtered case is rebuilt from the full one so the additions carry their real data, and
    the record remembers which chunk each addition surfaced in -- a screened set is then
    reproducible as "screening plus what the sweep found", not an unexplained file.
    """
    data = json.loads(case.case_file.read_text())
    record = json.loads(case.contingency_file.read_text())
    kept = list(record["contingencies"]) + [n for n in names
                                            if n not in record["contingencies"]]
    data["elements"]["contingency"] = {k: data["elements"]["contingency"][k] for k in kept}
    case.screened_file.write_text(json.dumps(data))

    for n in names:
        record["contingencies"].setdefault(
            n, {"n_configs_bound": 0, "max_mu": 0.0, "found_in_sweep_chunk": chunk})
    record["n_kept"] = len(kept)
    case.contingency_file.write_text(json.dumps(record, indent=2))
    return len(kept)


def previous_kept(case):
    """Contingency names in the existing screened record, empty if there is none."""
    if not case.contingency_file.exists():
        return set()
    return set(json.loads(case.contingency_file.read_text())["contingencies"])


def report(case, found, kept, n_total, before):
    """Print the kept count, the strongest binders, and the change against the old set."""
    print(f"{case.name}: kept {len(kept)} of {n_total} contingencies "
          f"({100 * len(kept) / n_total:.1f}%)")
    for c in kept[:10]:
        print(f"  ctg {c:<10} max mu {found[c][1]:8.2f} $/MWh  "
              f"bound in {found[c][0]} configs")
    added, dropped = sorted(set(kept) - before), sorted(before - set(kept))
    print(f"vs previous set ({len(before)} kept): +{len(added)} new, -{len(dropped)} gone")
    if added:
        print(f"  new:               {added}")
    if dropped:
        print(f"  no longer binding: {dropped}")


def screen(case, n_stress, workers, seed):
    """Solve the stress configurations against the full set and keep the binders."""
    configs = stress_configs(case.ngen, n_stress, seed)
    n_total = len(json.loads(case.case_file.read_text())["elements"]["contingency"])
    relief = case.relief_buses()
    print(f"{case.name}: screening {n_total} contingencies of {case.case_file.name} at "
          f"factor {case.recipe.emergency_factor}, ORDC {case.recipe.ordc_min:g}->"
          f"{case.recipe.ordc_max:g} over {case.recipe.ordc_blocks} blocks on "
          f"{ALL_LOAD_BUSES if relief == ALL_LOAD_BUSES else len(relief)} relief buses, cap "
          f"${case.recipe.shadow_price_cap:g}, {len(configs)} stress configs", flush=True)

    before = previous_kept(case)
    with evaluator.for_case(case, relief, n_workers=workers, screened=False) as target:
        found = binding_contingencies(target.evaluate(configs))

    kept = sorted(found, key=lambda c: found[c][1], reverse=True)
    write_screened(case, kept, found, configs, n_total)
    report(case, found, kept, n_total, before)
    print(f"wrote {case.screened_file}\nwrote {case.contingency_file}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="a case name, a name substring, or 'all'")
    parser.add_argument("--n-stress", type=int, default=32)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for case in registry.get_cases(args.case):
        if case.model != "scdcopf":
            continue
        screen(case, args.n_stress, args.workers, args.seed)


if __name__ == "__main__":
    main()
