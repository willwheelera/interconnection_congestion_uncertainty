"""Screen a queue case's N-1 contingency set down to the ones that ever bind.

The security-constrained DCOPF pays for every contingency it carries through EGRET's lazy
LODF screening loop, and on these networks most single-branch outages never produce a
binding post-contingency limit under any buildout.  This solves a spread of stress
configurations against the full set and keeps the union of contingencies that bind in at
least one of them.

The stress set brackets the extremes (nothing built, everything built) and then samples
half-built configurations, whose specific composition varies -- a dense build perturbs the
network far more than a sparse one, so half-built draws probe the region the sweep
actually visits.

The result is an approximation, not a reduction: the kept set is a union over the stress
configurations, and a configuration outside them can bind a contingency that was dropped.
Measured on case14, one configuration in 62 does, and its max mu comes out 2% low.  So this
is for fast diagnostic runs, and the exhaustive ground-truth sweep uses the full set --
enumerate_configs only screens when asked (--screened).

Writes data/<case>_screened.json (a drop-in replacement, which sample_harness prefers when
present) and data/<case>_contingencies_screened.json (the kept names plus diagnostics).

Usage:
    uv run python -m ieee_queue.screen_contingencies
    uv run python -m ieee_queue.screen_contingencies --case case118_queue --n-stress 32
"""

import argparse
import json

import numpy as np

from ieee_queue.build_queue_case import NETWORKS
from ieee_queue.sample_harness import DATA, ParallelTarget


def stress_configs(ncand, n_stress, seed):
    """Nothing built, everything built, and n_stress half-built configurations."""
    rng = np.random.default_rng(seed)
    configs = [[0] * ncand, [1] * ncand]
    for _ in range(n_stress):
        on = set(rng.choice(ncand, ncand // 2, replace=False).tolist())
        configs.append([1 if j in on else 0 for j in range(ncand)])
    return configs


def binding_contingencies(results):
    """{contingency: [cells bound, max mu]} over every stress configuration's duals."""
    found = {}
    for mus, _, _ in results:
        for cid, mu in mus.items():
            if not cid.startswith("ctg:"):
                continue
            rec = found.setdefault(cid.split(":")[1], [0, 0.0])
            rec[0] += 1
            rec[1] = max(rec[1], mu)
    return found


def write_screened(case, kept, found, n_total, n_stress):
    """Write the case filtered to the kept contingencies, plus the screening record."""
    data = json.loads((DATA / f"{case}.json").read_text())
    full = data["elements"]["contingency"]
    data["elements"]["contingency"] = {k: full[k] for k in kept}
    (DATA / f"{case}_screened.json").write_text(json.dumps(data))
    (DATA / f"{case}_contingencies_screened.json").write_text(json.dumps({
        "n_total": n_total, "n_kept": len(kept), "n_stress": n_stress,
        "contingencies": {c: {"n_cells_bound": found[c][0], "max_mu": found[c][1]}
                          for c in kept}}, indent=2))


def screen(case, n_stress, workers, seed):
    """Solve the stress configurations against the full contingency set and keep the binders."""
    target = ParallelTarget(case, n_workers=workers, screened=False)
    configs = stress_configs(len(target.candidates), n_stress, seed)
    target.start()
    results = target.evaluate_detailed(configs)
    target.close()

    found = binding_contingencies(results)
    kept = sorted(found, key=lambda c: found[c][1], reverse=True)
    n_total = len(json.loads((DATA / f"{case}.json").read_text())["elements"]["contingency"])
    write_screened(case, kept, found, n_total, n_stress)
    print(f"{case}: kept {len(kept)} of {n_total} contingencies "
          f"({100 * len(kept) / n_total:.0f}%) over {len(configs)} stress configs")
    for c in kept[:8]:
        print(f"  ctg {c:<6} max mu {found[c][1]:8.2f} $/MWh  bound in {found[c][0]} cells")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", choices=[*NETWORKS, "all"])
    parser.add_argument("--n-stress", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for case in (NETWORKS if args.case == "all" else [args.case]):
        screen(case, args.n_stress, args.workers, args.seed)


if __name__ == "__main__":
    main()
