"""Screen the N-1 contingency set down to the ones that actually bind.

Running the security-constrained DCOPF against all 2756 contingencies makes each
solve ~6x slower than plain DCOPF because of the lazy LODF screening loop. Almost
all of those contingencies never produce a binding post-contingency limit under
our load snapshots. This screens the full set once, at a spread of stress
configurations, and keeps only the contingencies that bind in at least one
(config, snapshot) cell.

The stress configs bracket the extremes (all off, all on) and then sample many
half-loaded configurations -- each with exactly HALF the candidates online, the
specific ten drawn at random so the spatial/fuel composition varies. Denser
configs bind far more contingencies than sparse ones (a single fuel group barely
perturbs the network), so half-on draws probe the operating region the sampler
actually visits. The union of contingencies binding across these is the frozen
set; a contingency that never binds here is dropped but recovered by the audit
in sample_harness if the sampler ever reaches a config that violates it.

Screens the expanded 2025 model (data/activsg2000_2025.json), with each snapshot's
renewables derated to its capacity factor so the kept set reflects realistic wind
and solar output rather than nameplate injection.

Outputs:
  data/activsg2000_2025_contingencies_screened.json  -- kept names + per-contingency
      binding count and max mu (diagnostics), and the stress configs used.
  data/activsg2000_2025_screened.json  -- the full case with contingency filtered to
      the kept set; a drop-in replacement for the sampler's SCDCOPF.

Usage:
    uv run python src/texas/screen_contingencies.py
"""

import copy
import json
import random
import sys
from pathlib import Path

from egret.data.model_data import ModelData

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "texas"))

from load_averaged_target import add_queue_generators, snapshot_mu

CASE = REPO_ROOT / "data" / "activsg2000_2025.json"
QUEUE = REPO_ROOT / "data" / "activsg2000_queue20.json"
COLLECTION = REPO_ROOT / "data" / "activsg2000_load_collection.json"
OUT_LIST = REPO_ROOT / "data" / "activsg2000_2025_contingencies_screened.json"
OUT_CASE = REPO_ROOT / "data" / "activsg2000_2025_screened.json"

# Number of half-loaded random stress configs, and their draw seed.
N_STRESS = 24
STRESS_SEED = 0


def stress_configs(candidates: list[dict]) -> dict[str, list[int]]:
    """Extremes plus N_STRESS configs each with exactly half the candidates on."""
    n = len(candidates)
    rng = random.Random(STRESS_SEED)
    configs = {"all_off": [0] * n, "all_on": [1] * n}
    for i in range(N_STRESS):
        on = set(rng.sample(range(n), n // 2))
        configs[f"half{i}"] = [1 if j in on else 0 for j in range(n)]
    return configs


def binding_in_config(md_base, candidates, snapshots, config, solver) -> dict[str, list]:
    """Return {contingency_name: [n_cells_bound, max_mu]} for one config."""
    md_config = copy.deepcopy(md_base)
    add_queue_generators(md_config, candidates, config)
    found: dict[str, list] = {}
    for s in snapshots:
        mus = snapshot_mu(md_config, s["zone_scales"], solver, s.get("renewable_cf"))
        for cid, mu in mus.items():
            if not cid.startswith("ctg:"):
                continue
            cn = cid.split(":")[1]
            rec = found.setdefault(cn, [0, 0.0])
            rec[0] += 1
            rec[1] = max(rec[1], mu)
    return found


def merge(agg: dict[str, list], found: dict[str, list]) -> None:
    """Accumulate a config's binding contingencies into the running union."""
    for cn, (count, mx) in found.items():
        rec = agg.setdefault(cn, [0, 0.0])
        rec[0] += count
        rec[1] = max(rec[1], mx)


def write_screened_case(kept: list[str]) -> None:
    """Write the full case with the contingency set filtered to kept names."""
    case = json.loads(CASE.read_text())
    full = case["elements"]["contingency"]
    case["elements"]["contingency"] = {k: full[k] for k in kept}
    OUT_CASE.write_text(json.dumps(case))


def main():
    md_base = ModelData.read(str(CASE))
    candidates = json.loads(QUEUE.read_text())["candidates"]
    snapshots = json.loads(COLLECTION.read_text())["snapshots"]
    configs = stress_configs(candidates)

    agg: dict[str, list] = {}
    for label, cfg in configs.items():
        found = binding_in_config(md_base, candidates, snapshots, cfg, "gurobi")
        merge(agg, found)
        print(f"{label:<12} bound {len(found):>4} contingencies   "
              f"union now {len(agg)}")

    kept = sorted(agg, key=lambda cn: agg[cn][1], reverse=True)
    n_total = len(md_base.data["elements"]["contingency"])
    OUT_LIST.write_text(json.dumps({
        "n_total": n_total,
        "n_kept": len(kept),
        "stress_configs": list(configs),
        "contingencies": {cn: {"n_cells_bound": agg[cn][0], "max_mu": agg[cn][1]}
                          for cn in kept},
    }, indent=2))
    write_screened_case(kept)

    print(f"\nkept {len(kept)} of {n_total} contingencies "
          f"({100 * len(kept) / n_total:.1f}%)")
    print("top by max mu:")
    for cn in kept[:10]:
        print(f"  ctg {cn:<8} max_mu {agg[cn][1]:7.2f}  bound in {agg[cn][0]} cells")
    print(f"\nwrote {OUT_LIST}\nwrote {OUT_CASE}")


if __name__ == "__main__":
    main()
