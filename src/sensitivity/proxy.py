"""Barrier-sensitivity proxy s_k(x) that ranks which generator flip to evaluate.

The ascent samplers can climb S(x) without evaluating every neighbor if they know,
cheaply, which flips are most likely to raise S.  The proxy is the barrier-regularized
sensitivity of the congestion duals to injection at each candidate generator's bus,

    s_k(x) = sum_l |d(mu_l) / dP_bus(k)|   at the OPF solution of x,

which is large exactly where a flip is about to trigger an active-set change (the
barrier eps keeps it finite there).  It reuses compute_sensitivity_barrier via
build_sensitivity_cache.sensitivity_matrix, the same quantity informed_mcmc scores.

Two interchangeable backends, both exposing score(config_id) -> ngen-vector:

  - CachedProxy: reduce a precomputed sensitivity_cache.parquet (branch x bus) to the
    per-generator vector.  A fast table lookup; needs the cache.
  - LiveProxy: solve the config's OPF and its barrier sensitivity on demand -- no
    cache, works on any generic (DCOPF) case, but a real solve per queried config.

The proxy is a byproduct of the config's OPF solution, so it charges no extra
Evaluator budget; LiveProxy is only slower in wall-clock.
"""

import numpy as np
import pandas as pd

from egret.data.model_data import ModelData

from utils.availability import id_to_avail
from model.dcopf import create_case, solve_opf, store_branch_duals
from sensitivity.build_cache import sensitivity_matrix
from sampling.problem import REPO_ROOT


def generator_buses(case_json):
    """Buses of each generator in availability-bit order (MSB = first generator)."""
    md = ModelData.read(str(case_json))
    return [g["bus"] for g in md.data["elements"]["generator"].values()]


class CachedProxy:
    """Per-generator score by lookup into a precomputed branch x bus sensitivity cache."""

    def __init__(self, sens_path, gen_buses):
        df = pd.read_parquet(sens_path)                        # cols (branch, bus), idx avail_id
        per_bus = df.abs().T.groupby(level="bus").sum().T      # |.| summed over branches
        self.table = per_bus.reindex(columns=gen_buses).fillna(0.0)
        self.ngen = len(gen_buses)
        self.seen = set(self.table.index.astype(int))

    def score(self, config_id):
        if int(config_id) not in self.seen:
            return np.zeros(self.ngen)
        return self.table.loc[int(config_id)].to_numpy()


class LiveProxy:
    """Per-generator score by solving the config's OPF + barrier sensitivity on demand."""

    def __init__(self, case_json, ngen, gen_buses, solver="gurobi"):
        self.ref = ModelData.read(str(case_json))
        self.ngen, self.gen_buses, self.solver = ngen, gen_buses, solver

    def score(self, config_id):
        md = create_case(self.ref, id_to_avail(int(config_id), self.ngen))
        md_sol, m = solve_opf(md, solver=self.solver)
        store_branch_duals(m, md_sol)
        dmudL, _branches, buses = sensitivity_matrix(md_sol)
        per_bus = pd.Series(np.abs(dmudL).sum(axis=0), index=buses)
        return per_bus.reindex(self.gen_buses).fillna(0.0).to_numpy()


def make_proxy(kind, case, ngen):
    """Build the requested proxy for a generic gen-per-bus case, or None if unavailable.

    kind is None (no proxy -> ascent 'all' only), "cached", or "live".  Texas cases
    have no generic per-bus sensitivity and get None (Texas-live is a later extension).
    """
    if kind is None:
        return None
    case_json = REPO_ROOT / "data" / f"{case}.json"
    if not case_json.exists():
        return None
    gen_buses = generator_buses(case_json)
    if kind == "cached":
        sens = REPO_ROOT / "results" / case / "sensitivity_cache.parquet"
        return CachedProxy(sens, gen_buses) if sens.exists() else None
    return LiveProxy(case_json, ngen, gen_buses)


def main():
    """Sanity: cached vs live top-flip agreement on a few case14 configs."""
    case, ngen = "case14_gen_per_bus", 14
    cached, live = make_proxy("cached", case, ngen), make_proxy("live", case, ngen)
    for cid in (5000, 9000, 12000):
        c, l = cached.score(cid), live.score(cid)
        print(f"id {cid}: cached argmax {int(np.argmax(c))}  live argmax {int(np.argmax(l))}  "
              f"corr {np.corrcoef(c, l)[0, 1]:.3f}")


if __name__ == "__main__":
    main()
