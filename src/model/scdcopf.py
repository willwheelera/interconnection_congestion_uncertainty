"""One configuration -> congestion duals, under the case's recipe.

This is the single place the physical model lives.  Everything above it (sweeps,
samplers, descriptors) treats a configuration as a bit vector and gets back
`{constraint_id: mu}`; everything below it is EGRET.

A *configuration* is a bit vector over the case's queue candidates, 1 if that project
is built.  Rather than rebuild the model per configuration, `build_persistent` builds it
once with every candidate present and `solve_config` applies a configuration by setting
each candidate's upper bound to its capacity or to zero and warm re-solving.  That
carries the basis, the PTDF factorization, and the accumulated lazy contingency
constraints across configurations, which is what makes a 2^20 sweep affordable.

Constraint ids are `base:{line}` for normal thermal limits and `ctg:{contingency}:{line}`
for post-contingency ones, the same in both case families.

The recipe (emergency factor, ORDC relief, shadow-price cap) comes off the case record,
so a case is solved at its own recipe with no environment variables and no per-family
branch here.  `snapshot=None` is the single-load-level path used by the IEEE queue cases;
a snapshot record scales loads by weather zone and derates renewables to that hour.
"""

import json
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import egret.common.lazy_ptdf_utils as lpu
from egret.common.solver_interface import _solve_model
from egret.data.model_data import ModelData
from egret.data.ptdf_utils import VirtualPTDFMatrix
from egret.model_library.defn import BasePointType
from egret.models.dcopf import _lazy_ptdf_dcopf_model_solve_loop
from egret.models.scopf import create_scopf_model

from model.relief import (add_lostload_generators_ordc, add_queue_generators,
                          scale_loads_by_zone, scale_renewables_by_snapshot,
                          set_emergency_factor)
from model.soft_limits import enable_soft_limits

# Post-contingency overflow (per unit) above which a dropped contingency counts as
# violated at the screened optimum -- about 0.01 MW on a 100 MVA base.
AUDIT_TOL = 1e-4


def extract_mu(m, base_mva: float) -> dict[str, float]:
    """{constraint_id: mu} for every binding base and post-contingency thermal limit."""
    mus = {}
    for bn, con in m.ineq_pf_branch_thermal_bounds.items():
        mu = abs(m.dual.get(con, 0.0) or 0.0) / base_mva
        if mu > 1e-6:
            mus[f"base:{bn}"] = mu
    for (cn, bn), con in m.ineq_pf_contingency_branch_thermal_bounds.items():
        mu = abs(m.dual.get(con, 0.0) or 0.0) / base_mva
        if mu > 1e-6:
            mus[f"ctg:{cn}:{bn}"] = mu
    return mus


class PersistentModel:
    """A built SCDCOPF plus its persistent solver, reusable across configurations."""

    def __init__(self, m, md_scaled, solver, cap_pu, base_mva, screened):
        self.m = m
        self.md = md_scaled
        self.solver = solver
        self.cap_pu = cap_pu                 # {generator name: pu upper bound when built}
        self.names = list(cap_pu)
        self.base_mva = base_mva
        self.screened = screened             # contingency names actually in the model
        self._audit_ptdf = None

    def solve(self, config) -> dict[str, float]:
        """Apply a configuration via candidate bounds, warm re-solve, return mu."""
        for on, name in zip(config, self.names):
            self.m.pg[name].setub(self.cap_pu[name] if on else 0.0)
            self.solver.update_var(self.m.pg[name])
        _solve_model(self.m, self.solver, solver_tee=False, return_solver=True,
                     set_instance=False)
        _lazy_ptdf_dcopf_model_solve_loop(self.m, self.md, self.solver, solver_tee=False)
        return extract_mu(self.m, self.base_mva)

    def load_shed(self) -> float:
        """System-wide fallback shed at the current solution (MW)."""
        return float(pyo.value(self.m.p_load_shed)) * self.base_mva

    def audit_dropped(self, full_case) -> list[tuple]:
        """Contingencies dropped by screening that are violated at the current dispatch.

        An empty result proves the screened solution is also the full problem's solution:
        adding already-satisfied constraints cannot change the optimum.  Each entry is
        (contingency, worst branch, |post-outage flow| pu, emergency limit pu).
        """
        if self._audit_ptdf is None:
            self._audit_ptdf = build_audit_ptdf(self.md, full_case)
        ptdf = self._audit_ptdf
        PFV, _, VA = ptdf.calculate_PFV(self.m)
        limits, bkeys = ptdf.contingency_limits_array, ptdf.branches_keys
        violated = []
        for cn in ptdf.contingencies:
            if cn in self.screened:
                continue
            post = PFV + ptdf.calculate_PFV_delta(cn, PFV, VA)
            over = np.abs(post) - limits
            j = int(np.argmax(over))
            if over[j] > AUDIT_TOL:
                violated.append((cn, bkeys[j], float(abs(post[j])), float(limits[j])))
        return violated


def build_audit_ptdf(md, full_case) -> VirtualPTDFMatrix:
    """Full-contingency PTDF on `md`'s topology, whatever contingencies `md` carries.

    Reuses md's branches and buses -- so passing a solve model's scaled ModelData gives
    identical ratings and ordering -- but attaches the *complete* contingency set, so
    post-outage flows can be computed for the contingencies a screened case omits.
    PTDF is pure topology, so an unscaled case read straight from JSON works too when
    only the shift factors are wanted.
    """
    branches = dict(md.elements(element_type="branch"))
    buses = dict(md.elements(element_type="bus"))
    ref = md.data["system"]["reference_bus"]
    full = json.loads(Path(full_case).read_text())["elements"]["contingency"]
    return VirtualPTDFMatrix(branches, buses, ref, BasePointType.FLATSTART,
                             lpu.populate_default_ptdf_options(None),
                             branches_keys=tuple(branches), buses_keys=tuple(buses),
                             contingencies=full)


USE_CASE_RELIEF = object()      # sentinel: take the relief bus set off the case record


def build_persistent(case, snapshot=None, screened=True, threads=0,
                     relief_buses=USE_CASE_RELIEF) -> PersistentModel:
    """Build the case's SCDCOPF once with every candidate present, ready for warm re-solves.

    `snapshot` is one record from the case's snapshot collection (zone load scales and
    per-fuel capacity factors) or None for a single-load-level case.  A built candidate's
    online bound is its capacity derated by the snapshot's capacity factor for its fuel,
    so solar and wind availability tracks the hour.

    `relief_buses` overrides the case's validated relief set -- None means every load bus.
    Only `cases.texas.select_relief_set`, which exists to choose that set, should pass it.
    """
    recipe = case.recipe
    cf = (snapshot or {}).get("renewable_cf") or {}
    candidates = case.candidates()
    if relief_buses is USE_CASE_RELIEF:
        relief_buses = case.relief_buses()

    md = ModelData.read(str(case.solve_file(screened)))
    set_emergency_factor(md, recipe.emergency_factor)
    add_queue_generators(md, candidates, [1] * len(candidates))
    scale_loads_by_zone(md, (snapshot or {}).get("zone_scales"))
    scale_renewables_by_snapshot(md, cf or None)
    add_lostload_generators_ordc(md, recipe.ordc_min, recipe.ordc_max, recipe.ordc_blocks,
                                 buses=relief_buses)

    m, md_scaled = create_scopf_model(md, include_feasibility_slack=True)
    base_mva = md.data["system"]["baseMVA"]
    enable_soft_limits(m, recipe.shadow_price_cap, base_mva)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    solver = pyo.SolverFactory("gurobi_persistent")
    solver.options["OutputFlag"] = 0
    if threads:
        solver.options["Threads"] = threads
    solver.set_instance(m)

    cap_pu = {f"queue_{c['id']}": cf.get(c["fuel"], 1.0) * c["capacity_mw"] / base_mva
              for c in candidates}
    return PersistentModel(m, md_scaled, solver, cap_pu, base_mva,
                           set(md.data["elements"]["contingency"]))


def scalar_target(mus: dict[str, float]) -> float:
    """The sampling target: max mu over constraints (0 when nothing binds)."""
    return max(mus.values(), default=0.0)
