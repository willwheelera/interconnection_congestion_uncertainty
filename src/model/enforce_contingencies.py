"""Enforce the contingency limits EGRET's lazy loop skipped.

EGRET computes post-outage flows only in an iteration whose base-case check found no
violations (`lazy_ptdf_utils.check_violations`, the `total_violations == 0` gate).  Under
this recipe a base branch held by its soft-limit slack is over its rating at *every*
optimum, so that gate can close permanently and the loop exits having never looked at a
contingency.  Measured on case118_queue: a first solve with one base slack active returned
a dispatch violating 172 contingency limits, the worst at 381% of its emergency rating,
with objective 1954703 and sum_mu 115155 against 2124291 and 142597 for the same
configuration once those constraints were in the model.

This closes the gate.  After the loop returns, post-outage flows are computed straight
from the solved angles -- no precondition -- and any contingency limit over its rating
with no constraint holding it is added and the model re-solved, until the check comes back
clean.  It also covers the other way the loop can stop early: EGRET's violation store
keeps only the `max_violations_per_iteration` largest violations and drops smaller ones
from the count as well as from the additions, and this check has no such cap.

An empty check is the proof `scdcopf.PersistentModel.audit_dropped` already relies on:
constraints satisfied at an optimum cannot move it, so the solution solves the problem
carrying all of them.  The two checks are complements -- this one covers the contingencies
the model carries, that one the contingencies screening removed -- so between them every
contingency in the case is verified at the dispatch that gets returned.
"""

import egret.common.lazy_ptdf_utils as lpu
from egret.common.solver_interface import _solve_model
from egret.models.dcopf import _lazy_ptdf_dcopf_model_solve_loop

from src.model.dropped_flows import DroppedFlowCheck

# Add-and-re-solve rounds allowed before giving up.  Each round adds at least one
# constraint that was missing, so this is a bound on how far the dispatch can chase its
# own violations, not an iteration count anything is expected to reach.
MAX_PASSES = 20


class ContingencyEnforcer:
    """The model's own contingency set, checked and enforced after every solve."""

    def __init__(self, m, md):
        self.md = md
        self.check = DroppedFlowCheck(m._PTDF, set(m._PTDF.contingencies))
        self.index = m._PTDF.branchname_to_index_masked_map

    def enforce(self, m, solver) -> int:
        """Add every unenforced violated contingency limit, re-solving until none is left.

        Returns the number of rounds that added constraints, 0 when the lazy loop had
        already enforced everything.
        """
        for npass in range(MAX_PASSES):
            monitored = set(m._contingencies_monitored)
            viol = {(cn, self.index[bn]) for cn, bn, _flow, _limit in self.check.violations(m)}
            viol -= monitored
            if not viol:
                return npass
            lpu.add_violations(lpu._LazyViolations(set(), set(), viol),
                               lpu._CalculatedFlows(), m, self.md, solver,
                               m._ptdf_options, m._PTDF)
            _solve_model(m, solver, solver_tee=False, return_solver=True, set_instance=False)
            _lazy_ptdf_dcopf_model_solve_loop(m, self.md, solver, solver_tee=False)
        raise RuntimeError(f"contingency limits still violated after {MAX_PASSES} rounds of "
                           f"adding them; last round had {len(viol)}")
