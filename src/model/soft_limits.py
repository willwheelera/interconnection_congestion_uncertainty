"""Soft transmission limits: cap the congestion shadow price like ERCOT's Max Shadow Price.

ERCOT enforces a Max Shadow Price on transmission constraints in SCED: rather than pay
unbounded redispatch to hold a limit, the constraint may overload at a fixed penalty, so
no constraint's shadow price exceeds the cap. This attaches that mechanism to an EGRET
scopf model via a per-branch and per-contingency violation slack penalized at `cap`
($/MWh), bounding every thermal dual mu by `cap` while the primal reports the (small)
overload it buys.

Reuses EGRET's own lazy-PTDF slack hooks (pf_slack_pos/neg, pfc_slack_pos/neg): the vars
are declared sparse over the full branch / contingency index so each lazily added violated
constraint instantiates its slack and books the penalty through the persistent solver's
add_column. The single-period model is shimmed with the attributes EGRET's slack path
expects (TimePeriodLengthHours, BranchLimitPenalty, SystemContingencyLimitPenalty).

Objective coefficient is cap * baseMVA: EGRET scales generator $/MWh costs by baseMVA in
the objective and reports duals as raw/baseMVA, so a slack penalized at cap*baseMVA per pu
caps the reported mu at exactly `cap` $/MWh.
"""

import pyomo.environ as pyo


def enable_soft_limits(m, cap: float, base_mva: float) -> None:
    """Attach a violation slack to every thermal limit, penalized at `cap` $/MWh."""
    for nm in ("pf_slack_pos", "pf_slack_neg", "pfc_slack_pos", "pfc_slack_neg"):
        if m.component(nm) is not None:
            m.del_component(nm)
    m.pf_slack_pos = pyo.Var(m._branches, dense=False, within=pyo.NonNegativeReals)
    m.pf_slack_neg = pyo.Var(m._branches, dense=False, within=pyo.NonNegativeReals)
    m.pfc_slack_pos = pyo.Var(m._contingencies * m._branches, dense=False, within=pyo.NonNegativeReals)
    m.pfc_slack_neg = pyo.Var(m._contingencies * m._branches, dense=False, within=pyo.NonNegativeReals)
    m.TimePeriodLengthHours = 1.0
    m.BranchLimitPenalty = {bn: cap * base_mva for bn in m._branches}
    m.SystemContingencyLimitPenalty = cap * base_mva


def thermal_violations(m, base_mva: float) -> dict[str, float]:
    """{constraint_id: overload_MW} for thermal limits whose violation slack is active."""
    viol = {}
    for bn in m.pf_slack_pos:
        mw = ((m.pf_slack_pos[bn].value or 0.0) + (m.pf_slack_neg[bn].value or 0.0)) * base_mva
        if mw > 1e-3:
            viol[f"base:{bn}"] = mw
    for key in m.pfc_slack_pos:
        mw = ((m.pfc_slack_pos[key].value or 0.0) + (m.pfc_slack_neg[key].value or 0.0)) * base_mva
        if mw > 1e-3:
            viol[f"ctg:{key[0]}:{key[1]}"] = mw
    return viol
