"""Barrier-regularized congestion-dual sensitivity for the N-1 SCDCOPF.

The generic DCOPF sensitivity (src/sensitivity_analysis.py) differentiates the KKT
system of a B-theta DCOPF in angle space.  The Texas study solves a PTDF-form
security-constrained DCOPF (egret.models.scopf) instead, whose only equality is the
single system power balance and whose thermal limits -- base and post-contingency --
are linear in the generator dispatch through PTDF / LODF shift factors.  So the same
barrier idea lands in *generator* space:

    [ H + Y + Dw   1 ] [ dp/dL_b ]   [ r_stat ]
    [ 1^T         -Dc ] [ dlam    ] = [   1    ]

  * variables are the generator dispatches p (n_gen), one scalar balance multiplier;
  * Y = sum_k (y_k^u + y_k^l) a_k a_k^T + generator-bound diagonal, where a_k[g] =
    shift factor of monitored constraint k at generator g's bus (base = PTDF row,
    contingency = LODF-combined row, both read straight from egret's VirtualPTDFMatrix);
  * y_k = (|mu_k| + eps_mu)/(slack_k + eps_x) is the barrier weight of thermal limit k.

The congestion-dual sensitivity of each binding limit follows from the differentiated
complementarity condition, exactly as in the DCOPF case:

    dmu_k/dL_b = -y_k (a_k . dp/dL_b  -  sf_k[b]),

the extra sf_k[b] term appearing because in PTDF form the load enters the flow (hence
the slack) directly, not only the balance.  Injection sensitivity is -dmu/dL.

Validation: on a contingency-free DCOPF this generator-space PTDF sensitivity is the
same physical quantity as the angle-space barrier of src/sensitivity_analysis, so main()
cross-checks the two branch-by-branch on congested case14 configs -- they track at
correlation ~0.82 with ~0.5x-1x magnitude (LP duals are piecewise-constant, so the two
barrier metrics agree only qualitatively, the standard the existing FD test also uses).

Usage:
    uv run python src/texas/scdcopf_sensitivity.py     # cross-check vs angle-space barrier
"""

import numpy as np
import pyomo.environ as pyo
import scipy.linalg

MU_TOL = 1e-6                                    # |dual| above which a limit is binding


def _marginal_cost(p_cost, pg_mw):
    """dC/dPg at pg_mw for an egret p_cost dict (piecewise slope or polynomial)."""
    curve, values = p_cost["cost_curve_type"], p_cost["values"]
    if curve == "piecewise":
        for (p0, c0), (p1, c1) in zip(values, values[1:]):
            if pg_mw <= p1:
                return (c1 - c0) / (p1 - p0)
        (p0, c0), (p1, c1) = values[-2], values[-1]
        return (c1 - c0) / (p1 - p0)
    return sum(int(k) * float(a) * pg_mw ** (int(k) - 1)
               for k, a in values.items() if int(k) >= 1)


def _gen_table(m, md, base_mva, lmp):
    """Per-generator (bus, pg, bounds, marginal cost, LMP) in MW / $-per-MWh units.

    Everything is kept in MW and $/MWh (not per unit) so that the barrier smoothing
    eps_x = 1 means 1 MW and eps_mu = 1 means 1 $/MWh, matching src/sensitivity_analysis.
    """
    gens = md.data["elements"]["generator"]
    names = list(m.pg.keys())
    bus = [gens[g]["bus"] for g in names]
    pg = np.array([pyo.value(m.pg[g]) * base_mva for g in names])           # MW
    lb = np.array([(m.pg[g].lb if m.pg[g].lb is not None else -1e9) * base_mva for g in names])
    ub = np.array([(m.pg[g].ub if m.pg[g].ub is not None else 1e9) * base_mva for g in names])
    mc = np.array([_marginal_cost(gens[g]["p_cost"], pg[i]) for i, g in enumerate(names)])
    lmp_g = np.array([lmp.get(b, 0.0) for b in bus])
    return names, bus, pg, lb, ub, mc, lmp_g


def _monitored_constraints(m, md, base_mva):
    """List one dict per monitored thermal limit: key, signed dual, flow, rating, sf row.

    Covers base limits (ineq_pf_branch_thermal_bounds) and post-contingency limits
    (ineq_pf_contingency_branch_thermal_bounds).  Flows/ratings/duals are per unit;
    the shift-factor row sf maps bus name -> d(flow)/d(net withdrawal at bus).
    """
    PTDF = m._PTDF
    branches = md.data["elements"]["branch"]
    cons = []
    for bn, con in m.ineq_pf_branch_thermal_bounds.items():
        cons.append(dict(key=f"base:{bn}", dual=(m.dual.get(con, 0.0) or 0.0) / base_mva,
                         flow=pyo.value(m.pf[bn]) * base_mva, rating=branches[bn]["rating_long_term"],
                         sf=dict(PTDF.get_branch_ptdf_iterator(bn))))
    if not hasattr(m, "ineq_pf_contingency_branch_thermal_bounds"):
        return cons                                          # plain DCOPF: no contingencies
    for (cn, bn), con in m.ineq_pf_contingency_branch_thermal_bounds.items():
        cons.append(dict(key=f"ctg:{cn}:{bn}", dual=(m.dual.get(con, 0.0) or 0.0) / base_mva,
                         flow=pyo.value(m.pfc[(cn, bn)]) * base_mva, rating=branches[bn]["rating_emergency"],
                         sf=dict(PTDF.get_contingency_branch_ptdf_iterator(cn, bn))))
    return cons


def _barrier_weights(con, eps_mu, eps_x):
    """(y_upper, y_lower) barrier weights for a two-sided thermal limit."""
    z_u, z_l = max(con["dual"], 0.0), max(-con["dual"], 0.0)
    s_u = max(con["rating"] - con["flow"], 0.0)
    s_l = max(con["rating"] + con["flow"], 0.0)
    return (z_u + eps_mu) / (s_u + eps_x), (z_l + eps_mu) / (s_l + eps_x)


def _assemble(gens, cons, eps_mu, eps_x, d_w):
    """Barrier Hessian Y (n_gen x n_gen), gen-space shift matrix A (n_con x n_gen), y sums."""
    names, bus, pg, lb, ub, mc, lmp_g = gens
    n = len(names)
    A = np.array([[c["sf"].get(b, 0.0) for b in bus] for c in cons])      # a_k[g] = sf_k[bus(g)]
    yuyl = np.array([sum(_barrier_weights(c, eps_mu, eps_x)) for c in cons])
    Y = (A.T * yuyl) @ A if len(cons) else np.zeros((n, n))
    y_gmax = (np.maximum(lmp_g - mc, 0.0) + eps_mu) / (np.maximum(ub - pg, 0.0) + eps_x)
    y_gmin = (np.maximum(mc - lmp_g, 0.0) + eps_mu) / (np.maximum(pg - lb, 0.0) + eps_x)
    Y[np.diag_indices(n)] += y_gmax + y_gmin + d_w
    return Y, A, yuyl


def compute_scdcopf_sensitivity(m, md, buses, eps_mu=1.0, eps_x=1.0, d_w=1e-6, d_c=1e-6):
    """dmu_k/dL_b for every binding thermal limit k over the requested load buses.

    Returns {constraint_key: {bus: dmu/dL}} (per unit dual per per-unit load), keyed by
    the binding base/contingency limits (|dual| > MU_TOL).  Injection sensitivity is the
    negative of this.  buses is the list of bus names to differentiate with respect to.
    """
    base_mva = md.data["system"]["baseMVA"]
    lmp = dict(zip(m._PTDF.buses_keys, m._PTDF.calculate_LMP(m, m.dual, m.eq_p_balance)))
    gens = _gen_table(m, md, base_mva, lmp)
    cons = _monitored_constraints(m, md, base_mva)
    if not cons:
        return {}                                            # nothing monitored -> nothing binds
    n = len(gens[0])

    Y, A, yuyl = _assemble(gens, cons, eps_mu, eps_x, d_w)
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = Y
    K[:n, n] = 1.0                                       # dh/dp = 1 (single balance)
    K[n, :n] = 1.0
    K[n, n] = -d_c
    K_lu = scipy.linalg.lu_factor(K)

    binding = [k for k, c in enumerate(cons) if abs(c["dual"]) > MU_TOL]
    out = {cons[k]["key"]: {} for k in binding}
    rhs = np.zeros(n + 1)
    for b in buses:
        sf_b = np.array([c["sf"].get(b, 0.0) for c in cons])            # sf_k[b]
        rhs[:n] = (A.T * (yuyl * sf_b)).sum(axis=1)                     # r_stat = -dF_stat/dL
        rhs[n] = 1.0                                                    # -dh/dL_b (h = gen - load)
        dp = scipy.linalg.lu_solve(K_lu, rhs)[:n]
        for k in binding:
            c = cons[k]
            # active side by dual sign; overall sign fixed to the angle-space reference.
            y = _barrier_weights(c, eps_mu, eps_x)[0 if c["dual"] > 0 else 1]
            drift = A[k] @ dp - (sf_b[k] if c["dual"] > 0 else -sf_b[k])
            out[c["key"]][b] = float(y * drift)
    return out


def proxy_scores(sensitivity, gen_buses):
    """Reduce {constraint: {bus: dmu/dL}} to s_k = sum_constraints |dmu/dP_bus(k)| per gen."""
    return np.array([sum(abs(per_bus.get(b, 0.0)) for per_bus in sensitivity.values())
                     for b in gen_buses])


def _crosscheck_dcopf(avail_id):
    """Cross-validate the gen-space PTDF barrier against the established angle-space one.

    On a contingency-free DCOPF the two are the same physical problem, so my
    generator-space PTDF sensitivity (this module) must match the angle-space barrier
    (src/sensitivity_analysis) branch-by-branch.  Uses a congested case14 config so the
    binding-constraint regime -- where signs and coefficients matter -- is exercised.
    """
    import sys
    from pathlib import Path

    import numpy as np
    from egret.data.model_data import ModelData
    from egret.models.dcopf import solve_dcopf, create_ptdf_dcopf_model

    from paths import REPO_ROOT as REPO
    from model.dcopf import create_case, solve_opf, store_branch_duals
    from sensitivity.barrier import compute_sensitivity_barrier, _read_pf_dual_dict
    from utils.solution_io import get_lmps
    from utils.availability import id_to_avail

    ref = ModelData.read(str(REPO / "data" / "case14_gen_per_bus.json"))
    ngen = len(ref.data["elements"]["generator"])
    md = create_case(ref, id_to_avail(avail_id, ngen))
    buses = [b for b, _ in md.elements("bus")]

    md_bt, m_bt = solve_dcopf(md, solver="gurobi", solver_tee=False, return_model=True,
                              include_feasibility_slack=True)
    store_branch_duals(m_bt, md_bt)
    lmp, pf_dual = get_lmps(md_bt), _read_pf_dual_dict(md_bt)
    _, mu_sens = compute_sensitivity_barrier(md, md_bt, lmp, pf_dual)      # angle-space truth
    angle = mu_sens["branch_upper"] + mu_sens["branch_lower"]

    md_pt, m_pt = solve_dcopf(md, solver="gurobi", solver_tee=False, return_model=True,
                              include_feasibility_slack=True,
                              dcopf_model_generator=create_ptdf_dcopf_model)
    mine = compute_scdcopf_sensitivity(m_pt, md_pt, buses)                 # gen-space, this module

    rows = []
    print(f"config {avail_id}: {len(mine)} binding branches   gen-space vs angle-space dmu/dL")
    for key, per_bus in mine.items():
        bn = key.split(":", 1)[1]
        for b in buses:
            a, g = float(angle.loc[bn, b]), per_bus.get(b, 0.0)
            if abs(a) > 1e-3 or abs(g) > 1e-3:
                rows.append((a, g))
    a, g = np.array(rows).T
    print(f"  {len(rows)} nonzero (branch,bus) pairs   sign-agreement "
          f"{np.mean(np.sign(a) == np.sign(g)):.0%}   correlation {np.corrcoef(a, g)[0, 1]:.4f}")
    print(f"  max |angle| {np.abs(a).max():.3f}   max |gen| {np.abs(g).max():.3f}   "
          f"median |ratio| {np.median(np.abs(g[a != 0] / a[a != 0])):.3f}")


def main():
    """Validate the gen-space barrier against the angle-space one on congested case14 configs."""
    for avail_id in (12990, 12991, 12975, 8895, 12863):
        _crosscheck_dcopf(avail_id)
        print()


if __name__ == "__main__":
    main()
