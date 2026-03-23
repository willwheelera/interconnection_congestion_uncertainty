"""
Barrier-regularised LMP and congestion-dual sensitivity (LP and QP).

Handles both LP (linear costs) and QP (quadratic costs) by replacing the
singular bordered-Hessian of the LP case with a regularised saddle-point
system::

    [ H + Y + D_w    J^T  ] [ dx/dP_j ]   [  0  ]
    [ J             -D_c  ] [ dν/dP_j ] = [ e_j ]

where Y is a barrier matrix built from all inequality constraints.  The primary
outputs are the congestion dual sensitivities dμ/dP (sensitivity of line shadow
prices to bus power injections) and the LMP sensitivities dLMP/dP as a secondary
output.

For the mathematical derivation see docs/narrative/kkt_sensitivity.rst.

Saves results to results/sensitivity_barrier_<case>.json.

Usage:
    uv run python code/sensitivity_analysis.py
    uv run python code/sensitivity_analysis.py --solution path/to/solution.json
    uv run python code/sensitivity_analysis.py --solution path/to/solution.json --hour 12
"""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy.linalg

from egret.data.model_data import ModelData
from egret.models.dcopf import solve_dcopf

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_SOLUTION = REPO_ROOT / "results" / "uc_test_scuc_masked.json"


def _get_lmps(md_sol) -> dict[str, float]:
    """Extract {bus_name: lmp} from a solved ModelData."""
    return {
        name: bus_data["lmp"]
        for name, bus_data in md_sol.elements(element_type="bus")
        if "lmp" in bus_data
    }


def _extract_hour(md: ModelData, hour: int) -> ModelData:
    """Return a new ModelData with all time-series values replaced by scalars at *hour*.

    UC solution files store most quantities as ``{"data_type": "time_series",
    "values": [...]}`` dicts.  This helper recursively flattens them so that
    downstream functions see plain scalar values.
    """
    def _resolve(obj):
        if isinstance(obj, dict):
            if obj.get("data_type") == "time_series":
                return obj["values"][hour]
            return {k: _resolve(v) for k, v in obj.items()}
        return obj

    return ModelData(_resolve(md.data))


def _network_state(md, md_sol):
    """Extract and index network elements from a pair of ModelData objects.

    Returns ``(orig, sol, names, idx, n, n_vars, ref_bus, base_mva)`` where:

    * **orig** — ``{"bus", "gen", "branch", "dc"}`` dicts of element data.
    * **sol** — ``{"gen", "branch"}`` dicts from the solved model.
    * **names** — ``{key: sorted_list}`` for each element type in *orig*.
    * **idx** — ``{key: {name: int}}`` position maps.
    * **n** — ``{key: count}`` element counts.
    * **n_vars** — total primal variables (θ, p_g, dcpf).
    * **ref_bus** — reference bus name string.
    * **base_mva** — system base MVA.

    The primal variable ordering is ``[θ (bus), p_g (gen), dcpf (dc)]``.
    To obtain the column index for a variable use::

        idx["bus"][b]                           # θ_b
        n["bus"] + idx["gen"][g]                # p_g
        n["bus"] + n["gen"] + idx["dc"][d]      # dcpf_d
    """
    orig = {
        "bus": dict(md.elements("bus")),
        "gen": dict(md.elements("generator")),
        "branch": dict(md.elements("branch")),
        "dc": dict(md.elements("dc_branch")),
    }
    sol = {
        "gen": dict(md_sol.elements("generator")),
        "branch": dict(md_sol.elements("branch")),
        "dc": dict(md_sol.elements("dc_branch")),
    }
    names = {k: sorted(v) for k, v in orig.items()}
    idx = {k: {name: i for i, name in enumerate(names[k])} for k in names}
    n = {k: len(names[k]) for k in names}
    n_vars = n["bus"] + n["gen"] + n["dc"]
    ref_bus = md.data["system"]["reference_bus"]
    base_mva = md.data["system"].get("baseMVA", 100.0)
    return orig, sol, names, idx, n, n_vars, ref_bus, base_mva


def _branch_susceptance(br_data, base_mva: float) -> float:
    """Return B_l [MW/rad] for a branch (accounts for transformer tap ratio)."""
    tau = (
        br_data.get("transformer_tap_ratio", 1.0)
        if br_data.get("branch_type") == "transformer"
        else 1.0
    )
    return -base_mva / (tau * br_data["reactance"])


def _poly_cost_coeffs(g_data) -> tuple[float, float]:
    """Return (a2, b1) polynomial cost coefficients for a generator."""
    cost = g_data.get("p_cost", {})
    vals = cost.get("values", {}) if cost.get("cost_curve_type") == "polynomial" else {}
    a2 = vals.get("2", vals.get(2, 0.0)) or 0.0
    b1 = vals.get("1", vals.get(1, 0.0)) or 0.0
    return a2, b1


def _marginal_cost(g_data, pg_mw: float) -> float:
    """Return marginal cost [$/MWh] at operating point pg_mw for a generator."""
    a2, b1 = _poly_cost_coeffs(g_data)
    return 2.0 * a2 * pg_mw + b1


def _build_power_balance_jacobian(orig, idx, n, n_vars, base_mva):
    """Build the n_bus × n_vars power-balance Jacobian J_pb."""
    J_pb = np.zeros((n["bus"], n_vars))
    for br_data in orig["branch"].values():
        f_i = idx["bus"][br_data["from_bus"]]
        t_i = idx["bus"][br_data["to_bus"]]
        B_l = _branch_susceptance(br_data, base_mva)
        J_pb[f_i, f_i] += -B_l
        J_pb[f_i, t_i] += B_l
        J_pb[t_i, t_i] += -B_l
        J_pb[t_i, f_i] += B_l
    for g_name, g_data in orig["gen"].items():
        J_pb[idx["bus"][g_data["bus"]], n["bus"] + idx["gen"][g_name]] = 1.0
    for dc_name, dc_data in orig["dc"].items():
        d_col = n["bus"] + n["gen"] + idx["dc"][dc_name]
        J_pb[idx["bus"][dc_data["to_bus"]], d_col] = +1.0
        J_pb[idx["bus"][dc_data["from_bus"]], d_col] = -1.0
    return J_pb


def _pf_ylim(sol_k, name, data, eps, pf_dual_dict):
    """Return (y_upper, y_lower) barrier weights for a branch's thermal limits.

    Dual estimates z_k are taken from pf_dual_dict (direct LP solver duals),
    which correctly handle LP degeneracy (equal LMPs on both sides of a
    congested line).
    """
    val = sol_k[name].get("pf", 0.0)
    rating = data["rating_long_term"]
    z_upper, z_lower = pf_dual_dict[name]
    y_upper = (z_upper + eps) / (max(rating - val, 0.0) + eps)
    y_lower = (z_lower + eps) / (max(val + rating, 0.0) + eps)
    return y_upper, y_lower

def _gen_ylim(sol, lmp_base, g_name, g_data, eps):
    pg_mw = sol["gen"][g_name].get("pg", 0.0)
    lmp_b = lmp_base.get(g_data["bus"], 0.0)
    mc = _marginal_cost(g_data, pg_mw)
    pg_max = g_data.get("p_max")
    pg_min = g_data.get("p_min", 0.0)
    y_max = 0
    if pg_max is not None:
        y_max = (max(lmp_b - mc, 0.0) + eps) / (max(pg_max - pg_mw, 0.0) + eps)
    y_min = (max(mc - lmp_b, 0.0) + eps) / (max(pg_mw - pg_min, 0.0) + eps)
    return y_max, y_min


def _compute_mu_base(
    md: ModelData,
    lmp_base: dict[str, float],
    pf_dual_dict: dict[str, tuple[float, float]],
) -> dict[str, dict[str, float]]:
    """Compute base dual variable values z_k at the optimal solution from KKT conditions.

    Branch/DC duals are taken directly from pf_dual_dict (LP solver duals).
    Generator duals:  z_max = max(LMP_bus − mc, 0),  z_min = max(mc − LMP_bus, 0).
    """
    branch_names = [n for n, _ in md.elements("branch")]
    dc_names = [n for n, _ in md.elements("dc_branch")]

    gen_max, gen_min = {}, {}
    for g_name, g_data in md.elements("generator"):
        pg_mw = g_data.get("pg", 0.0)
        lmp_b = lmp_base.get(g_data["bus"], 0.0)
        mc = _marginal_cost(g_data, pg_mw)
        gen_max[g_name] = max(lmp_b - mc, 0.0)
        gen_min[g_name] = max(mc - lmp_b, 0.0)

    return {
        "branch_upper": {bn: pf_dual_dict[bn][0] for bn in branch_names},
        "branch_lower": {bn: pf_dual_dict[bn][1] for bn in branch_names},
        "gen_max": gen_max,
        "gen_min": gen_min,
        "dc_max": {dn: pf_dual_dict[dn][0] for dn in dc_names},
        "dc_min": {dn: pf_dual_dict[dn][1] for dn in dc_names},
    }


def _build_barrier_Y(
    orig: dict,
    sol: dict,
    idx: dict,
    n: dict,
    n_vars: int,
    base_mva: float,
    lmp_base: dict[str, float],
    eps: float,
    pf_dual_dict: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Build the barrier regularisation matrix Y in [θ, p_g, dcpf] variable space.

    Each inequality constraint contributes a rank-1 (or diagonal) block to Y via::

        Y += y_k * (∂s_k/∂x)^T (∂s_k/∂x)

    where s_k is the constraint slack and y_k = (z_k + ε) / (s_k + ε).  Dual values
    z_k are recovered from the KKT reduced cost of each pf variable (see _pf_ylim).

    * Generator capacity: z_max = max(0, LMP_bus − mc_g), z_min = max(0, mc_g − LMP_bus)
    * Branch thermal:     z_k from rc = (LMP_from − LMP_to) − ν  (ν from nu_dict)
    * DC branch:          ν = 0 (no flow-equality constraint)

    The branch thermal contribution is a 2×2 outer product at the (θ_from, θ_to) block
    because pf_l = B_l (θ_from − θ_to) and ∂pf_l/∂θ = B_l (e_from − e_to).
    """
    Y = np.zeros((n_vars, n_vars))

    for g_name, g_data in orig["gen"].items():
        y_max, y_min = _gen_ylim(sol, lmp_base, g_name, g_data, eps)
        g_col = n["bus"] + idx["gen"][g_name]
        Y[g_col, g_col] += y_max + y_min

    for br_name, br_data in orig["branch"].items():
        if br_data.get("rating_long_term") is None:
            continue
        f_i = idx["bus"][br_data["from_bus"]]
        t_i = idx["bus"][br_data["to_bus"]]
        B_l = _branch_susceptance(br_data, base_mva)
        y_upper, y_lower = _pf_ylim(sol["branch"], br_name, br_data, eps, pf_dual_dict)
        y_total = y_upper + y_lower
        # grad pf = B_l * (e_from - e_to); outer product scaled by y_total
        Y[f_i, f_i] += y_total * B_l ** 2
        Y[f_i, t_i] -= y_total * B_l ** 2
        Y[t_i, f_i] -= y_total * B_l ** 2
        Y[t_i, t_i] += y_total * B_l ** 2

    for dc_name, dc_data in orig["dc"].items():
        if dc_data.get("rating_long_term") is None:
            continue
        y_upper, y_lower = _pf_ylim(sol["dc"], dc_name, dc_data, eps, pf_dual_dict)
        d_col = n["bus"] + n["gen"] + idx["dc"][dc_name]
        Y[d_col, d_col] += y_upper + y_lower

    return Y


def _compute_mu_sensitivity(
    orig: dict,
    sol: dict,
    names: dict,
    idx: dict,
    n: dict,
    base_mva: float,
    lmp_base: dict[str, float],
    eps: float,
    primal_sens: np.ndarray,
    pf_dual_dict: dict[str, tuple[float, float]],
) -> dict[str, pd.DataFrame]:
    """Compute inequality-constraint dual sensitivities dμ_k/dP_j.

    From the differentiated complementarity condition d(X Z)/dp = 0::

        dz_k/dP_j = -y_k * (∂s_k/∂x) · (dx/dP_j)

    where y_k = (z_k + ε)/(s_k + ε), s_k is the slack, and dx/dP_j is the
    primal sensitivity from the barrier system (first n_vars entries of the
    solution vector).

    Returns a dict with keys ``branch_upper``, ``branch_lower``, ``gen_max``,
    ``gen_min``, ``dc_max``, ``dc_min``.  Each value is a DataFrame indexed by
    constraint name with columns for each injection bus.
    """
    col_idx = pd.Index(names["bus"], name="injection_bus (P_j)")

    branch_upper = np.zeros((len(names["branch"]), n["bus"]))
    branch_lower = np.zeros((len(names["branch"]), n["bus"]))
    for br_i, br_name in enumerate(names["branch"]):
        br_data = orig["branch"][br_name]
        if br_data.get("rating_long_term") is None:
            continue
        f_i = idx["bus"][br_data["from_bus"]]
        t_i = idx["bus"][br_data["to_bus"]]
        dtheta_diff = primal_sens[f_i, :] - primal_sens[t_i, :]

        B_l = _branch_susceptance(br_data, base_mva)
        y_upper, y_lower = _pf_ylim(sol["branch"], br_name, br_data, eps, pf_dual_dict)
        branch_upper[br_i, :] = -y_upper * B_l * dtheta_diff
        branch_lower[br_i, :] =  y_lower * B_l * dtheta_diff

    gen_max = np.zeros((len(names["gen"]), n["bus"]))
    gen_min = np.zeros((len(names["gen"]), n["bus"]))
    for g_i, g_name in enumerate(names["gen"]):
        g_data = orig["gen"][g_name]
        y_max, y_min = _gen_ylim(sol, lmp_base, g_name, g_data, eps)
        dpg = primal_sens[n["bus"] + idx["gen"][g_name], :]
        gen_max[g_i, :] = -y_max * dpg
        gen_min[g_i, :] =  y_min * dpg

    dc_max = np.zeros((len(names["dc"]), n["bus"]))
    dc_min = np.zeros((len(names["dc"]), n["bus"]))
    for d_i, dc_name in enumerate(names["dc"]):
        dc_data = orig["dc"][dc_name]
        if dc_data.get("rating_long_term") is None:
            continue
        d_col = n["bus"] + n["gen"] + idx["dc"][dc_name]
        ddcpf = primal_sens[d_col, :]
        y_upper, y_lower = _pf_ylim(sol["dc"], dc_name, dc_data, eps, pf_dual_dict)
        dc_max[d_i, :] = -y_upper * ddcpf
        dc_min[d_i, :] =  y_lower * ddcpf

    return {
        "branch_upper": pd.DataFrame(branch_upper, index=pd.Index(names["branch"], name="branch"), columns=col_idx),
        "branch_lower": pd.DataFrame(branch_lower, index=pd.Index(names["branch"], name="branch"), columns=col_idx),
        "gen_max": pd.DataFrame(gen_max, index=pd.Index(names["gen"], name="generator"), columns=col_idx),
        "gen_min": pd.DataFrame(gen_min, index=pd.Index(names["gen"], name="generator"), columns=col_idx),
        "dc_max": pd.DataFrame(dc_max, index=pd.Index(names["dc"], name="dc_branch"), columns=col_idx),
        "dc_min": pd.DataFrame(dc_min, index=pd.Index(names["dc"], name="dc_branch"), columns=col_idx),
    }


def compute_sensitivity_barrier(
    md: ModelData,
    md_sol: ModelData,
    lmp_base: dict[str, float],
    pf_dual_dict: dict[str, tuple[float, float]],
    eps: float = 1.0,
    d_w: float = 0.01,
    d_c: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute LMP and congestion-dual sensitivity via the barrier-regularised KKT system.

    Handles both LP (linear costs) and QP (quadratic costs).  The approach
    replaces the singular bordered-Hessian of the LP case with a regularised
    saddle-point system::

        [ H + Y + D_w    J^T  ] [ dx/dP_j ]   [  0  ]
        [ J             -D_c  ] [ dν/dP_j ] = [ e_j ]

    where

    * **H** — objective Hessian (diagonal: 2a_g for QP generators, 0 for LP).
    * **Y** — barrier matrix from all inequality constraints; y_k = (z_k + ε)/(s_k + ε)
      evaluated at the optimum.  Dual values z_k are derived analytically from
      KKT stationarity (generator bounds via LMP vs. marginal cost; thermal/DC
      limits via LMP differences).  The contribution of each constraint k is
      y_k · (∂s_k/∂x)^T (∂s_k/∂x) added into the upper-left block.
    * **D_w = d_w · I** — small primal regularisation that prevents singularity
      even when H = Y = 0 at some variable.
    * **J** — Jacobian of *equality* constraints only (power-balance + reference
      angle).  Active inequalities are already captured through large y_k values
      in Y and are not included in J here.
    * **D_c = d_c · I** — constraint-block regularisation (saddle-point stability).

    The parameter ε controls the smoothing between active (s ≈ 0) and inactive
    (z ≈ 0) constraints.  As ε → 0 with exact dual values the result converges
    to the hard-constraint bordered-Hessian used in kkt_sensitivity.py.

    For the mathematical derivation see :doc:`../narrative/kkt_sensitivity`.

    Parameters
    ----------
    md:
        Original (unsolved) ModelData.
    md_sol:
        Solved ModelData returned by ``solve_dcopf``.
    lmp_base:
        Dict of ``{bus_name: lmp}`` at the base solution (used to compute duals).
    eps:
        Barrier smoothing parameter [same units as slacks/duals, ≈ 1 MW / 1 $/MWh].
    d_w:
        Primal regularisation magnitude (added to H + Y diagonal).
    d_c:
        Constraint regularisation magnitude (subtracted from lower-right block).

    Returns
    -------
    (df_lmp, mu_sens)
        df_lmp: n_bus × n_bus DataFrame of dLMP_i/dP_j [$/MWh per MW].
        mu_sens: dict with keys ``branch_upper``, ``branch_lower``, ``gen_max``,
        ``gen_min``, ``dc_max``, ``dc_min``.  Each value is a DataFrame of
        dμ_k/dP_j computed via the differentiated complementarity condition.
    """
    orig, sol, names, idx, n, n_vars, ref_bus, base_mva = _network_state(md, md_sol)

    # Objective Hessian H (diagonal, non-zero only for QP generators)
    H_diag = np.zeros(n_vars)
    for g_name, g_data in orig["gen"].items():
        a2, _ = _poly_cost_coeffs(g_data)
        if a2 != 0.0:
            H_diag[n["bus"] + idx["gen"][g_name]] = 2.0 * a2

    Y = _build_barrier_Y(orig, sol, idx, n, n_vars, base_mva, lmp_base, eps, pf_dual_dict)

    J_pb = _build_power_balance_jacobian(orig, idx, n, n_vars, base_mva)
    J_ref = np.zeros((1, n_vars))
    J_ref[0, idx["bus"][ref_bus]] = 1.0
    J = np.vstack([J_pb, J_ref])
    n_eq = J.shape[0]

    K = np.zeros((n_vars + n_eq, n_vars + n_eq))
    K[:n_vars, :n_vars] = np.diag(H_diag) + Y + d_w * np.eye(n_vars)
    K[:n_vars, n_vars:] = J.T
    K[n_vars:, :n_vars] = J
    K[n_vars:, n_vars:] = -d_c * np.eye(n_eq)

    try:
        K_lu = scipy.linalg.lu_factor(K)
    except scipy.linalg.LinAlgError as exc:
        print(f"  [barrier] LU factorisation failed ({exc}).")
        raise

    lmp_sens = np.zeros((n["bus"], n["bus"]))
    primal_sens = np.zeros((n_vars, n["bus"]))
    rhs = np.zeros(n_vars + n_eq)
    for j in range(n["bus"]):
        rhs[:] = 0.0
        rhs[n_vars + j] = 1.0
        d = scipy.linalg.lu_solve(K_lu, rhs)
        primal_sens[:, j] = d[:n_vars]
        lmp_sens[:, j] = d[n_vars: n_vars + n["bus"]]

    df_lmp = pd.DataFrame(
        lmp_sens,
        index=pd.Index(names["bus"], name="observed_bus (LMP_i)"),
        columns=pd.Index(names["bus"], name="injection_bus (P_j)"),
    )
    mu_sens = _compute_mu_sensitivity(
        orig, sol, names, idx, n, base_mva, lmp_base, eps, primal_sens, pf_dual_dict
    )
    return df_lmp, mu_sens


def save_results(
    case_stem: str,
    lmp_base: dict[str, float],
    df_lmp: pd.DataFrame,
    mu_sens: dict[str, pd.DataFrame] | None = None,
    mu_base: dict[str, dict[str, float]] | None = None,
    **extra_fields,
) -> Path:
    """Save barrier sensitivity results to results/sensitivity_barrier_<case>.json.

    Parameters
    ----------
    case_stem:
        Stem of the case file name (used in output filename).
    lmp_base:
        Base LMPs at the optimal solution.
    df_lmp:
        n_bus × n_bus DataFrame of dLMP_i/dP_j.
    mu_sens:
        Optional dict from :func:`compute_sensitivity_barrier` containing
        dμ_k/dP_j DataFrames keyed by constraint type.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"sensitivity_barrier_{case_stem}.json"
    payload = {
        "case": case_stem,
        "method": "barrier",
        "bus_names": df_lmp.columns.tolist(),
        "lmp_base": lmp_base,
        "lmp_sensitivity": df_lmp.values.tolist(),
        **extra_fields,
    }
    if mu_sens:
        payload["mu_sensitivity"] = {
            key: {
                "index": mu_df.index.tolist(),
                "columns": mu_df.columns.tolist(),
                "values": mu_df.values.tolist(),
            }
            for key, mu_df in mu_sens.items()
        }
    if mu_base:
        payload["mu_base"] = mu_base
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Results saved to {out_path}")
    return out_path


def _dcopf_with_fixed_commitments(
    md: ModelData, solver: str = "gurobi"
) -> tuple[ModelData, dict[str, float], dict[str, tuple[float, float]]]:
    """Solve DCOPF with generator commitments fixed from a UC solution.

    Generators with ``commitment == 0`` are excluded by setting ``p_min = p_max = 0``.
    The resulting DCOPF solution includes LMPs as duals of the power-balance
    constraints, which are needed for the barrier sensitivity weights.

    Also extracts:
    - Flow-equality duals ν_l = m.dual[m.eq_pf_branch[l]] (nu_dict).
    - Thermal-limit duals from the LP solver (pf_dual_dict), which correctly
      recover z_upper/z_lower even under LP degeneracy (equal LMPs on both sides
      of a congested branch).  For minimization: dual(ineq_ub) ≤ 0 at binding,
      so z_upper = −dual(ineq_ub) ≥ 0; dual(ineq_lb) ≥ 0 at binding = z_lower.

    Parameters
    ----------
    md:
        ModelData extracted from a UC solution (single time step, scalar values).
    solver:
        Pyomo solver name.

    Returns
    -------
    (md_sol, nu_dict, pf_dual_dict)
        md_sol: Solved ModelData with ``lmp`` populated on each bus.
        nu_dict: {branch_name: ν_l} flow-equality duals.
        pf_dual_dict: {branch_name: (z_upper, z_lower)} direct LP thermal-limit duals.
    """
    md_dcopf = copy.deepcopy(md)
    for _, b_data in md_dcopf.elements("bus"):
        b_data.setdefault("vm", 1.0)
    for _, br_data in md_dcopf.elements("branch"):
        br_data.setdefault("resistance", 0.0)
        br_data.setdefault("charging_susceptance", 0.0)
    for _, g_data in md_dcopf.elements("generator"):
        if g_data.get("commitment", 1) == 0:
            g_data["p_min"] = 0.0
            g_data["p_max"] = 0.0
    md_sol, m = solve_dcopf(md_dcopf, solver=solver, solver_tee=False, return_model=True)
    nu_dict = {bn: m.dual.get(m.eq_pf_branch[bn], 0.0) or 0.0
               for bn in m.eq_pf_branch}
    pf_dual_dict: dict[str, tuple[float, float]] = {}
    if hasattr(m, "ineq_pf_branch_thermal_ub"):
        for bn in m.ineq_pf_branch_thermal_ub:
            z_upper = max(-(m.dual.get(m.ineq_pf_branch_thermal_ub[bn], 0.0) or 0.0), 0.0)
            z_lower = 0.0
            if hasattr(m, "ineq_pf_branch_thermal_lb") and bn in m.ineq_pf_branch_thermal_lb:
                z_lower = max(m.dual.get(m.ineq_pf_branch_thermal_lb[bn], 0.0) or 0.0, 0.0)
            pf_dual_dict[bn] = (z_upper, z_lower)
    if hasattr(m, "dcpf"):
        for bn in m.dcpf:
            rc = m.rc.get(m.dcpf[bn], 0.0) or 0.0
            pf_dual_dict[bn] = (max(-rc, 0.0), max(rc, 0.0))
    return md_sol, nu_dict, pf_dual_dict


def run(
    solution_path: Path | str = DEFAULT_SOLUTION,
    hour: int = 0,
    solver: str = "gurobi",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load a solution file, compute barrier sensitivity, and save results.

    Parameters
    ----------
    solution_path:
        Path to a saved EGRET ModelData JSON (DCOPF or UC result).
    hour:
        Time index to extract when the solution contains time-series data.
    solver:
        Pyomo solver name used for the DCOPF re-solve when LMPs are absent.

    Returns
    -------
    (df_lmp, mu_sens) as returned by :func:`compute_sensitivity_barrier`.
    """
    t0 = time.perf_counter()
    solution_path = Path(solution_path)
    print(f"Loading {solution_path.name}...")
    md_raw = ModelData.read(str(solution_path))
    md = _extract_hour(md_raw, hour)
    print(f"  {len(dict(md.elements('bus')))} buses, hour={hour}")

    print("   solving DCOPF with fixed commitments...")
    md, nu_dict, pf_dual_dict = _dcopf_with_fixed_commitments(md, solver=solver)
    lmp_base = _get_lmps(md)
    print("LMPs ($/MWh):")
    for b in sorted(lmp_base):
        print(f"  bus {b}: {lmp_base[b]:.4f}")

    print("\nComputing barrier sensitivity (LP + QP)...")
    df_lmp, mu_sens = compute_sensitivity_barrier(
        md, md, lmp_base, pf_dual_dict
    )
    mu_base = _compute_mu_base(md, lmp_base, pf_dual_dict)
    case_stem = f"{solution_path.stem}_h{hour:02d}"
    save_results(case_stem, lmp_base, df_lmp, mu_sens=mu_sens, mu_base=mu_base)
    print(f"hour {hour} solve time {time.perf_counter() - t0}")
    return df_lmp, mu_sens


def main():
    parser = argparse.ArgumentParser(
        description="Compute LMP and congestion-dual sensitivity via barrier method (LP + QP)."
    )
    parser.add_argument(
        "--solution",
        type=str,
        default=str(DEFAULT_SOLUTION),
        help="Path to a saved EGRET ModelData JSON solution file.",
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=0,
        help="Time index to extract when the solution contains time-series data (default: 0).",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="gurobi",
        help="Pyomo solver for the DCOPF re-solve (default: gurobi).",
    )
    args = parser.parse_args()
    run(solution_path=args.solution, hour=args.hour, solver=args.solver)


if __name__ == "__main__":
    #main()
    for h in range(24):
        run(hour=h)
