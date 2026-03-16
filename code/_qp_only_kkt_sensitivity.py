"""
Analytical LMP sensitivity via the KKT / bordered-Hessian system (QP only).

For a DCOPF with quadratic generator costs (QP), the sensitivity of LMPs to
power injections is derived directly from the KKT conditions at the optimal
solution — no re-solving required.

The KKT sensitivity system is:

    [ H_zz   J^T ] [ dz/dP_j ]   [  0  ]
    [ J       0  ] [ dν/dP_j ] = [ e_j ]

where H_zz is the objective Hessian (diagonal: 2*a_g for each generator with
quadratic cost a_g * p_g^2), J is the Jacobian of the full active constraint
set (power-balance equalities + reference-angle + binding thermal/capacity
limits), and e_j selects the j-th power-balance row.  The LMP sensitivities
dLMP_i/dP_j are the first n_bus entries of dν/dP_j.

For LP (all-linear costs): H_zz = 0, the KKT matrix is singular, and LMPs are
piecewise-constant within the current basis — sensitivity is identically zero.
Use sensitivity_analysis.py (barrier method) for LP cases.

Saves results to results/sensitivity_kkt_<case>.json.

Usage:
    uv run python code/kkt_sensitivity.py
    uv run python code/kkt_sensitivity.py --case path/to/case.json
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg

from egret.data.model_data import ModelData
from egret.models.dcopf import solve_dcopf
from sensitivity_analysis import _get_lmps
from sensitivity_analysis import _network_state
from sensitivity_analysis import _branch_susceptance
from sensitivity_analysis import _build_power_balance_jacobian

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_CASE = (
    REPO_ROOT
    / "Egret/egret/models/tests/transmission_test_instances/test_instances/hvdc_test_case3.json"
)


def _active_inequality_constraints(orig, sol, idx, n, n_vars, base_mva, tol):
    """Return list of gradient rows for binding inequality constraints."""
    ineq_rows = []

    for br_name, br_data in orig["branch"].items():
        rating = br_data.get("rating_long_term")
        if rating is None:
            continue
        pf_mw = sol["branch"][br_name].get("pf", 0.0)
        f_i = idx["bus"][br_data["from_bus"]]
        t_i = idx["bus"][br_data["to_bus"]]
        B_l = _branch_susceptance(br_data, base_mva)
        scale = max(abs(rating), 1.0)
        if abs(abs(pf_mw) - rating) < tol * scale:
            row = np.zeros(n_vars)
            row[f_i], row[t_i] = B_l, -B_l
            ineq_rows.append(row * np.sign(pf_mw))

    for g_name, g_data in orig["gen"].items():
        pg_mw = sol["gen"][g_name].get("pg", 0.0)
        pg_max = g_data.get("p_max")
        pg_min = g_data.get("p_min", 0.0)
        at_lim = lambda L: L is not None and abs(pg_mw - L) < tol * max(abs(L), 1.0)
        if at_lim(pg_max) or at_lim(pg_min):
            row = np.zeros(n_vars)
            row[n["bus"] + idx["gen"][g_name]] = 1.0
            ineq_rows.append(row)

    for dc_name, dc_data in orig["dc"].items():
        rating_dc = dc_data.get("rating_long_term")
        if rating_dc is None:
            continue
        dcpf_val = sol["dc"][dc_name].get("pf", 0.0)
        d_col = n["bus"] + n["gen"] + idx["dc"][dc_name]
        scale = max(abs(rating_dc), 1.0)
        if abs(abs(dcpf_val) - rating_dc) < tol * scale:
            row = np.zeros(n_vars)
            row[d_col] = np.sign(dcpf_val)
            ineq_rows.append(row)

    return ineq_rows


def compute_sensitivity_kkt(
    md: ModelData,
    md_sol: ModelData,
    tol: float = 1e-4,
) -> tuple[pd.DataFrame, bool]:
    """Compute LMP sensitivity analytically from the KKT / bordered-Hessian system.

    Builds and solves the linear system::

        [ H_zz   J^T ] [ dz/dP_j ]   [  0  ]
        [ J       0  ] [ dν/dP_j ] = [ e_j ]

    where H_zz is the second-derivative of the Lagrangian with respect to primal
    variables z = (θ, p_g, dcpf), J is the Jacobian of the active constraint set
    (equality power-balance + reference-angle + binding thermal/generator/DC limits),
    and e_j is the unit vector selecting the j-th power-balance constraint.

    The LMP sensitivities are the entries of dν/dP corresponding to the power-
    balance dual variables (first n_bus entries of ν).

    Parameters
    ----------
    md:
        Original (unsolved) ModelData — network topology and cost data.
    md_sol:
        Solved ModelData returned by solve_dcopf — primal solution values.
    tol:
        Relative tolerance for deciding a constraint is active.

    Returns
    -------
    (df_sensitivity, is_qp)
        df_sensitivity: n_bus × n_bus DataFrame of dLMP_i/dP_j [$/MWh per MW].
        is_qp: True if at least one generator has a nonzero quadratic cost
        coefficient.  If False, the returned matrix is all zeros.
    """
    orig, sol, names, idx, n, n_vars, ref_bus, base_mva = _network_state(md, md_sol)

    # Objective Hessian H (diagonal, non-zero only for QP generators)
    H_diag = np.zeros(n_vars)
    is_qp = False
    for g_name, g_data in orig["gen"].items():
        cost = g_data.get("p_cost", {})
        if cost.get("cost_curve_type") == "polynomial":
            vals = cost.get("values", {})
            a2 = vals.get("2", vals.get(2, 0.0)) or 0.0
            if a2 != 0.0:
                H_diag[n["bus"] + idx["gen"][g_name]] = 2.0 * a2
                is_qp = True

    if not is_qp:
        return pd.DataFrame(
            np.zeros((n["bus"], n["bus"])),
            index=pd.Index(names["bus"], name="observed_bus (LMP_i)"),
            columns=pd.Index(names["bus"], name="injection_bus (P_j)"),
        ), False

    J_pb = _build_power_balance_jacobian(orig, idx, n, n_vars, base_mva)
    J_ref = np.zeros((1, n_vars))
    J_ref[0, idx["bus"][ref_bus]] = 1.0

    ineq_rows = _active_inequality_constraints(orig, sol, idx, n, n_vars, base_mva, tol)

    J_blocks = [J_pb, J_ref]
    if ineq_rows:
        J_blocks.append(np.vstack(ineq_rows))
    J = np.vstack(J_blocks)
    n_active = J.shape[0]

    K = np.zeros((n_vars + n_active, n_vars + n_active))
    K[:n_vars, :n_vars] = np.diag(H_diag)
    K[:n_vars, n_vars:] = J.T
    K[n_vars:, :n_vars] = J

    try:
        K_lu = scipy.linalg.lu_factor(K)
    except scipy.linalg.LinAlgError as exc:
        print(f"  [KKT] LU factorisation failed ({exc}); returning zeros.")
        return pd.DataFrame(
            np.zeros((n["bus"], n["bus"])),
            index=pd.Index(names["bus"], name="observed_bus (LMP_i)"),
            columns=pd.Index(names["bus"], name="injection_bus (P_j)"),
        ), is_qp

    sensitivity = np.zeros((n["bus"], n["bus"]))
    rhs = np.zeros(n_vars + n_active)
    for j in range(n["bus"]):
        rhs[:] = 0.0
        rhs[n_vars + j] = 1.0
        d = scipy.linalg.lu_solve(K_lu, rhs)
        sensitivity[:, j] = d[n_vars: n_vars + n["bus"]]

    return pd.DataFrame(
        sensitivity,
        index=pd.Index(names["bus"], name="observed_bus (LMP_i)"),
        columns=pd.Index(names["bus"], name="injection_bus (P_j)"),
    ), is_qp


def save_results(
    case_stem: str,
    lmp_base: dict[str, float],
    df_lmp: pd.DataFrame,
    **extra_fields,
) -> Path:
    """Save KKT sensitivity results to results/sensitivity_kkt_<case>.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"sensitivity_kkt_{case_stem}.json"
    payload = {
        "case": case_stem,
        "method": "kkt",
        "bus_names": df_lmp.columns.tolist(),
        "lmp_base": lmp_base,
        "lmp_sensitivity": df_lmp.values.tolist(),
        **extra_fields,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Results saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute analytical LMP sensitivity via KKT bordered-Hessian (QP only)."
    )
    parser.add_argument(
        "--case",
        type=str,
        default=str(DEFAULT_CASE),
        help="Path to EGRET ModelData JSON (or MatPower .m) file.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="gurobi",
        help="Pyomo solver name (default: gurobi).",
    )
    args = parser.parse_args()

    case_path = Path(args.case)
    print(f"Loading {case_path.name}...")
    md = ModelData.read(str(case_path))
    print(f"  {len(dict(md.elements('bus')))} buses")

    print(f"\nSolving base DCOPF (solver={args.solver})...")
    md_sol = solve_dcopf(
        copy.deepcopy(md), solver=args.solver, solver_tee=False
    )
    lmp_base = _get_lmps(md_sol)
    print("Base LMPs ($/MWh):")
    for b in sorted(lmp_base):
        print(f"  bus {b}: {lmp_base[b]:.4f}")

    print("\nComputing KKT sensitivity (QP only)...")
    df_lmp, is_qp = compute_sensitivity_kkt(md, md_sol)
    if is_qp:
        print("KKT LMP sensitivity matrix ($/MWh per MW):")
        print(df_lmp.to_string())
        save_results(case_path.stem, lmp_base, df_lmp, is_qp=is_qp)
    else:
        print(
            "  All generator costs are linear (LP).\n"
            "  KKT sensitivity is identically zero within the current basis."
        )


if __name__ == "__main__":
    main()
