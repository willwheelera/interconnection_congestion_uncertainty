"""
Compare analytically-inferred branch congestion duals (μ) against raw Gurobi
duals extracted from the solved Pyomo DCOPF model.

The analytical formula used in sensitivity_analysis._compute_mu_base is:
    μ_upper = max(λ_from − λ_to, 0)
    μ_lower = max(λ_to − λ_from, 0)

The true shadow price of a thermal bound (from KKT stationarity for the pf
variable) is the *reduced cost* of the pf variable:
    rc(pf_l) = λ_from − λ_to − ν_l

where ν_l = m.dual[m.eq_pf_branch[l]] is the flow-equality dual.

    pf at lower bound (−rating):  true μ_lower = max(rc, 0)
    pf at upper bound (+rating):  true μ_upper = max(−rc, 0)

The analytical formula uses only the λ difference and omits ν_l.

Usage:
    uv run python src/test_mu_consistency.py
    uv run python src/test_mu_consistency.py --solution results/uc_test_scuc_masked.json --hour 18
"""

import argparse
import copy
from pathlib import Path

import pyomo.environ as pe
import pyomo.core as pc

from egret.data.model_data import ModelData
from egret.models.dcopf import create_btheta_dcopf_model
from egret.model_library.transmission.tx_utils import scale_ModelData_to_pu, unscale_ModelData_to_pu

from src.sensitivity.barrier import (
    _extract_hour,
    _compute_mu_base,
    _network_state,
    RESULTS_DIR,
)

from src.paths import REPO_ROOT


def _solve_dcopf_with_rc(md: ModelData, barrier: bool = True):
    """Build and solve B-theta DCOPF, returning (md_sol, m) with m.rc populated.

    Adds a reduced-cost suffix to the Pyomo model before solving so that
    Gurobi returns the reduced costs of bounded pf variables.  These are the
    actual shadow prices of the thermal-limit variable bounds.

    By default runs barrier with no crossover (Method=2, Crossover=0) for a
    strictly interior primal-dual point that avoids LP degeneracy.
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

    scale_ModelData_to_pu(md_dcopf, inplace=True)
    m, md_dcopf = create_btheta_dcopf_model(md_dcopf)
    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    m.rc   = pe.Suffix(direction=pe.Suffix.IMPORT)

    opts = {"Method": 2, "Crossover": 0} if barrier else {}
    solver = pe.SolverFactory("gurobi")
    results = solver.solve(m, options=opts, tee=False)
    if str(results.solver.termination_condition) != "optimal":
        raise RuntimeError(f"Solver did not find optimal: {results.solver.termination_condition}")

    unscale_ModelData_to_pu(md_dcopf, inplace=True)
    return md_dcopf, m


def _extract_lmps_from_model(m, baseMVA: float) -> dict[str, float]:
    """Extract LMPs ($/MWh) from power-balance constraint duals."""
    return {b: pc.value(m.dual[m.eq_p_balance[b]]) * baseMVA
            for b in m.eq_p_balance}


def _extract_rc_mu(m, branches: dict) -> dict[str, dict[str, float]]:
    """Extract true thermal bound shadow prices from pf variable reduced costs.

    rc(pf_l) = λ_from − λ_to − ν_l  (from KKT stationarity of pf variable)
      rc > 0 → pf at lower bound, true μ_lower = rc
      rc < 0 → pf at upper bound, true μ_upper = −rc
    """
    out = {}
    for bn in branches:
        rc = m.rc.get(m.pf[bn], 0.0) or 0.0
        out[bn] = {"upper": max(-rc, 0.0), "lower": max(rc, 0.0)}
    return out


def compare_mu(solution_path: Path, hour: int = 0) -> None:
    print(f"Loading {solution_path.name}, hour={hour}...")
    md_raw = ModelData.read(str(solution_path))
    md = _extract_hour(md_raw, hour)

    print("Solving DCOPF (barrier, no crossover) with rc suffix...")
    md_sol, m = _solve_dcopf_with_rc(md)
    baseMVA = md_sol.data["system"].get("baseMVA", 100.0)
    lmp_base = _extract_lmps_from_model(m, baseMVA)

    orig, sol, names, *_ = _network_state(md, md_sol)
    nu_dict = {bn: m.dual.get(m.eq_pf_branch[bn], 0.0) or 0.0
               for bn in m.eq_pf_branch}
    mu_analytical = _compute_mu_base(orig, sol, lmp_base, nu_dict=nu_dict)
    rc_mu = _extract_rc_mu(m, dict(md_sol.elements("branch")))

    print(f"\n{'Branch':<20}  {'Ana upper':>10}  {'RC upper':>10}  {'Δ upper':>10}  "
          f"{'Ana lower':>10}  {'RC lower':>10}  {'Δ lower':>10}  {'pf/rating':>10}")
    print("-" * 107)

    max_err = 0.0
    n_shown = 0
    for bn in names["branch"]:
        br_data = orig["branch"][bn]
        rating = br_data.get("rating_long_term")
        if rating is None:
            continue

        ana_u = mu_analytical["branch_upper"].get(bn, 0.0)
        ana_l = mu_analytical["branch_lower"].get(bn, 0.0)
        rc_u  = rc_mu.get(bn, {}).get("upper", 0.0)
        rc_l  = rc_mu.get(bn, {}).get("lower", 0.0)

        # pf value (from md_sol after unscale)
        pf = sol["branch"][bn].get("pf", 0.0)
        util = pf / rating  # signed utilisation

        threshold = 0.01
        if max(ana_u, ana_l, rc_u, rc_l) >= threshold:
            du = ana_u - rc_u
            dl = ana_l - rc_l
            max_err = max(max_err, abs(du), abs(dl))
            n_shown += 1
            print(f"{bn:<20}  {ana_u:>10.4f}  {rc_u:>10.4f}  {du:>+10.4f}  "
                  f"{ana_l:>10.4f}  {rc_l:>10.4f}  {dl:>+10.4f}  {util:>+10.3f}")

    print("-" * 107)
    print(f"Branches shown (|μ| ≥ {threshold}): {n_shown}")
    print(f"Max |Δ|: {max_err:.4f} $/MWh")
    if max_err < 0.01:
        print("PASS")
    else:
        print("FAIL — formula omits flow-equality dual ν; see module docstring")


def main():
    parser = argparse.ArgumentParser(
        description="Check consistency of analytical μ formula vs. Gurobi reduced costs."
    )
    parser.add_argument("--solution", type=str, required=True,
                        help=f"path to a solved case under {RESULTS_DIR}")
    parser.add_argument("--hour", type=int, default=0)
    args = parser.parse_args()
    compare_mu(Path(args.solution), hour=args.hour)


if __name__ == "__main__":
    main()
