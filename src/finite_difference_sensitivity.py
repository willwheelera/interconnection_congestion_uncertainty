"""
LMP sensitivity via finite-difference perturbation of DCOPF re-solves.

For each bus j, load is decreased by delta_mw (equivalent to a +delta_mw
injection) and the DCOPF is re-solved.  The sensitivity column for bus j is
(LMP_perturbed - LMP_base) / delta_mw.

Works for both LP (linear costs) and QP (quadratic costs).  For LP problems,
LMPs are piecewise-constant, so the delta must be large enough to cross a basis
change (binding-constraint transition).  The default delta — max(10 MW, 15% of
total system load) — is chosen to be practically informative.  For QP problems,
this method serves as a numerical cross-check against sensitivity_analysis.py.

Saves results to results/sensitivity_fd_<case>.json.

Usage:
    uv run python src/finite_difference_sensitivity.py
    uv run python src/finite_difference_sensitivity.py --case path/to/case.json --delta 50.0
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from egret.data.model_data import ModelData
from egret.models.dcopf import solve_dcopf

from utils.solution_io import get_lmps

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_CASE = (
    REPO_ROOT
    / "Egret/egret/models/tests/transmission_test_instances/test_instances/hvdc_test_case3.json"
)


def _total_load_mw(md: ModelData) -> float:
    """Return total active power load in the system (MW)."""
    total = 0.0
    for _, ld in md.elements(element_type="load"):
        pl = ld.get("p_load", 0.0)
        if isinstance(pl, (int, float)):
            total += pl
    return total


def _inject_at_bus(md: ModelData, bus_name: str, delta_mw: float) -> ModelData:
    """Return a deep copy of md with a +delta_mw injection at bus_name.

    Injection is modelled as a load reduction.  If no load element exists at
    the bus, a temporary negative-load element is added.
    """
    md_perturbed = copy.deepcopy(md)
    loads = md_perturbed.data["elements"]["load"]

    bus_loads = [(name, ld) for name, ld in loads.items() if ld.get("bus") == bus_name]

    if bus_loads:
        load_name, load_data = bus_loads[0]
        load_data["p_load"] = load_data["p_load"] - delta_mw
    else:
        loads[f"_tmp_inj_{bus_name}"] = {
            "in_service": True,
            "p_load": -delta_mw,
            "q_load": 0.0,
            "bus": bus_name,
        }

    return md_perturbed


def compute_sensitivity_fd(
    md: ModelData,
    delta_mw: float | None = None,
    solver: str = "gurobi",
) -> tuple[pd.DataFrame, dict[str, float], float]:
    """Compute the LMP sensitivity matrix dLMP_i/dP_j by finite difference.

    Parameters
    ----------
    md:
        ModelData for the network to analyse.
    delta_mw:
        Perturbation size in MW.  Defaults to max(10, 15% of total load).
    solver:
        Pyomo solver name.

    Returns
    -------
    (df_sensitivity, lmp_base, delta_mw_used)
        df_sensitivity: n_bus × n_bus DataFrame [$/MWh per MW].
        lmp_base: base-case LMPs {bus_name: $/MWh}.
        delta_mw_used: the delta actually applied (useful when auto-computed).
    """
    if delta_mw is None:
        delta_mw = max(10.0, 0.15 * _total_load_mw(md))
        print(f"  Auto delta: {delta_mw:.1f} MW (15% of total load)")

    md_base_sol = solve_dcopf(copy.deepcopy(md), solver=solver, solver_tee=False)
    lmp_base = get_lmps(md_base_sol)
    bus_names = sorted(lmp_base.keys())
    n = len(bus_names)

    print("Base LMPs ($/MWh):")
    for b in bus_names:
        print(f"  bus {b}: {lmp_base[b]:.4f}")

    sensitivity = np.zeros((n, n))
    for j, inj_bus in enumerate(bus_names):
        print(f"  Perturbing bus {inj_bus} (+{delta_mw:.1f} MW)...")
        lmp_pert = get_lmps(
            solve_dcopf(_inject_at_bus(md, inj_bus, delta_mw), solver=solver, solver_tee=False)
        )
        for i, obs_bus in enumerate(bus_names):
            sensitivity[i, j] = (lmp_pert[obs_bus] - lmp_base[obs_bus]) / delta_mw

    df = pd.DataFrame(
        sensitivity,
        index=pd.Index(bus_names, name="observed_bus (LMP_i)"),
        columns=pd.Index(bus_names, name="injection_bus (P_j)"),
    )
    return df, lmp_base, delta_mw


def save_results(
    case_stem: str,
    lmp_base: dict[str, float],
    df: pd.DataFrame,
    delta_mw: float,
) -> Path:
    """Save finite-difference sensitivity results to results/sensitivity_fd_<case>.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"sensitivity_fd_{case_stem}.json"
    payload = {
        "case": case_stem,
        "method": "finite_difference",
        "delta_mw": delta_mw,
        "bus_names": df.columns.tolist(),
        "lmp_base": lmp_base,
        "sensitivity": df.values.tolist(),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Results saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute LMP sensitivity by finite-difference DCOPF re-solves."
    )
    parser.add_argument(
        "--case",
        type=str,
        default=str(DEFAULT_CASE),
        help="Path to EGRET ModelData JSON (or MatPower .m) file.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help=(
            "Perturbation size in MW.  Defaults to max(10, 15%% of total load).  "
            "For LP problems, must be large enough to trigger a basis change."
        ),
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

    buses = dict(md.elements(element_type="bus"))
    loads = dict(md.elements(element_type="load"))
    print(f"  {len(buses)} buses, {len(loads)} load elements")

    print(f"\nComputing finite-difference sensitivity (solver={args.solver})...")
    df, lmp_base, delta_used = compute_sensitivity_fd(md, delta_mw=args.delta, solver=args.solver)

    print("\nSensitivity matrix ($/MWh per MW):")
    print(df.to_string())

    save_results(case_path.stem, lmp_base, df, delta_used)


if __name__ == "__main__":
    main()
