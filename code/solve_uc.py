"""
Solve security-constrained unit commitment (SCUC) with EGRET, then re-solve
with fixed commitments to obtain LP duals (LMPs, congestion duals).

The saved result includes custom fields on each branch element:
  - mu_upper / mu_lower: congestion dual decomposition ($/MWh)
  - nu: power-flow equality constraint dual ($/MWh)

Usage:
    uv run python code/solve_uc.py
    uv run python code/solve_uc.py --case path/to/case.json
    uv run python code/solve_uc.py --case path/to/case.json --availability 1 1 0 1 ...
"""

import argparse
import copy
from pathlib import Path
import numpy as np

import pyomo.environ as pe

from egret.common.solver_interface import _solve_model
from egret.data.model_data import ModelData
from egret.models.unit_commitment import (
    create_tight_unit_commitment_model,
    solve_unit_commitment,
    _save_uc_results,
)

REPO_ROOT = Path(__file__).parent.parent
EGRET_DIR = REPO_ROOT / "Egret"
TEST_CASE = EGRET_DIR / "egret" / "models" / "tests" / "uc_test_instances" / "test_scuc_masked.json"
RESULTS_DIR = REPO_ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve network-constrained unit commitment problem."
    )
    parser.add_argument(
        "--case", type=str, default=str(TEST_CASE),
        help="Path to an EGRET ModelData JSON case file.",
    )
    parser.add_argument(
        "--availability", type=int, nargs="+",
        help="Binary list (0/1) of generator availability, one per generator.",
    )
    parser.add_argument(
        "--solver", type=str, default="gurobi",
        help="Solver for UC and LP re-solve (default: gurobi).",
    )
    return parser.parse_args()


def remove_unavailable_generators(md: ModelData, availability: list[int]) -> None:
    """Delete generators whose availability flag is 0."""
    gen_names = list(md.data["elements"]["generator"].keys())
    assert len(availability) == len(gen_names), \
        f"Availability list length ({len(availability)}) != number of generators ({len(gen_names)})"
    for name, avail in zip(gen_names, availability):
        if avail == 0:
            del md.data["elements"]["generator"][name]
            print(f"  Removed unavailable generator: {name}")
    print(f"  {len(md.data['elements']['generator'])} generators remaining")


def solve_uc(md: ModelData, solver: str = "gurobi") -> ModelData:
    """Solve unit commitment and return the solved ModelData."""
    print("Solving unit commitment...")
    md_sol = solve_unit_commitment(
        md, solver=solver, mipgap=0.01, timelimit=300, solver_tee=True,
    )
    total_cost = md_sol.data["system"]["total_cost"]
    print(f"\nObjective value (total cost): ${total_cost:,.2f}")
    return md_sol


def solve_dcopf_with_fixed_commitments(
    md_uc: ModelData, solver: str = "gurobi",
) -> ModelData:
    """Re-solve with fixed UC commitments to get LP duals; store them on branches.

    Builds EGRET's tight UC model with ``btheta_power_flow`` and fixes all
    thermal generator binaries.  The resulting LP yields duals that correctly
    reflect ramping constraints coupling consecutive hours.

    Dual variables stored on each branch element (as time-series):
      - ``nu``: dual of the power-flow equality constraint pf = B(θ_f − θ_t)
      - ``mu_upper`` / ``mu_lower``: congestion dual decomposition where
        mu = LMP_to − LMP_from + nu, split into positive / negative parts
    """
    md_fixed = copy.deepcopy(md_uc)
    for _, b_data in md_fixed.elements("bus"):
        b_data.setdefault("vm", 1.0)
    for _, br_data in md_fixed.elements("branch"):
        br_data.setdefault("resistance", 0.0)
        br_data.setdefault("charging_susceptance", 0.0)
    for _, g_data in md_fixed.elements("generator"):
        if g_data.get("generator_type") == "thermal":
            commitment = g_data.get("commitment")
            if commitment is not None:
                g_data["fixed_commitment"] = commitment

    m = create_tight_unit_commitment_model(
        md_fixed, relaxed=True, network_constraints="btheta_power_flow",
    )
    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    _solve_model(m, solver, solver_tee=False)
    md_sol = _save_uc_results(m, relaxed=True)

    _store_branch_duals(m, md_fixed, md_sol)
    return md_sol


def _store_branch_duals(m, md_fixed: ModelData, md_sol: ModelData) -> None:
    """Extract nu and mu duals from the Pyomo model and store on branch elements."""
    time_period_length_hours = float(m.TimePeriodLengthHours.value)
    buses_data = md_sol.data["elements"]["bus"]
    branches_sol = md_sol.data["elements"]["branch"]
    branches_fixed = md_fixed.data["elements"]["branch"]
    dual = m.model().dual
    time_periods = list(m.TimePeriods)
    n_hours = len(time_periods)

    for bn in branches_sol:
        branches_sol[bn]["nu"] = {"data_type": "time_series", "values": [0.0] * n_hours}
        branches_sol[bn]["mu_upper"] = {"data_type": "time_series", "values": [0.0] * n_hours}
        branches_sol[bn]["mu_lower"] = {"data_type": "time_series", "values": [0.0] * n_hours}

    for dt, mt in enumerate(time_periods):
        block = m.TransmissionBlock[mt]
        lmp_at_t = {b: buses_data[b]["lmp"]["values"][dt] for b in buses_data}
        nu_dict = {
            bn: (dual.get(block.eq_pf_branch[bn], 0.0) or 0.0) / time_period_length_hours
            for bn in block.eq_pf_branch
        }
        for bn, br_data in branches_fixed.items():
            lmp_f = lmp_at_t.get(br_data["from_bus"], 0.0)
            lmp_t = lmp_at_t.get(br_data["to_bus"], 0.0)
            nu = nu_dict.get(bn, 0.0)
            mu = lmp_t - lmp_f + nu
            branches_sol[bn]["nu"]["values"][dt] = nu
            branches_sol[bn]["mu_upper"]["values"][dt] = max(0.0, mu)
            branches_sol[bn]["mu_lower"]["values"][dt] = max(0.0, -mu)


def result_path(case_path: Path, availability: list[int] | None) -> Path:
    """Build the output file path from case name and availability flags."""
    RESULTS_DIR.mkdir(exist_ok=True)
    case_id = case_path.stem
    if availability is not None:
        avail_code = int("".join(str(b) for b in availability), 2)
        return RESULTS_DIR / f"uc_{case_id}_avail{avail_code}.json"
    return RESULTS_DIR / f"uc_{case_id}.json"


def main():
    args = parse_args()
    case_path = Path(args.case)

    print(f"Loading {case_path.name}...")
    md = ModelData.read(str(case_path))

    if args.availability is not None:
        remove_unavailable_generators(md, args.availability)

    md_uc = solve_uc(md, solver=args.solver)

    print("Re-solving with fixed commitments for LP duals...")
    md_sol = solve_dcopf_with_fixed_commitments(md_uc, solver=args.solver)

    out = result_path(case_path, args.availability)
    md_sol.write(str(out))
    print(f"\nResults saved to {out}")


def run(solver="gurobi", prob=0.4, seed=1, availability=None):
    np.random.seed(seed)
    case_path = REPO_ROOT / "data" / "many_gen_four_bus_uc.json"
    md = ModelData.read(str(case_path))

    if availability is None:
        random_vals = np.random.random(len(md.data["elements"]["generator"]))
        availability = (random_vals < prob).astype(int)
    remove_unavailable_generators(md, availability)
    md_uc = solve_uc(md, solver=solver)

    print("Re-solving with fixed commitments for LP duals...")
    md_sol = solve_dcopf_with_fixed_commitments(md_uc, solver=solver)
    out = result_path(case_path, availability)
    md_sol.write(str(out))
    print(f"\nResults saved to {out}")
    print(f"avail: {availability.sum()}")
    return out

    
if __name__ == "__main__":
    av = np.array(list(map(int, "1111011100110010")))
    av = np.array(list(map(int, "1111001100100000")))
    av = np.array(list(map(int, "1000100010000000")))
    out = []
    out.append(run(availability=av))
    #run(prob=1.)
    #for seed in [1, 2, 3]:
    #    out.append(run(seed=seed, prob=0.2))

    for o in out:
        print(f"{o.stem}".replace("uc_many_gen_four_bus_uc_avail", ""))
