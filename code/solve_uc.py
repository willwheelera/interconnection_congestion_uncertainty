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
from scipy.special import comb
import itertools
import concurrent.futures

import pyomo.environ as pe

from egret.common.solver_interface import _solve_model
from egret.data.model_data import ModelData
from egret.models.unit_commitment import (
    create_tight_unit_commitment_model,
    solve_unit_commitment,
    _save_uc_results,
)
import logging
logging.getLogger('egret').setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).parent.parent
EGRET_DIR = REPO_ROOT / "Egret"
TEST_CASE = EGRET_DIR / "egret" / "models" / "tests" / "uc_test_instances" / "test_scuc_masked.json"
RESULTS_DIR = REPO_ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default=str(TEST_CASE),)
    #"Binary list (0/1) of generator availability, one per generator.",
    parser.add_argument("--availability", type=int, nargs="+",)
    parser.add_argument("--solver", type=str, default="gurobi",)
    return parser.parse_args()


def remove_unavailable_generators(md: ModelData, availability: list[int]) -> None:
    """Delete generators whose availability flag is 0."""
    gen_names = list(md.data["elements"]["generator"].keys())
    assert len(availability) == len(gen_names), \
        f"Availability list length ({len(availability)}) != number of generators ({len(gen_names)})"
    for name, avail in zip(gen_names, availability):
        if avail == 0:
            del md.data["elements"]["generator"][name]
            #print(f"  Removed unavailable generator: {name}")
    #print(f"  {len(md.data['elements']['generator'])} generators remaining")


def solve_uc(md: ModelData, solver: str = "gurobi") -> ModelData:
    """Solve unit commitment and return the solved ModelData."""
    md_sol = solve_unit_commitment(
        md, solver=solver, mipgap=0.01, timelimit=300, solver_tee=False,
    )
    total_cost = md_sol.data["system"]["total_cost"]
    #print(f"\nObjective value (total cost): ${total_cost:,.2f}")
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


def get_load(md):
    total_load = np.zeros(24)
    for load in md.data["elements"]["load"].values():
        total_load += np.array(load["p_load"]["values"])
    max_load = total_load.max()
    initial_load = total_load[0]
    return max_load, initial_load

def get_gen_case(md, av):
    gen_dict = md.data["elements"]["generator"]
    genkeys = list(gen_dict.keys())
    total_cap = 0.
    min_on = 0.
    for w in np.where(av)[0]:
        p_max = gen_dict[genkeys[w]]["p_max"]
        p_min = gen_dict[genkeys[w]]["p_min"]
        if type(p_max) == dict: # initial val only
            p_max = p_max["values"][0]
            p_min = p_min["values"][0]
        total_cap += p_max
        min_on += p_min
    return total_cap, min_on


def create_case(md_ref, availability, load_ratio):
    md = copy.deepcopy(md_ref)
    remove_unavailable_generators(md, availability)
    for gen in md.data["elements"]["generator"].values():
        gen["initial_p_output"] = 0#gen["p_min"] #max(gen["p_min"], load_ratio * gen["p_max"])
        if "min_down_time" in gen.keys():
            gen["initial_status"] = -gen["min_down_time"]
    return md
        
    
def avail2code(availability):
    return int("".join(str(b) for b in availability), 2)


def result_path(case_path: Path, availability: list[int] | None) -> Path:
    """Build the output file path from case name and availability flags."""
    case_id = case_path.stem
    case_dir = case_id.replace("_uc.json", "").replace(".json", "")
    out_dir = RESULTS_DIR / case_dir
    out_dir.mkdir(exist_ok=True)
    if availability is not None:
        avail_code = int("".join(str(b) for b in availability), 2)
        return out_dir / f"uc_{case_id}_avail{avail_code}.json.gz"
    return out_dir / f"uc_{case_id}.json.gz"


def main():
    args = parse_args()
    case_path = Path(args.case)
    md = ModelData.read(str(case_path))

    availability = args.availability
    if args.availability is None:
        availability = "1"* len(md.data["elements"]["generator"])
    av = list(map(int, availability))
    run(md, av, case_path, args.solver)


def run(md, availability, case_path, max_load=None, solver="gurobi"):
    total_cap, min_on = get_gen_case(md, availability)

    if max_load is None:
        max_load, initial_load = get_load(md)
    if total_cap < max_load:
        print(f"Skipping {avail2code(availability)} ({np.sum(availability)} gens): max load infeasible")
        return None

    initial_load = 0.
    for load in md.data["elements"]["load"].values():
        initial_load += load["p_load"]["values"][0]
    
    md = create_case(md, availability, initial_load / total_cap)
    md_uc = solve_uc(md, solver=solver)
    md_sol = solve_dcopf_with_fixed_commitments(md_uc, solver=solver)
    out = result_path(case_path, availability)
    md_sol.write(str(out))
    #print(f"Result {avail2code(availability)}, avail: {np.sum(availability)}")
    return out


def run_many_cases(case_path, n_avail=3):
    md = ModelData.read(str(case_path))
    max_load, initial_load = get_load(md)

    gen_dict = md.data["elements"]["generator"]
    ngen = len(gen_dict)
    genkeys = list(gen_dict.keys())
    print(f"Checking {comb(ngen, n_avail):.0f} configs")

    viable_configs = 0

    with concurrent.futures.ProcessPoolExecutor() as ex:
        out = []
        #for indices in itertools.combinations(range(ngen), n_avail):
        #    av[:] = 0
        #    av[list(indices)] = 1
        for i in range(int(2**ngen)):
            av = list(map(int, np.binary_repr(i, ngen)))
            out.append(ex.submit(run, md, av, case_path, max_load=max_load))
            #out = run(md, av, case_path)
        for o in out:
            if o.result():
                viable_configs += 1

    #print(f"viable {viable_configs} / {comb(ngen, n_avail):.0f}")
    print(f"viable {viable_configs} / {2**ngen:.0f}")

def run_nminus1_cases(case_path, nremove=1):
    md = ModelData.read(str(case_path))
    md.data["elements"]["contingency"] = {}
    max_load, initial_load = get_load(md)

    gen_dict = md.data["elements"]["generator"]
    ngen = len(gen_dict)
    genkeys = list(gen_dict.keys())
    n_avail = ngen - nremove
    print(f"Checking {comb(ngen, n_avail):.0f} configs")

    viable_configs = 0
    total_configs = 0

    with concurrent.futures.ProcessPoolExecutor() as ex:
        out = []
        for indices in itertools.combinations(range(ngen), n_avail):
            av = np.zeros(ngen, dtype=int)
            av[list(indices)] = 1
            out.append(ex.submit(run, md, av, case_path))
        for o in out:
            total_configs += 1
            if o.result():
                viable_configs += 1

    print(f"viable {viable_configs} / {total_configs}")

    
if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    case_path = REPO_ROOT / "data" / "case14_gen_per_bus.json"
    run_nminus1_cases(case_path, 0)
    print(f"time {time.perf_counter()-t0:.2f}")
