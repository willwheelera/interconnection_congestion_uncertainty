"""
Solve DC optimal power flow (DCOPF) for a single time period with EGRET,
then extract LP duals (LMPs and congestion shadow prices).

The saved result includes custom fields on each branch element:
  - mu_upper / mu_lower: congestion dual decomposition ($/MWh)
  - nu: power-flow equality constraint dual ($/MWh)

Usage:
    uv run python code/solve_opf.py
    uv run python code/solve_opf.py --case data/case14_gen_per_bus.json
    uv run python code/solve_opf.py --case data/case14_gen_per_bus.json --availability 1 1 0 1 ...
"""

import argparse
import copy
from pathlib import Path
import numpy as np
from scipy.special import comb
import itertools
import concurrent.futures
import time

import pyomo.environ as pe
from pyomo.environ import value
from math import degrees

from egret.data.model_data import ModelData
from egret.models.dcopf import create_btheta_dcopf_model, solve_dcopf
from egret.common.solver_interface import _solve_model

from utils.availability import avail_to_id, id_to_avail, remove_unavailable_generators
from utils.modify_network import scale_all_loads
import logging
logging.getLogger('egret').setLevel(logging.WARNING)
logging.getLogger('pyomo.core').addFilter(lambda r: getattr(r, 'id', None) != 'W1002')

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_CASE = REPO_ROOT / "data" / "case14_gen_per_bus.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default=str(DEFAULT_CASE))
    parser.add_argument("--availability", type=int, nargs="+")
    parser.add_argument("--solver", type=str, default="gurobi")
    return parser.parse_args()


def solve_opf(md: ModelData, solver: str = "gurobi") -> tuple[ModelData, pe.ConcreteModel]:
    """Solve DCOPF and return (solved ModelData, pyomo model)."""
    md_sol, m = solve_dcopf(
        md,
        solver=solver,
        solver_tee=False,
        return_model=True,
        include_feasibility_slack=True,
    )
    return md_sol, m


def store_branch_duals(m, md_sol: ModelData) -> None:
    """Extract nu and mu duals from the Pyomo model and store on branch elements."""
    buses = md_sol.data["elements"]["bus"]
    branches = md_sol.data["elements"]["branch"]
    baseMVA = md_sol.data["system"]["baseMVA"]
    dual = m.dual

    for bn, br_data in branches.items():
        # The Pyomo dual is in per-unit ($/p.u.); divide by baseMVA to get $/MWh,
        # matching the already-unscaled LMPs on buses.
        nu = (dual.get(m.eq_pf_branch[bn], 0.0) or 0.0) / baseMVA
        lmp_f = buses[br_data["from_bus"]]["lmp"]
        lmp_t = buses[br_data["to_bus"]]["lmp"]
        mu = lmp_t - lmp_f + nu
        br_data["nu"] = nu
        br_data["mu_upper"] = max(0.0, mu)
        br_data["mu_lower"] = max(0.0, -mu)


def get_total_load(md: ModelData) -> float:
    return sum(load["p_load"] for load in md.data["elements"]["load"].values())


def get_gen_capacity(md: ModelData, availability) -> tuple[float, float]:
    gen_dict = md.data["elements"]["generator"]
    genkeys = list(gen_dict.keys())
    total_cap = 0.0
    min_on = 0.0
    for w in np.where(availability)[0]:
        total_cap += gen_dict[genkeys[w]]["p_max"]
        min_on += gen_dict[genkeys[w]]["p_min"]
    return total_cap, min_on


def create_case(md_ref: ModelData, availability) -> ModelData:
    md = copy.deepcopy(md_ref)
    remove_unavailable_generators(md, availability)
    return md


def result_path(case_path: Path, availability: list[int] | None, scale=None) -> Path:
    case_id = case_path.stem
    case_dir = case_id.replace(".json", "")
    out_dir = RESULTS_DIR / case_dir
    out_dir.mkdir(exist_ok=True)
    label = ""
    if availability is not None:
        label += f"_avail{avail_to_id(availability)}"
    if scale is not None:
        label += f"_load{int(scale*100)}"
    return out_dir / f"opf_{case_id}{label}.json.gz"


def run(md, availability, case_path, total_load=None, solver="gurobi"):
    md = create_case(md, availability)
    try:
        md_sol, m = solve_opf(md, solver=solver)
    except Exception as e:
        print(availability)
        print(avail_to_id(availability))
        print(e)
        return None
    store_branch_duals(m, md_sol)
    out = result_path(case_path, availability)
    md_sol.write(str(out))
    return out


def main():
    args = parse_args()
    case_path = Path(args.case)
    md = ModelData.read(str(case_path))

    availability = args.availability
    if availability is None:
        availability = [1] * len(md.data["elements"]["generator"])
    run(md, availability, case_path, solver=args.solver)


def run_many_cases(case_path, solver="gurobi"):
    md = ModelData.read(str(case_path))
    total_load = get_total_load(md)

    ngen = len(md.data["elements"]["generator"])
    print(f"Checking {2**ngen} configs")

    viable_configs = 0
    total_configs = 0
    
    t0 = time.perf_counter()
    times = []

    with concurrent.futures.ProcessPoolExecutor() as ex:
        out = []
        for i in range(int(2**ngen)):
            av = list(map(int, np.binary_repr(i, ngen)))
            out.append(ex.submit(run, md, av, case_path, total_load=total_load, solver=solver))
        for o in out:
            total_configs += 1
            if o.result():
                viable_configs += 1

            # Print progress
            times.append(time.perf_counter() - t0)
            if (total_configs) % 100 == 0:
                print(f"{total_configs:>5d} ({viable_configs:>5d}) time:", end="  ")
                for t in times:
                    print(f"{t:.1f}", end="  ")
                print()
                times = []

    print(f"viable {viable_configs} / {total_configs}")


def run_nminus1_cases(case_path, nremove=1, solver="gurobi"):
    md = ModelData.read(str(case_path))
    md.data["elements"]["contingency"] = {}

    gen_dict = md.data["elements"]["generator"]
    ngen = len(gen_dict)
    n_avail = ngen - nremove
    print(f"Checking {comb(ngen, n_avail):.0f} configs")

    viable_configs = 0
    total_configs = 0

    with concurrent.futures.ProcessPoolExecutor() as ex:
        out = []
        for indices in itertools.combinations(range(ngen), n_avail):
            av = np.zeros(ngen, dtype=int)
            av[list(indices)] = 1
            out.append(ex.submit(run, md, av, case_path, solver=solver))
        for o in out:
            total_configs += 1
            if o.result():
                viable_configs += 1

    print(f"viable {viable_configs} / {total_configs}")


def run_load(md, av, scale, case_path, solver="gurobi"):
    md = copy.deepcopy(md)
    scale_all_loads(md, scale)
    try:
        md_sol, m = solve_opf(md, solver=solver)
    except Exception as e:
        print("scale", scale)
        print(e)
        return None
    store_branch_duals(m, md_sol)
    out = result_path(case_path, av, scale)
    md_sol.write(str(out))
    return out

def run_load_scan(case_path, solver="gurobi"):
    md = ModelData.read(str(case_path))
    md.data["elements"]["contingency"] = {}

    gen_dict = md.data["elements"]["generator"]
    ngen = len(gen_dict)
    av = id_to_avail(540, ngen)
    remove_unavailable_generators(md, av)

    run_load(md, av, 0.5, case_path, solver=solver)
    quit()
    with concurrent.futures.ProcessPoolExecutor() as ex:
        out = []
        for scale in [.5, .6, .7, .8, .9, 1.]:
            out.append(ex.submit(run_load, md, av, scale, case_path, solver=solver))
        for o in out:
            o.result()
    

if __name__ == "__main__":
    t0 = time.perf_counter()
    case_path = REPO_ROOT / "data" / "case14_gen_per_bus.json"
    #run_many_cases(case_path)
    #run_nminus1_cases(case_path, 7)
    run_load_scan(case_path)
    print(f"time {time.perf_counter()-t0:.2f}")
