"""Time a single DCOPF solve on the ACTIVSg2000 model and summarize congestion.

Loads the converted EGRET JSON, solves DCOPF several times to measure per-solve
wall time (parse cost excluded), and reports the congestion picture: number of
lines at their thermal limit and the largest congestion dual mu. This confirms
the model solves and sets the per-evaluation cost budget for the sampling study.

Usage:
    uv run python src/texas/time_dcopf.py
    uv run python src/texas/time_dcopf.py --repeats 10 --solver gurobi
"""

import argparse
import time
from pathlib import Path

from src.paths import REPO_ROOT

from egret.data.model_data import ModelData
from src.model.dcopf import solve_opf, store_branch_duals

CASE = REPO_ROOT / "data" / "activsg2000.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default=str(CASE))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--solver", type=str, default="gurobi")
    return parser.parse_args()


def time_solves(md: ModelData, repeats: int, solver: str) -> tuple[list[float], ModelData, object]:
    """Solve DCOPF `repeats` times; return per-solve seconds and the last result."""
    times = []
    md_sol = m = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        md_sol, m = solve_opf(md, solver=solver)
        times.append(time.perf_counter() - t0)
    return times, md_sol, m


def congestion_summary(md_sol: ModelData, m) -> dict:
    """Count binding lines and find the max congestion dual after storing duals."""
    store_branch_duals(m, md_sol)
    branches = md_sol.data["elements"]["branch"]
    mus = {bn: max(b["mu_upper"], b["mu_lower"]) for bn, b in branches.items()}
    binding = {bn: mu for bn, mu in mus.items() if mu > 1e-6}
    top = max(mus.items(), key=lambda kv: kv[1])
    loads = md_sol.data["elements"]["load"]
    return {
        "total_load_MW": sum(l["p_load"] for l in loads.values()),
        "total_cost": md_sol.data["system"]["total_cost"],
        "n_binding_lines": len(binding),
        "max_mu": top[1],
        "max_mu_line": top[0],
    }


def main():
    args = parse_args()
    md = ModelData.read(args.case)
    times, md_sol, m = time_solves(md, args.repeats, args.solver)
    s = congestion_summary(md_sol, m)
    print(f"case {args.case}")
    print(f"solver {args.solver}  repeats {args.repeats}")
    print(f"solve time (s)  min {min(times):.3f}  mean {sum(times)/len(times):.3f}  "
          f"max {max(times):.3f}")
    print(f"total load {s['total_load_MW']:.0f} MW  total cost {s['total_cost']:.0f}")
    print(f"binding lines {s['n_binding_lines']}  "
          f"max mu {s['max_mu']:.2f} $/MWh on line {s['max_mu_line']}")


if __name__ == "__main__":
    main()
