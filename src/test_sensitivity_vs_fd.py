"""
Compare barrier-regularised sensitivity (dλ/dL and dμ/dL) against finite-difference
estimates from DCOPF re-solves.

The barrier method in sensitivity_analysis.compute_sensitivity_barrier returns
analytical derivatives of LMPs and congestion duals w.r.t. bus loads.  This test
verifies that the barrier sensitivity qualitatively agrees with FD re-solves for
both LP and QP cost functions.

FD perturbation: a +delta MW load increase at bus j, implemented as -delta MW
injection via ``_inject_at_bus(md, bus, -delta)``.

Test case: case14_gen_per_bus.json (14-bus IEEE case with one generator per bus).

Tolerance design
----------------
The barrier parameter eps controls the smoothing radius around active/inactive
constraint boundaries.  The FD step delta should be comparable in magnitude to
eps so that both methods "see across" the same smoothing region.

For LP costs the exact LP duals are piecewise-constant, so FD of exact re-solves
gives ≈ 0 for small perturbations that do not change the active set.  The barrier
method, by contrast, returns O(eps) non-zero values that smoothly approximate the
sub-gradient.  The two methods agree qualitatively: both are small, and a formula
error (wrong coefficient, wrong variable) would produce O(1) values that violate
the tolerance.

For QP costs the duals vary smoothly and FD provides a tighter ground truth;
the same tolerance accommodates both regimes.

Usage:
    uv run python src/test_sensitivity_vs_fd.py
    uv run python src/test_sensitivity_vs_fd.py --delta 0.5 --atol 0.05
"""

import argparse
import copy
from pathlib import Path

import numpy as np

from egret.data.model_data import ModelData
from egret.models.dcopf import solve_dcopf

from sensitivity_analysis import (
    _get_lmps,
    _network_state,
    _compute_mu_base,
    _read_pf_dual_dict,
    compute_sensitivity_barrier,
)
from solve_opf import store_branch_duals
from finite_difference_sensitivity import _inject_at_bus


def _solve_with_duals(md: ModelData, solver: str = "gurobi") -> ModelData:
    """Solve DCOPF and attach nu / mu_upper / mu_lower on branches."""
    md_sol, m = solve_dcopf(
        copy.deepcopy(md), solver=solver, solver_tee=False, return_model=True,
    )
    store_branch_duals(m, md_sol)
    return md_sol

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CASE = REPO_ROOT / "data/case14_gen_per_bus.json"
DEFAULT_DELTA = 1.0   # MW — comparable to eps so FD spans the barrier smoothing region
DEFAULT_ATOL  = 0.05  # $/MWh per MW — accommodates O(eps) barrier smoothing at eps=1


def _constraint_slacks(orig: dict, sol: dict) -> dict[str, dict[str, float]]:
    """Return {key: {name: slack}} for all inequality constraints.

    slack > 0 means the constraint is inactive; slack ≈ 0 means exactly active.
    """
    slacks: dict[str, dict[str, float]] = {
        "branch_upper": {}, "branch_lower": {},
        "gen_max": {}, "gen_min": {},
        "dc_max": {}, "dc_min": {},
    }
    for br_name, br_data in orig["branch"].items():
        rating = br_data.get("rating_long_term")
        if rating is None:
            continue
        pf = sol["branch"][br_name].get("pf", 0.0)
        slacks["branch_upper"][br_name] = max(rating - pf, 0.0)
        slacks["branch_lower"][br_name] = max(pf + rating, 0.0)
    for g_name, g_data in orig["gen"].items():
        pg = sol["gen"][g_name].get("pg", 0.0)
        pg_max = g_data.get("p_max")
        pg_min = g_data.get("p_min", 0.0)
        if pg_max is not None:
            slacks["gen_max"][g_name] = max(pg_max - pg, 0.0)
        slacks["gen_min"][g_name] = max(pg - pg_min, 0.0)
    for dc_name, dc_data in orig["dc"].items():
        rating = dc_data.get("rating_long_term")
        if rating is None:
            continue
        pf = sol["dc"][dc_name].get("pf", 0.0)
        slacks["dc_max"][dc_name] = max(rating - pf, 0.0)
        slacks["dc_min"][dc_name] = max(pf + rating, 0.0)
    return slacks


def _fd_lmp_column(md_base: ModelData, bus_name: str, delta: float, solver: str = "gurobi") -> dict[str, float]:
    """Return LMPs of the DCOPF perturbed by +delta MW load at bus_name."""
    md_pert = _inject_at_bus(md_base, bus_name, -delta)  # -delta injection = +delta load
    md_pert_sol = _solve_with_duals(md_pert, solver=solver)
    return _get_lmps(md_pert_sol)


def _fd_mu_column(
    md_base: ModelData,
    bus_name: str,
    delta: float,
    mu_base_vals: dict[str, dict[str, float]],
    solver: str = "gurobi",
) -> dict[str, dict[str, float]]:
    """Return FD estimate {key: {name: (μ_pert − μ_base) / delta}} for +delta load at bus."""
    md_pert = _inject_at_bus(md_base, bus_name, -delta)  # -delta injection = +delta load
    md_pert_sol = _solve_with_duals(md_pert, solver=solver)
    lmp_pert = _get_lmps(md_pert_sol)
    pf_dual_pert = _read_pf_dual_dict(md_pert_sol)
    mu_pert = _compute_mu_base(md_pert_sol, lmp_pert, pf_dual_pert)
    return {
        key: {name: (mu_pert[key].get(name, 0.0) - v) / delta
              for name, v in base_vals.items()}
        for key, base_vals in mu_base_vals.items()
    }


def compare_sensitivity(
    case_path: Path = DEFAULT_CASE,
    delta: float = DEFAULT_DELTA,
    atol: float = DEFAULT_ATOL,
    solver: str = "gurobi",
) -> bool:
    """Run barrier vs FD comparison and return True if all tested entries pass atol.

    Prints a table of the largest discrepancies for LMP and μ sensitivity.
    """
    print(f"Loading {case_path.name}...")
    md = ModelData.read(str(case_path))

    print("Solving base DCOPF...")
    md_sol = _solve_with_duals(md, solver=solver)
    lmp_base = _get_lmps(md_sol)
    pf_dual_dict = _read_pf_dual_dict(md_sol)
    print("Base LMPs ($/MWh):", {b: f"{v:.4f}" for b, v in sorted(lmp_base.items())})

    orig, sol, names, *_ = _network_state(md, md_sol)
    mu_base_vals = _compute_mu_base(md_sol, lmp_base, pf_dual_dict)
    slacks = _constraint_slacks(orig, sol)

    print("\nComputing barrier sensitivity (eps=1.0)...")
    df_lmp_bar, mu_sens_bar = compute_sensitivity_barrier(
        md, md_sol, lmp_base, pf_dual_dict
    )

    bus_names = names["bus"]
    n = len(bus_names)
    dim_map = {
        "branch_upper": names["branch"], "branch_lower": names["branch"],
        "gen_max":      names["gen"],    "gen_min":      names["gen"],
        "dc_max":       names["dc"],     "dc_min":       names["dc"],
    }

    # ── Collect FD columns ──────────────────────────────────────────────────
    lmp_fd = np.zeros((n, n))
    mu_fd: dict[str, np.ndarray] = {
        key: np.zeros((len(row_names), n)) for key, row_names in dim_map.items()
    }
    for j, load_bus in enumerate(bus_names):
        print(f"  FD perturbing bus {load_bus} (+{delta:.1f} MW load)...")
        lmp_pert = _fd_lmp_column(md, load_bus, delta, solver=solver)
        for i, obs_bus in enumerate(bus_names):
            lmp_fd[i, j] = (lmp_pert[obs_bus] - lmp_base[obs_bus]) / delta
        fd_mu_col = _fd_mu_column(md, load_bus, delta, mu_base_vals, solver=solver)
        for key, row_names in dim_map.items():
            for r_i, name in enumerate(row_names):
                mu_fd[key][r_i, j] = fd_mu_col[key].get(name, 0.0)

    # ── Compare LMP sensitivity ─────────────────────────────────────────────
    print(f"\n{'=== LMP sensitivity dλ/dL (barrier vs FD) ':=<60}")
    lmp_bar = df_lmp_bar.values
    lmp_err = np.abs(lmp_bar - lmp_fd)
    max_lmp_err = lmp_err.max()
    lmp_pass = max_lmp_err <= atol
    print(f"  Max |Δ| = {max_lmp_err:.5f}  (atol = {atol})  "
          f"[{'PASS' if lmp_pass else 'FAIL'}]")
    if not lmp_pass:
        print(f"  {'obs_bus':>8}  {'load_bus':>8}  {'barrier':>10}  {'FD':>10}  {'Δ':>10}")
        for ij in np.argwhere(lmp_err > atol * 0.5):
            i, j = ij
            print(f"  {bus_names[i]:>8}  {bus_names[j]:>8}  "
                  f"{lmp_bar[i,j]:>10.5f}  {lmp_fd[i,j]:>10.5f}  "
                  f"{lmp_bar[i,j]-lmp_fd[i,j]:>+10.5f}")

    # ── Compare μ sensitivity ────────────────────────────────────────────────
    print(f"\n{'=== μ sensitivity dμ/dL (barrier vs FD) ':=<60}")
    max_mu_err = 0.0
    mu_fail = False
    for key, row_names in dim_map.items():
        bar_vals = mu_sens_bar[key].values
        fd_vals  = mu_fd[key]
        key_slacks = slacks.get(key, {})
        for r_i, name in enumerate(row_names):
            slack = key_slacks.get(name, float("inf"))
            row_err = np.abs(bar_vals[r_i, :] - fd_vals[r_i, :]).max()
            if row_err < 1e-9 and np.abs(bar_vals[r_i, :]).max() < 1e-9:
                continue  # both barrier and FD are zero — trivially pass
            max_mu_err = max(max_mu_err, row_err)
            row_pass = row_err <= atol
            if not row_pass:
                mu_fail = True
            print(f"  {key}[{name}]: max |Δ| = {row_err:.5f}  "
                  f"[{'PASS' if row_pass else 'FAIL'}]  slack={slack:.2f} MW")
            if not row_pass:
                for j, load_bus in enumerate(bus_names):
                    e = abs(bar_vals[r_i, j] - fd_vals[r_i, j])
                    if e > atol * 0.5:
                        print(f"    ← bus {load_bus}: "
                              f"barrier={bar_vals[r_i,j]:+.5f}  "
                              f"FD={fd_vals[r_i,j]:+.5f}  "
                              f"Δ={bar_vals[r_i,j]-fd_vals[r_i,j]:+.5f}")

    passed = lmp_pass and not mu_fail
    print(f"\n{'PASS' if passed else 'FAIL'} — max LMP err={max_lmp_err:.5f}, "
          f"max μ err (interior)={max_mu_err:.5f}  (atol={atol})")
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Compare barrier sensitivity against finite-difference estimates."
    )
    parser.add_argument("--case",      type=str,   default=str(DEFAULT_CASE))
    parser.add_argument("--delta",     type=float, default=DEFAULT_DELTA,
                        help="FD perturbation in MW (default: 1.0)")
    parser.add_argument("--atol",      type=float, default=DEFAULT_ATOL,
                        help="Absolute tolerance in $/MWh per MW (default: 0.05)")
    parser.add_argument("--solver",    type=str,   default="gurobi",
                        help="Pyomo solver name (default: gurobi)")
    args = parser.parse_args()
    passed = compare_sensitivity(
        Path(args.case), delta=args.delta, atol=args.atol, solver=args.solver,
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
