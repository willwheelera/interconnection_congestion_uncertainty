"""Warm-started parallel N-1 SCDCOPF evaluator for the IEEE queue testbeds.

Given a binary configuration of the queue candidates (1 if that project is built),
this returns the congestion dual mu of every binding base and post-contingency thermal
limit, under the same modeling recipe as the Texas sweep so the five-case matrix is
comparable:

  - post-contingency ratings tightened to EMERGENCY_FACTOR x the long-term rating,
  - nodal load shed priced on an ORDC-shaped rising curve (ORDC_C_MIN -> ORDC_C_MAX),
    so an unservable pocket is relieved locally at a scarcity price instead of going
    infeasible,
  - every thermal limit made soft at SHADOW_PRICE_CAP, bounding mu at the cap the way
    ERCOT's Max Shadow Price does.

The IEEE cases have one load level rather than the Texas snapshot collection, so the
workers parallelize over configurations instead of snapshots: each worker builds its
SCDCOPF once with all candidates present and applies a configuration by toggling each
candidate's upper bound and warm re-solving, which carries the basis, the PTDF
factorization, and the accumulated lazy contingency constraints across configurations.

The relief and soft-limit mechanisms are imported from src/texas -- they are generic and
duplicating them would let the two recipes drift apart.

Usage:
    uv run python -m ieee_queue.sample_harness --case case30_queue
    uv run python -m ieee_queue.sample_harness --case case118_queue --random 8
"""

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

import pyomo.environ as pyo
from egret.common.solver_interface import _solve_model
from egret.data.model_data import ModelData
from egret.models.dcopf import _lazy_ptdf_dcopf_model_solve_loop
from egret.models.scopf import create_scopf_model

from ieee_queue.build_queue_case import NETWORKS, add_queue_generators
from texas.load_averaged_target import add_lostload_generators_ordc
from texas.soft_limits import enable_soft_limits, thermal_violations

REPO_ROOT = Path(__file__).parents[2]
DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"

EMERGENCY_FACTOR = 1.2
ORDC_C_MIN = 500.0
ORDC_C_MAX = 5000.0
ORDC_BLOCKS = 8
SHADOW_PRICE_CAP = 3500.0

SOLVER_THREADS = 1              # one thread per worker; the parallelism is over configs


def case_files(case, screened=True):
    """(case json, queue json) for a registered case; the screened case if it exists."""
    path = DATA / f"{case}_screened.json"
    if not (screened and path.exists()):
        path = DATA / f"{case}.json"
    return path, DATA / f"{case}_candidates.json"


def load_candidates(case):
    """The queue candidate records for a case."""
    return json.loads((DATA / f"{case}_candidates.json").read_text())["candidates"]


def set_emergency_factor(md, factor):
    """Override every branch's post-contingency rating to factor x its long-term rating."""
    for b in md.data["elements"]["branch"].values():
        b["rating_emergency"] = factor * b["rating_long_term"]


def extract_mu(m, base_mva):
    """{constraint_id: mu} for binding base ("base:L") and post-contingency ("ctg:C:L") limits."""
    mus = {}
    for bn, con in m.ineq_pf_branch_thermal_bounds.items():
        mu = abs(m.dual.get(con, 0.0) or 0.0) / base_mva
        if mu > 1e-6:
            mus[f"base:{bn}"] = mu
    for (cn, bn), con in m.ineq_pf_contingency_branch_thermal_bounds.items():
        mu = abs(m.dual.get(con, 0.0) or 0.0) / base_mva
        if mu > 1e-6:
            mus[f"ctg:{cn}:{bn}"] = mu
    return mus


def build_persistent(case_path, candidates):
    """Build the SCDCOPF once with every candidate present, wrapped in a persistent solver."""
    md = ModelData.read(str(case_path))
    set_emergency_factor(md, EMERGENCY_FACTOR)
    add_queue_generators(md, candidates, [1] * len(candidates))
    add_lostload_generators_ordc(md, ORDC_C_MIN, ORDC_C_MAX, ORDC_BLOCKS)
    m, md_scaled = create_scopf_model(md, include_feasibility_slack=True)
    base_mva = md.data["system"]["baseMVA"]
    enable_soft_limits(m, SHADOW_PRICE_CAP, base_mva)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solver = pyo.SolverFactory("gurobi_persistent")
    solver.options["OutputFlag"] = 0
    solver.options["Threads"] = SOLVER_THREADS
    solver.set_instance(m)
    cap_pu = {f"queue_{c['id']}": c["capacity_mw"] / base_mva for c in candidates}
    return m, md_scaled, solver, cap_pu, list(cap_pu), base_mva


def solve_config(m, md, solver, config, cap_pu, names, base_mva):
    """Apply a configuration via candidate bounds, warm re-solve the N-1 SCDCOPF, return mu."""
    for on, name in zip(config, names):
        m.pg[name].setub(cap_pu[name] if on else 0.0)
        solver.update_var(m.pg[name])
    _solve_model(m, solver, solver_tee=False, return_solver=True, set_instance=False)
    _lazy_ptdf_dcopf_model_solve_loop(m, md, solver, solver_tee=False)
    return extract_mu(m, base_mva)


def worker(case_path, candidates, task_q, result_q):
    """Persistent worker: build once, then serve configurations off the shared queue."""
    logging.getLogger("egret").setLevel(logging.WARNING)
    m, md, solver, cap_pu, names, base_mva = build_persistent(case_path, candidates)
    for cid, config in iter(task_q.get, None):
        mus = solve_config(m, md, solver, config, cap_pu, names, base_mva)
        overload = thermal_violations(m, base_mva)
        shed = float(pyo.value(m.p_load_shed)) * base_mva
        result_q.put((cid, mus, overload, shed))


class ParallelTarget:
    """Manager over config-parallel warm-started SCDCOPF workers for one case."""

    def __init__(self, case, n_workers=8, screened=True):
        self.case = case
        self.case_path, _ = case_files(case, screened)
        self.candidates = load_candidates(case)
        self.n_workers = n_workers
        self.ctx = mp.get_context("spawn")
        self.workers, self.task_q, self.result_q = [], None, None

    def start(self):
        self.task_q, self.result_q = self.ctx.Queue(), self.ctx.Queue()
        for _ in range(self.n_workers):
            p = self.ctx.Process(target=worker, args=(str(self.case_path), self.candidates,
                                                      self.task_q, self.result_q))
            p.start()
            self.workers.append(p)

    def evaluate_detailed(self, configs):
        """Per-config (mu dict, thermal overload dict MW, system load shed MW), in order.

        A large overload or a nonzero shed marks a configuration the cap or the nodal
        relief had to serve, which is the signal that the recipe is running out of room."""
        for cid, cfg in enumerate(configs):
            self.task_q.put((cid, list(cfg)))
        out = [None] * len(configs)
        for _ in range(len(configs)):
            cid, mus, overload, shed = self.result_q.get()
            out[cid] = (mus, overload, shed)
        return out

    def evaluate(self, configs):
        """Per-config {constraint: mu}, in the order the configs were given."""
        return [r[0] for r in self.evaluate_detailed(configs)]

    def close(self):
        for _ in self.workers:
            self.task_q.put(None)
        for p in self.workers:
            p.join()


def scalar_target(mus):
    """The sampling target: max mu over constraints (0 if nothing binds)."""
    return max(mus.values(), default=0.0)


def describe(label, mus, overload, shed):
    """One line summarizing a configuration's congestion and any relief it needed."""
    con, mu = max(mus.items(), key=lambda kv: kv[1], default=("none", 0.0))
    n_base = sum(1 for k in mus if k.startswith("base:"))
    print(f"{label:<8} max mu {mu:9.2f} $/MWh at {con:<18} "
          f"binding {n_base} base / {len(mus) - n_base} ctg   "
          f"overload {max(overload.values(), default=0.0):7.2f} MW   shed {shed:7.2f} MW")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case30_queue", choices=list(NETWORKS))
    parser.add_argument("--random", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full", action="store_true", help="use the unscreened case")
    args = parser.parse_args()

    import numpy as np
    target = ParallelTarget(args.case, args.workers, screened=not args.full)
    n = len(target.candidates)
    rng = np.random.default_rng(args.seed)
    p = np.array([c["p"] for c in target.candidates])
    configs = [[0] * n, [1] * n] + [(rng.random(n) < p).astype(int).tolist()
                                    for _ in range(args.random)]
    labels = ["all-off", "all-on"] + [f"rand{i}" for i in range(args.random)]

    print(f"{args.case}: {target.case_path.name}  {n} candidates  {args.workers} workers")
    target.start()
    t0 = time.time()
    results = target.evaluate_detailed(configs)
    dt = time.time() - t0
    for label, (mus, overload, shed) in zip(labels, results):
        describe(label, mus, overload, shed)
    print(f"\n{len(configs)} configs in {dt:.1f}s ({dt / len(configs):.3f}s each, "
          f"{dt / len(configs) * (1 << n) / 3600:.1f} h for 2^{n} at this rate)")
    target.close()


if __name__ == "__main__":
    main()
