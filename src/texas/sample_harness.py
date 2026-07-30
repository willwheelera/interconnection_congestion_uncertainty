"""Parallel warm-started evaluator for the load-averaged congestion target.

Implements speedups (2) parallelization and (3) warm-starting together, in the
snapshot-pinned manager/worker design:

  - One worker process per load snapshot (the load vector a worker sees never
    changes). The manager broadcasts each candidate configuration to every
    worker and reduces the per-snapshot congestion duals into the load average.
  - Each worker builds its SCDCOPF model ONCE with all 20 queue candidates
    present, wrapped in a persistent Gurobi solver. A configuration is applied by
    toggling each candidate generator's upper bound (capacity if online, else 0)
    and warm re-solving. Because only variable bounds change between configs and
    the load is fixed per worker, the basis, the PTDF factorization, and the
    accumulated lazy contingency constraints all carry across configs.

The workers stay alive across evaluate() calls, so a sampler can reuse them for
many batches without rebuilding the persistent models. Use the screened case
(data/activsg2000_2025_screened.json) so each solve is cheap.

Usage:
    uv run python src/texas/sample_harness.py            # verify vs known configs
    uv run python src/texas/sample_harness.py --random 8 # time a batch
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import egret.common.lazy_ptdf_utils as lpu
from egret.data.model_data import ModelData
from egret.data.ptdf_utils import VirtualPTDFMatrix
from egret.model_library.defn import BasePointType
from egret.models.scopf import create_scopf_model
from egret.models.dcopf import _lazy_ptdf_dcopf_model_solve_loop
from egret.common.solver_interface import _solve_model

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "texas"))

from load_averaged_target import (add_queue_generators, add_lostload_generators_ordc,
                                   scale_loads_by_zone, scale_renewables_by_snapshot)
from soft_limits import enable_soft_limits, thermal_violations

# Which fleet/recipe the harness solves, selected by HARNESS_CASE (default 2025). Each
# entry names the screened SCDCOPF, the full case (all contingencies, for audit), and the
# results subdirectory the sweep writes to. "original_v12" is the ~2016 fleet under the
# same factor-1.2 soft-cap recipe as 2025. Both screened sets come from
# screen_harness_recipe.py, which screens at this recipe rather than the JSON's native 1.5.
CASE_REGISTRY = {
    "2025": dict(screened="activsg2000_2025_screened.json",
                 full="activsg2000_2025.json", results="activsg2000_2025"),
    "original_v12": dict(screened="activsg2000_original_v12_screened.json",
                         full="activsg2000.json", results="activsg2000_original_v12"),
}
ACTIVE_CASE = os.environ.get("HARNESS_CASE", "2025")
_spec = CASE_REGISTRY[ACTIVE_CASE]
CASE = REPO_ROOT / "data" / _spec["screened"]
FULL_CASE = REPO_ROOT / "data" / _spec["full"]
RESULTS_NAME = _spec["results"]
QUEUE = REPO_ROOT / "data" / "activsg2000_queue20.json"
COLLECTION = REPO_ROOT / "data" / "activsg2000_load_collection.json"

# Realistic-mu recipe (see project note / check_relief_curve_mu.py): tighten the
# post-contingency rating to EMERGENCY_FACTOR x long-term to create scarcity, price
# nodal load-shed on an ORDC-shaped rising curve (ORDC_C_MIN -> ORDC_C_MAX over
# ORDC_BLOCKS), and cap every congestion dual at the ERCOT Max Shadow Price
# SHADOW_PRICE_CAP via soft (overload-penalized) thermal limits.
EMERGENCY_FACTOR = 1.2
ORDC_C_MIN = 500.0
ORDC_C_MAX = 5000.0
ORDC_BLOCKS = int(os.environ.get("HARNESS_ORDC_BLOCKS", "8"))
SHADOW_PRICE_CAP = 3500.0

# Gurobi threads per snapshot worker. With one worker per snapshot, the default
# (all cores per solver) oversubscribes the machine; a small cap avoids the
# contention that dominates the parallel wall time. 0 = Gurobi default.
SOLVER_THREADS = int(os.environ.get("HARNESS_THREADS", "0"))

# Restrict the nodal ORDC relief to a chosen bus set (a json list of bus ids from
# select_relief_set.py) instead of every load bus; None = all load buses. Only buses
# that ever shed need relief, so slimming shrinks the LP ~5x. A config that needs a
# bus not in the set stays feasible via line overload / the system slack, which the
# saved max-overload and load-shed columns expose for a later full-relief recheck.
_relief_file = os.environ.get("HARNESS_RELIEF_SET", "")
RELIEF_BUSES = set(json.loads(Path(_relief_file).read_text())) if _relief_file else None

# Post-contingency overflow (per-unit) above which a dropped contingency is
# judged violated at the screened optimum -- ~0.01 MW on a 100 MVA base.
AUDIT_TOL = 1e-4


def extract_mu(m, base_mva: float) -> dict[str, float]:
    """Return {constraint_id: mu} for binding base and post-contingency limits."""
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


def set_emergency_factor(md, factor):
    """Override every branch's post-contingency emergency rating to factor x long-term."""
    for b in md.data["elements"]["branch"].values():
        b["rating_emergency"] = factor * b["rating_long_term"]


def build_persistent(snapshot, case_path, candidates):
    """Build this snapshot's SCDCOPF once (all candidates present) + persistent solver.

    Applies the realistic-mu recipe: tightened emergency ratings, ORDC-shaped nodal
    relief, and soft (capped) thermal limits, so every congestion dual is bounded by
    SHADOW_PRICE_CAP. Tagged renewables (CDR fleet and queue wind/solar) are derated
    to this snapshot's capacity factor, so solar/wind availability tracks the hour.
    The 20 queue candidates are all present; a config is applied later by toggling
    their bounds (solve_config), the online bound being the CF-derated capacity, so
    the persistent basis, PTDF factorization, relief generators, and slacks all carry
    across configs.
    """
    cf = snapshot.get("renewable_cf") or {}
    md = ModelData.read(case_path)
    set_emergency_factor(md, EMERGENCY_FACTOR)
    add_queue_generators(md, candidates, [1] * len(candidates))
    scale_loads_by_zone(md, snapshot["zone_scales"])
    scale_renewables_by_snapshot(md, cf)
    add_lostload_generators_ordc(md, ORDC_C_MIN, ORDC_C_MAX, ORDC_BLOCKS, buses=RELIEF_BUSES)
    m, md_scaled = create_scopf_model(md, include_feasibility_slack=True)
    base_mva = md.data["system"]["baseMVA"]
    enable_soft_limits(m, SHADOW_PRICE_CAP, base_mva)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solver = pyo.SolverFactory("gurobi_persistent")
    solver.options["OutputFlag"] = 0
    if SOLVER_THREADS:
        solver.options["Threads"] = SOLVER_THREADS
    solver.set_instance(m)
    cap_pu = {f"queue_{c['inr']}": cf.get(c["fuel"], 1.0) * c["capacity_mw"] / base_mva
              for c in candidates}
    names = list(cap_pu)
    return m, md_scaled, solver, cap_pu, names, base_mva


def solve_config(m, md, solver, config, cap_pu, names, base_mva) -> dict[str, float]:
    """Apply a config via generator bounds, warm re-solve N-1 SCDCOPF, return mu."""
    for on, name in zip(config, names):
        m.pg[name].setub(cap_pu[name] if on else 0.0)
        solver.update_var(m.pg[name])
    _solve_model(m, solver, solver_tee=False, return_solver=True, set_instance=False)
    _lazy_ptdf_dcopf_model_solve_loop(m, md, solver, solver_tee=False)
    return extract_mu(m, base_mva)


def build_audit_ptdf(md_scaled) -> VirtualPTDFMatrix:
    """Full-contingency PTDF on the scaled topology, for auditing dropped ctgs.

    Reuses md_scaled's per-unit branches/buses (identical ratings and ordering to
    the solve model) but attaches the complete contingency set, so post-outage
    flows can be computed for the contingencies the screened case omits.
    """
    branches = dict(md_scaled.elements(element_type="branch"))
    buses = dict(md_scaled.elements(element_type="bus"))
    ref = md_scaled.data["system"]["reference_bus"]
    full = json.loads(FULL_CASE.read_text())["elements"]["contingency"]
    return VirtualPTDFMatrix(branches, buses, ref, BasePointType.FLATSTART,
                             lpu.populate_default_ptdf_options(None),
                             branches_keys=tuple(branches), buses_keys=tuple(buses),
                             contingencies=full)


def audit_dropped(m, audit_ptdf, screened: set) -> list[tuple]:
    """Return dropped contingencies violated at the current dispatch.

    If none are violated, the screened solution is provably the full-problem
    solution (adding already-satisfied constraints cannot change the optimum).
    Each entry: (contingency, worst branch, |post-flow| pu, emergency limit pu).
    """
    PFV, _, VA = audit_ptdf.calculate_PFV(m)
    limits = audit_ptdf.contingency_limits_array
    bkeys = audit_ptdf.branches_keys
    violated = []
    for cn in audit_ptdf.contingencies:
        if cn in screened:
            continue
        post = PFV + audit_ptdf.calculate_PFV_delta(cn, PFV, VA)
        over = np.abs(post) - limits
        j = int(np.argmax(over))
        if over[j] > AUDIT_TOL:
            violated.append((cn, bkeys[j], float(abs(post[j])), float(limits[j])))
    return violated


def worker(snapshot, case_path, candidates, task_q, result_q):
    """Persistent per-snapshot worker: build once, then serve streamed tasks."""
    logging.getLogger("egret").setLevel(logging.WARNING)
    m, md, solver, cap_pu, names, base_mva = build_persistent(snapshot, case_path, candidates)
    sid = snapshot["timestamp"]
    screened = set(md.data["elements"]["contingency"])
    audit_ptdf = None
    while True:
        item = task_q.get()
        if item is None:
            return
        kind, cid, config = item
        mus = solve_config(m, md, solver, config, cap_pu, names, base_mva)
        if kind == "audit":
            if audit_ptdf is None:
                audit_ptdf = build_audit_ptdf(md)
            result_q.put((cid, sid, audit_dropped(m, audit_ptdf, screened)))
        else:
            overload = thermal_violations(m, base_mva)          # {constraint: overload MW}
            shed = float(pyo.value(m.p_load_shed)) * base_mva   # system-wide fallback shed MW
            result_q.put((cid, sid, mus, overload, shed))


class ParallelTarget:
    """Manager over snapshot-pinned warm-started SCDCOPF workers."""

    def __init__(self, case=str(CASE), queue=QUEUE, collection=COLLECTION):
        self.case = case
        self.candidates = json.loads(Path(queue).read_text())["candidates"]
        self.snapshots = json.loads(Path(collection).read_text())["snapshots"]
        self.ctx = mp.get_context("spawn")
        self.workers, self.task_qs, self.result_q = [], [], None

    def start(self) -> None:
        self.result_q = self.ctx.Queue()
        for s in self.snapshots:
            tq = self.ctx.Queue()
            p = self.ctx.Process(target=worker,
                                 args=(s, self.case, self.candidates, tq, self.result_q))
            p.start()
            self.workers.append(p)
            self.task_qs.append(tq)

    def _broadcast(self, kind: str, configs) -> None:
        """Send one task per (config, snapshot) to the workers."""
        for cid, cfg in enumerate(configs):
            for tq in self.task_qs:
                tq.put((kind, cid, list(cfg)))

    def evaluate_detailed(self, configs):
        """Load-averaged mu per config AND the per-snapshot peak, in one broadcast.

        Returns (averages, snap_max): averages[cid] = {constraint: mean mu over
        snapshots} (the important quantity); snap_max[cid] = list of
        (snapshot_id, max_mu, argmax_constraint, max_overload_mw, overload_line,
        load_shed_mw), one entry per snapshot -- the single-hour peak mu and its line
        (BEFORE averaging, for comparison to real per-hour shadow prices), plus the
        worst thermal overload the cap allowed and the system-wide fallback shed, which
        together flag any config the slimmed relief set could not fully serve.
        """
        n = len(self.snapshots)
        self._broadcast("solve", configs)
        totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        snap: dict[int, list] = defaultdict(list)
        for _ in range(len(configs) * n):
            cid, sid, mus, overload, shed = self.result_q.get()
            for k, v in mus.items():
                totals[cid][k] += v
            con, mu = max(mus.items(), key=lambda kv: kv[1], default=("none", 0.0))
            ov_line, ov_mw = max(overload.items(), key=lambda kv: kv[1], default=("none", 0.0))
            snap[cid].append((sid, mu, con, ov_mw, ov_line, shed))
        averages = [{k: v / n for k, v in totals[cid].items()} for cid in range(len(configs))]
        return averages, [snap[cid] for cid in range(len(configs))]

    def evaluate(self, configs) -> list[dict[str, float]]:
        """Load-averaged mu for each config; broadcasts to all snapshot workers."""
        return self.evaluate_detailed(configs)[0]

    def audit(self, configs) -> list[dict[str, tuple]]:
        """Dropped contingencies violated by each config (union over snapshots).

        Empty dict for a config means its screened result is provably exact.
        Each value: {contingency: (worst branch, |post-flow| pu, limit pu)}.
        """
        n = len(self.snapshots)
        self._broadcast("audit", configs)
        per: dict[int, dict[str, tuple]] = defaultdict(dict)
        for _ in range(len(configs) * n):
            cid, _sid, violated = self.result_q.get()
            for cn, bn, flow, lim in violated:
                if cn not in per[cid] or flow > per[cid][cn][1]:
                    per[cid][cn] = (bn, flow, lim)
        return [dict(per[cid]) for cid in range(len(configs))]

    def close(self) -> None:
        for tq in self.task_qs:
            tq.put(None)
        for p in self.workers:
            p.join()


def scalar_target(avg_mu: dict[str, float]) -> float:
    """The sampling target: max load-averaged mu over constraints (0 if none bind)."""
    return max(avg_mu.values(), default=0.0)


def promote_contingencies(names) -> list[str]:
    """Add named contingencies from the full case into the screened case; return
    the ones actually added (workers must be restarted to pick them up)."""
    case = json.loads(CASE.read_text())
    full = json.loads(FULL_CASE.read_text())["elements"]["contingency"]
    present = case["elements"]["contingency"]
    added = [c for c in names if c not in present]
    for c in added:
        present[c] = full[c]
    CASE.write_text(json.dumps(case))
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=0,
                        help="also evaluate this many random Bernoulli-p configs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audit", action="store_true",
                        help="audit each config's dropped contingencies for violations")
    args = parser.parse_args()

    target = ParallelTarget()
    n = len(target.candidates)
    configs = [[0] * n, [1] * n]
    labels = ["base", "all-on"]
    rng = random.Random(args.seed)
    for i in range(args.random):
        p = [c["p"] for c in target.candidates]
        configs.append([int(rng.random() < pi) for pi in p])
        labels.append(f"rand{i}")

    target.start()
    t0 = time.time()
    results = target.evaluate(configs)
    dt = time.time() - t0
    for label, avg in zip(labels, results):
        print(f"{label:<8} scalar target {scalar_target(avg):8.2f} $/MWh  "
              f"({len(avg)} binding constraints)")
    print(f"\n{len(configs)} configs x {len(target.snapshots)} snapshots in "
          f"{dt:.1f}s  ({dt / len(configs):.2f}s per config)")

    if args.audit:
        t0 = time.time()
        audits = target.audit(configs)
        missing = sorted({cn for a in audits for cn in a})
        print(f"\naudit ({time.time() - t0:.1f}s): {len(missing)} dropped "
              f"contingency(ies) violated across configs")
        for label, a in zip(labels, audits):
            if a:
                worst = max(a.items(), key=lambda kv: kv[1][1] - kv[1][2])
                print(f"  {label:<8} {len(a)} violated; worst ctg {worst[0]} "
                      f"flow {worst[1][1]:.3f} > limit {worst[1][2]:.3f} pu")
    target.close()


if __name__ == "__main__":
    main()
