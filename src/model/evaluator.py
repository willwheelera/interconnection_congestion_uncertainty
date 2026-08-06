"""Parallel evaluation of many configurations against one case.

Building the SCDCOPF is expensive and solving a configuration against an already-built
model is cheap, so every worker builds once and then serves configurations off a queue.
Which axis the work is split along depends on the case:

  ConfigParallel     one load level, so `n_workers` identical models each take whole
                     configurations off a shared queue.  Used by the IEEE queue cases.
  SnapshotParallel   a collection of load snapshots that mu is averaged over, so each
                     configuration is broadcast to every snapshot.  Used by the Texas
                     cases, where a worker's model is pinned to its snapshot's scaled
                     loads and derated renewables.

`for_case` picks the right one from the case record and sizes it to the cores actually
available, so callers never choose.  Both expose the same `evaluate` /
`evaluate_detailed` interface and return results in the order the configurations were
given, so a sweep or a sampler is written once.
"""

import logging
import multiprocessing as mp
import os
from collections import defaultdict

from src.model.scdcopf import build_persistent, scalar_target  # noqa: F401  (re-exported)
from src.model.soft_limits import thermal_violations


def _worker(case, snapshot, screened, threads, task_q, result_q):
    """Build this worker's model once, then serve tasks until told to stop.

    A task is (kind, config id, config).  "solve" returns the duals plus the overload
    and shed the recipe's relief mechanisms had to buy; "audit" instead reports which
    screened-out contingencies the resulting dispatch violates.
    """
    logging.getLogger("egret").setLevel(logging.WARNING)
    pm = build_persistent(case, snapshot, screened=screened, threads=threads)
    sid = (snapshot or {}).get("timestamp")
    for kind, cid, config in iter(task_q.get, None):
        mus = pm.solve(config)
        if kind == "audit":
            result_q.put((cid, sid, pm.audit_dropped(case.case_file)))
        else:
            result_q.put((cid, sid, mus, thermal_violations(pm.m, pm.base_mva),
                          pm.load_shed()))


class _Pool:
    """Shared process management for the two evaluators."""

    def __init__(self, case, screened=True, threads=0):
        self.case = case
        self.screened = screened
        self.threads = threads
        self.candidates = case.candidates()
        self.ctx = mp.get_context("spawn")
        self.workers, self.result_q = [], None

    def _spawn(self, snapshot, task_q):
        p = self.ctx.Process(target=_worker,
                             args=(self.case, snapshot, self.screened, self.threads,
                                   task_q, self.result_q))
        p.start()
        self.workers.append(p)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()


class ConfigParallel(_Pool):
    """`n_workers` identical models pulling whole configurations off one shared queue."""

    def __init__(self, case, n_workers=8, screened=True, threads=1):
        super().__init__(case, screened, threads)
        self.n_workers = n_workers
        self.task_q = None

    def start(self):
        self.task_q, self.result_q = self.ctx.Queue(), self.ctx.Queue()
        for _ in range(self.n_workers):
            self._spawn(None, self.task_q)

    def evaluate_detailed(self, configs):
        """Per config: ({constraint: mu}, {constraint: overload MW}, shed MW), in order."""
        for cid, cfg in enumerate(configs):
            self.task_q.put(("solve", cid, list(cfg)))
        out = [None] * len(configs)
        for _ in range(len(configs)):
            cid, _sid, mus, overload, shed = self.result_q.get()
            out[cid] = (mus, overload, shed)
        return out

    def evaluate(self, configs):
        """Per config {constraint: mu}, in the order the configs were given."""
        return [r[0] for r in self.evaluate_detailed(configs)]

    def audit(self, configs):
        """Screened-out contingencies each config violates at its solved dispatch.

        An empty dict means that config's screened result is provably exact.  Each
        value is {contingency: (worst branch, |post-outage flow| pu, limit pu)}.
        """
        for cid, cfg in enumerate(configs):
            self.task_q.put(("audit", cid, list(cfg)))
        out = [None] * len(configs)
        for _ in range(len(configs)):
            cid, _sid, violated = self.result_q.get()
            out[cid] = {cn: (bn, flow, lim) for cn, bn, flow, lim in violated}
        return out

    def close(self):
        for _ in self.workers:
            self.task_q.put(None)
        for p in self.workers:
            p.join()


class SnapshotParallel(_Pool):
    """`replicas` workers per load snapshot; each configuration goes to every snapshot.

    One worker per snapshot uses only as many cores as the case has snapshots, which
    leaves most of a cluster node idle.  Replicas put several workers on each snapshot's
    queue so a chunk's configurations are shared between them; every (configuration,
    snapshot) pair is still produced exactly once, so the aggregation is unchanged.
    """

    def __init__(self, case, screened=True, threads=1, replicas=1):
        super().__init__(case, screened, threads)
        self.snapshots = case.snapshots()
        self.replicas = replicas
        self.task_qs = []

    def start(self):
        self.result_q = self.ctx.Queue()
        for s in self.snapshots:
            tq = self.ctx.Queue()
            self.task_qs.append(tq)
            for _ in range(self.replicas):
                self._spawn(s, tq)

    def _broadcast(self, kind, configs):
        for cid, cfg in enumerate(configs):
            for tq in self.task_qs:
                tq.put((kind, cid, list(cfg)))

    def evaluate_detailed(self, configs):
        """Load-averaged mu per config, and the per-snapshot detail behind it.

        Returns (averages, per_snapshot).  averages[cid] is {constraint: mean mu over
        snapshots}, the quantity of interest.  per_snapshot[cid] is one
        (snapshot id, max mu, argmax constraint, max overload MW, overload line, shed MW)
        tuple per snapshot -- the single-hour peak before averaging, which is what
        compares to real per-hour shadow prices, plus the overload the shadow-price cap
        allowed and the system-wide shed.  Together those two expose any configuration
        the slimmed relief set could not fully serve.
        """
        n = len(self.snapshots)
        self._broadcast("solve", configs)
        totals = defaultdict(lambda: defaultdict(float))
        snap = defaultdict(list)
        for _ in range(len(configs) * n):
            cid, sid, mus, overload, shed = self.result_q.get()
            for k, v in mus.items():
                totals[cid][k] += v
            con, mu = max(mus.items(), key=lambda kv: kv[1], default=("none", 0.0))
            ov_line, ov_mw = max(overload.items(), key=lambda kv: kv[1], default=("none", 0.0))
            snap[cid].append((sid, mu, con, ov_mw, ov_line, shed))
        averages = [{k: v / n for k, v in totals[cid].items()} for cid in range(len(configs))]
        return averages, [snap[cid] for cid in range(len(configs))]

    def evaluate(self, configs):
        """Load-averaged mu per config; broadcasts to every snapshot worker."""
        return self.evaluate_detailed(configs)[0]

    def audit(self, configs):
        """Screened-out contingencies each config violates, unioned over snapshots.

        An empty dict means that config's screened result is provably exact.  Each
        value is {contingency: (worst branch, |post-outage flow| pu, limit pu)}.
        """
        n = len(self.snapshots)
        self._broadcast("audit", configs)
        per = defaultdict(dict)
        for _ in range(len(configs) * n):
            cid, _sid, violated = self.result_q.get()
            for cn, bn, flow, lim in violated:
                if cn not in per[cid] or flow > per[cid][cn][1]:
                    per[cid][cn] = (bn, flow, lim)
        return [dict(per[cid]) for cid in range(len(configs))]

    def close(self):
        for tq in self.task_qs:
            for _ in range(self.replicas):
                tq.put(None)
        for p in self.workers:
            p.join()


def available_cores():
    """Cores this process may actually use: Slurm's allocation when set, else affinity."""
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(allocated) if allocated else len(os.sched_getaffinity(0))


def for_case(case, n_workers=None, screened=True, threads=None):
    """The evaluator the case's parallel axis calls for, sized to the cores available.

    Leaving `n_workers` and `threads` unset fills the allocation with single-threaded
    workers, which is what a sweep wants: these LPs parallelize far better over
    configurations than inside one solve.  Both can be set to override that.
    """
    cores = available_cores()
    if case.parallel == "config":
        n = cores if n_workers is None else n_workers
        print(f"evaluator: {n} config-parallel workers on {cores} cores", flush=True)
        return ConfigParallel(case, n, screened, threads=1 if threads is None else threads)

    n_snapshots = len(case.snapshots())
    budget = cores if n_workers is None else n_workers
    replicas = max(1, budget // n_snapshots)
    print(f"evaluator: {n_snapshots} snapshots x {replicas} replicas = "
          f"{n_snapshots * replicas} workers on {cores} cores", flush=True)
    return SnapshotParallel(case, screened, threads=1 if threads is None else threads,
                            replicas=replicas)
