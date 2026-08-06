# src — one package per workflow stage

The project asks one question over and over, on five testbeds: **given a probability
distribution over which queue generators get built, what is the probability that line `l`
ends up severely congested?**  Answering it is always the same pipeline, and each stage is
one package here.

```
cases/       define a testbed and screen its N-1 set   ->  data/<case>*.json
  |
model/       one configuration -> congestion duals mu
  |
sweeps/      many configurations -> caches            ->  results/<case>/*
  |
  +-- sampling/      estimate the tail from a sample, score against exact truth
  +-- descriptors/   try to predict the tail from one solve instead (it fails)
  |
figures/     draw all of it
```

Alongside those: `utils/` (low-level helpers), `diagnostics/` (one-off probes),
`sensitivity/` (the barrier-sensitivity research thread), `ercot/` (an independent
empirical study of real ERCOT congestion, which motivates the modeling recipe), and
`_archive/` (superseded code, kept readable, imported by nothing).

## The case registry

`cases/registry.py` is the single source of truth for what a case is: its family, its
physical model, where every input and cache lives, and the modeling recipe it is solved
under.  **Every entry point takes `--case <name>`** and reads the rest from the record, so
no script hardcodes a filename and no case needs custom code in a downstream stage.

```python
from src.cases import registry
case = registry.get("case30_queue")
case.screened_file      # data/case30_queue_screened.json
case.mu_cache           # results/case30_queue/mu_cache
case.recipe             # Recipe(emergency_factor=1.2, shadow_price_cap=3500.0, ...)
case.candidates()       # queue candidates, uniform schema whatever the source file
case.build_p()          # per-candidate build probabilities
```

The eight registered cases:

| case | buses | model | queue | ground truth |
|---|---|---|---|---|
| `case14_gen_per_bus` | 14 | DCOPF | 14 (no incumbents) | exhaustive `2^14` |
| `case30_gen_per_bus` | 30 | DCOPF | 20 (no incumbents) | exhaustive `2^20` |
| `case14_queue` | 14 | N-1 SCDCOPF | 14 | exhaustive `2^14` |
| `case30_queue` | 30 | N-1 SCDCOPF | 20 | exhaustive `2^20` |
| `case118_queue` | 118 | N-1 SCDCOPF | 20 | exhaustive `2^20` |
| `activsg2000_original` | 2000 | N-1 SCDCOPF | 20 | exhaustive `2^20` |
| `activsg2000_2025` | 2000 | N-1 SCDCOPF | 20 | exhaustive `2^20` |
| `activsg2000_original_v12` | 2000 | N-1 SCDCOPF | 20 | exhaustive `2^20` |

The two `gen_per_bus` cases predate the incumbent-fleet design and are kept for continuity
only; see `notes/paper_plan.md` §3.

## Running a case end to end

The project is an editable install rooted at `src/`, so every module imports by name and
runs with `-m` from anywhere.

```bash
# 1. build the testbed and its queue          (once per case)
uv run python -m src.cases.build_ieee_queue   --case case30_queue
uv run python -m src.cases.calibrate_ratings  --case case30_queue

# 2. screen the N-1 set down to the contingencies that bind, at the case's own recipe
uv run python -m src.cases.screen_contingencies --case case30_queue

# 3. exhaustive ground truth -> results/<case>/mu_cache/
uv run python -m src.sweeps.enumerate_configs --case case30_queue --workers 16

# 4. reduce to the common caches every consumer reads
uv run python -m src.sweeps.build_target_cache --case case30_queue
uv run python -m src.sweeps.build_dual_cache   --case case30_queue

# 5. prove step 2 did not throw away a binder -- read this before trusting step 3
uv run python -m src.sweeps.audit_screening    --case case30_queue

# 6. estimate, describe, draw
uv run python -m src.sampling.run_experiments  --case case30_queue
uv run python -m src.figures.tail.per_line_survivor --case case30_queue
uv run python -m src.figures.descriptors
```

Steps 1–2 are idempotent.  Step 3 skips parts that already exist, so an interrupted sweep
resumes, and `--shard i/n` splits it across jobs.  Step 5 is not optional: a screened set
that misses a binding contingency produces duals that are wrong, not merely imprecise, and
nothing earlier in the pipeline detects it.

## Running a sweep on the cluster

`src/sweeps/submit_cluster.py` opens with a settings block — which cases, the partition,
cores, shards, walltimes, and three switches (`SCREENED`, `DRY_RUN`, `TEST`).  Edit it and
run it; `src/sweeps/vacc.slurm` beside it is the batch script it submits.

```bash
uv sync                                          # login node, once
uv run python -m src.sweeps.submit_cluster       # with TEST = True first, ~5 min
uv run python -m src.sweeps.submit_cluster       # then TEST = False
```

Each case becomes a chain of dependent jobs, and a preparation step is submitted only when
its output file is missing:

```
select_relief_set -> screen_contingencies -> sweep (array of SHARDS) x LINKS
                                                 -> build_target_cache -> audit_screening
```

Both preparation steps depend on where the queue sits, so **a new queue gets its own relief
set and its own screened contingency set** — neither is inherited from another case, even
one sharing the same network and fleet.  Screening runs at the case's own recipe, which
includes its relief set, so it waits on relief selection.  The sweep links are chained with
`--dependency=afterany` so a shard stopped at the walltime is continued by the next; cached
chunks are skipped, and a case whose enumeration is already complete gets only the last two
jobs.

The contingency set dominates the cost.  Measured on ACTIVSg2000, one worker per snapshot,
one Gurobi thread, warm-started re-solves: 403 ms/config over the full 2756-contingency set
against 12 ms over v12's 33-contingency screened set, and 0.89 GB against 0.24 GB resident
per worker.  Screening is what makes a 2^20 enumeration affordable; the audit is what makes
it exact, which is why the chain ends there rather than at the sweep.

### Reading the audit

`sweeps/audit_screening.py` re-solves a selection of configurations — the largest `max_mu`,
plus a uniform sample — and recomputes every *dropped* contingency's post-outage flows at
the solved dispatch.  A configuration with no violation is proved exact: adding constraints
the solution already satisfies cannot move the optimum, so the screened duals are the full
problem's duals.  A configuration with a violation is not slightly off; its sweep entry is
wrong and the case needs re-screening.

This is not hypothetical.  `activsg2000_original_v12` ships a 33-contingency screened set
whose record carries no recipe, because it was screened before the factor-1.2 recipe existed
— and the audit fails on it.  On the 2025-02-19 06:00 snapshot, contingency 457 overloads
branch 449 by 0.36%; putting that one contingency back in the model prices it at the $3500
cap, moves the objective by 0.11%, and moves that snapshot's `sum_mu` from 25288 to 28770.
For comparison, `activsg2000_2025`, screened by the current recipe-aware screener, keeps 104
contingencies where v12 keeps 33.

## The data contract between stages

Every case follows the same layout, which is what lets a stage consume the previous one's
output with no per-case glue:

```
data/<case>.json                          full case, every N-1 contingency
data/<case>_screened.json                 contingency set filtered to the binders
data/<case>_candidates.json               queue candidates and their build probabilities
data/<case>_relief_set.json               slim ORDC relief buses            (optional)

results/<case>/
  mu_cache/part_*.parquet       long   id, n_on, constraint, mu          [the raw sweep]
  snapmax_cache/part_*.parquet  long   id, n_on, snapshot, max_mu, max_line,
                                       max_overload_mw, overload_line, load_shed_mw
  dual_cache.parquet            wide   id x (group, element), `mu` group always present
  target_cache.parquet          wide   max_mu sum_mu one_mu one_lmp n_on
                                       max_mu_single driver
  lmp_cache.parquet             wide   lmp_range lmp_std lmp_maxabs      (optional)
  static_descriptors_<base>.parquet
  experiment_{samples,curves}/b<budget>.parquet
```

Configuration ids are MSB-first over the candidate order in the queue file, everywhere,
and every per-configuration frame is indexed by `id`.

Two conventions worth knowing:

- **`dual_cache` carries `mu` always and other groups optionally.** The gen-per-bus sweeps
  recorded the whole solution, so theirs also has `flow`, `p_gen`, `lmp`, `va`, `nu`,
  `load`, `mu_pmax`, `mu_pmin`; the security-constrained sweeps keep only `mu`.
  `Case.dual_cache_groups()` reports which a case has, and `descriptors/compute.py`
  branches on that rather than on the case family.
- **`snapmax_cache` has one schema for both parallel axes.** A snapshot-parallel case has
  one row per (configuration, snapshot) holding that hour's peak mu *before* averaging; a
  single-load-level case has one row per configuration with `snapshot` null.

Everything except `mu_cache` is derived and rebuildable. Earlier layouts had the Texas
cases carrying the same reduction three times — an `S_cache` (whose `S` and `driver_mu`
were the same column), a partial `config_cache` with that schema, and an `lmp_cache` that
repeated all four and added the LMP spread — plus `activsg2000_original`'s caches under
`results/activsg2000/original_case_source/`. That is all folded into `target_cache` and the
standard per-case directory now.

## Package notes

- **`cases/`** — `registry.py`, the three builders, and one generic screener.
  `cases/texas/` holds the one-time ACTIVSg2000 construction from ERCOT and MATPOWER
  sources; see its README.
- **`model/`** — the only place the physical model lives.  `scdcopf.build_persistent`
  builds the model once with every candidate present and `PersistentModel.solve` applies a
  configuration by toggling bounds and warm re-solving, which is what makes `2^20`
  affordable.  `evaluator.for_case` picks the parallel axis from the case: config-parallel
  for single-load-level cases, snapshot-parallel (averaging mu over snapshots) for Texas.
- **`sweeps/`** — `enumerate_configs` for both families and both axes; the `build_*_cache`
  reducers.
- **`sampling/`** — `problem.py` exposes the nominal law only as a weight vector, a pmf
  lookup, and a categorical sampler, so no estimator assumes it is a product law.  See its
  README for the experiment matrix.
- **`descriptors/`** — Claim 2 of the paper.  `compute.py` produces the descriptors,
  `study.py` scores them, `figures/descriptors.py` draws the result.  `additivity.py` tests
  the other cheap replacement for the sweep -- a linear model in the queue bits -- and finds
  it fits only where one candidate carries the target, and never reaches the tail.
- **`figures/`** — nothing here computes a result; each module reads a cache and renders it.

## Reproducibility note

The SCDCOPF sweeps are **not bit-reproducible**, for two separate reasons.  They are
distinguished by whether the *objective* moves when the same configuration is solved on a
cold model and on a warm one.

**Dropped constraints** (objective moves).  With lazy PTDF a constraint enters the model
only once it is seen violated by more than EGRET's `rel_flow_tol = 1e-5`.  A constraint over
its limit by less than that never enters, carries no dual, and reports `mu = 0`, and the
solution is slightly infeasible — so the objective differs from a run in which some earlier
configuration had already forced that constraint in.  Which constraints are resident depends
on the order a warm worker happened to see configurations in.  This is what moves
`case14_queue`: the 0.2% of configurations that differ between two sweeps are all of this
kind, and they move a lot (worst case 449 $/MWh).  Texas takes a dropped constraint on
essentially every configuration, but never one that reaches the peak, so `max_mu` is stable
to below 1e-3 $/MWh.

**Non-unique duals** (objective identical, `mu` differs).  The multiplier vector is pinned
down only when the active constraint gradients are linearly independent over the free
generators.  Measured per configuration as the rank of the tight PTDF rows (base and
post-contingency, plus the power-balance row) restricted to the off-bound generators:
`case14_queue` is always full rank, `activsg2000_2025` is always deficient by 4–6 rows out
of ~46, and `case30_queue` is deficient on about a quarter of configurations.  Deficiency is
necessary but not sufficient — nothing was observed to move with a full-rank active set, and
most deficient configurations are stable anyway, because a wider optimal dual face need not
be wide in the direction of `max_b mu_b`.  `case30_queue` is the case that actually moves
(~8% of configurations, worst 32.8 $/MWh) because it is the most cornered: a median of 7
tight constraints against 6 free generators, versus 46 against 78 on Texas.  Nothing about
this is a property of the solver — dual simplex, primal simplex, and barrier with crossover
give identical duals on a fresh model — nor of tied generator costs, which produce multiple
optimal *dispatches* rather than multiple prices.

**Worker count is part of this, and it is the second reason, not the first.**
`SnapshotParallel` runs several workers per snapshot so a sweep fills its whole
allocation.  Each replica sees a different subset of the configurations, so their resident
constraint sets diverge — but what that changes is which of several tied optimal dual
vectors comes back, not the answer.  Two measurements on `activsg2000_original_v12`,
screened set:

- *One process, two histories.*  Two models solve the same 40 configurations; one has ten
  extra configurations in front of them.  Relative objective difference: median 1.4e-10,
  max 4.7e-9, never above 1e-6.  Worst single dual difference 0.17 $/MWh.  One
  configuration in 40 prices a constraint the other does not.
- *Real evaluator, 1 worker per snapshot against 2, 64 configurations.*  958 constraint entries
  disagree; 946 of them are present in both runs with a different value and only 12 are
  missing from one.  The worst disagreements come in pairs of equal magnitude on the same
  contingency (`ctg:768:776` and `ctg:768:777`, 34.6 $/MWh each) — dual mass moving between
  two tied rows.  It is mass moving rather than appearing: per-constraint differences reach
  34.6 while `sum_mu` differs by at most 0.043 and `max_mu` by at most 0.037, against a
  `max_mu` range of 915–2018.

So the dispatch, the cost, and both scalar targets are effectively independent of how many workers share a snapshot; a
per-line study off `dual_cache` that cares about a specific constraint on a specific
contingency is the work that would notice.  Set `CPUS` to the snapshot count in
`submit_cluster` to get back one worker per snapshot.

Extra workers per snapshot are not free throughput either.  Going 8 → 16 workers measured 29.9 → 22.2
ms/config, a 1.34x speedup for 2x the workers — but that was on a 10-physical-core laptop,
where 16 single-threaded solvers are oversubscribed.  Scaling on a 40-core node is
unmeasured.

Concretely: re-running `case14_queue` twice changes about 8 of 66,000 dual rows; re-running
against a cache built on a different day changes about 120 rows and roughly 0.2% of
per-configuration `max_mu` values.  This predates the current layout (the pre-reorganization
code diverges from the committed caches by the same amount).  Treat a committed `mu_cache`
as *the* ground truth for a case rather than regenerating it casually.
