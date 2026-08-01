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
from cases import registry
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
uv run python -m cases.build_ieee_queue   --case case30_queue
uv run python -m cases.calibrate_ratings  --case case30_queue

# 2. screen the N-1 set for fast interactive runs (the ground-truth sweep skips this)
uv run python -m cases.screen_contingencies --case case30_queue

# 3. exhaustive ground truth -> results/<case>/mu_cache/
uv run python -m sweeps.enumerate_configs --case case30_queue --workers 16

# 4. reduce to the common caches every consumer reads
uv run python -m sweeps.build_target_cache --case case30_queue
uv run python -m sweeps.build_dual_cache   --case case30_queue

# 5. estimate, describe, draw
uv run python -m sampling.run_experiments  --case case30_queue
uv run python -m figures.tail.per_line_survivor --case case30_queue
uv run python -m figures.descriptors
```

Steps 1–2 are idempotent.  Step 3 skips parts that already exist, so an interrupted sweep
resumes; `--shard i/n` splits it across cluster jobs (see `scripts/sweep_vacc.slurm`).

## The data contract between stages

Every case follows the same layout, which is what lets a stage consume the previous one's
output with no per-case glue:

```
data/<case>.json                          full case, every N-1 contingency
data/<case>_screened.json                 contingency set filtered to the binders
data/<case>_candidates.json               queue candidates and their build probabilities
data/<case>_relief_set.json               slim ORDC relief buses            (optional)

results/<case>/mu_cache/part_*.parquet    long: id, n_on, constraint, mu
results/<case>/snapmax_cache/  or  stress_cache/    what the relief mechanisms bought
results/<case>/dual_cache.parquet         wide: id x {mu, flow, p_gen, lmp}
results/<case>/target_cache.parquet       common: max_mu sum_mu one_mu one_lmp n_on ...
results/<case>/experiment_{samples,curves}/b<budget>.parquet
results/<case>/static_descriptors_<base>.parquet
```

Configuration ids are MSB-first over the candidate order in the queue file, everywhere.

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
  `study.py` scores them, `figures/descriptors.py` draws the result.
- **`figures/`** — nothing here computes a result; each module reads a cache and renders it.

## Reproducibility note

The SCDCOPF sweeps are **not bit-reproducible**.  Congestion duals are frequently
degenerate, and with lazy PTDF a constraint only carries a dual once it has been added to
the model — which depends on the order a warm worker happened to see configurations in.
Re-running `case14_queue` twice changes about 8 of 66,000 dual rows; re-running against a
cache built on a different day changes about 120 rows and roughly 0.2% of per-configuration
`max_mu` values, occasionally by a large amount.  This predates the current layout (the
pre-reorganization code diverges from the committed caches by the same amount).  Treat a
committed `mu_cache` as *the* ground truth for a case rather than regenerating it casually.
