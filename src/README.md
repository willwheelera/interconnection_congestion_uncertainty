# src — one package per pipeline stage

The project asks one question, per testbed: **given a probability distribution over which
queued generators get built, what is the distribution of the congestion shadow price `μ` on
each line and contingency?**  Answering it is always the same pipeline, and each stage is one
package here.

```
cases/       define a testbed and screen its N-1 set   ->  data/<case>/*
  |
model/       one build-out configuration -> congestion duals mu
  |
sweeps/      many configurations -> caches             ->  results/<case>/*
  |
sampling/    estimate the distribution from a sample (direct Monte Carlo),
  |          and for the full queue draw configurations by MCMC
  |
figures/     draw all of it
```

## The case registry

`cases/registry.py` is the single source of truth for what a case is: its physical model,
where every input and cache lives, and the modeling recipe it is solved under.  **Every entry
point takes `--case <name>`** (a case name, a substring of one, or `all`) and reads the rest
from the record, so no script hardcodes a filename and no case needs custom code in a
downstream stage.

```python
from src.cases import registry
case = registry.get("case30_queue")
case.screened_file      # data/case30_queue/screened.json
case.mu_cache           # results/case30_queue/mu_cache
case.recipe             # Recipe(emergency_factor=1.2, shadow_price_cap=3500.0, ...)
case.candidates()       # queue candidates, uniform schema whatever the source file
case.build_p()          # per-candidate build probabilities
```

The cases shipped on this branch:

| case | buses | model | queue | ground truth |
|---|---|---|---|---|
| `case14_queue` | 14 | N-1 SCDCOPF | 14 | exhaustive `2^14` |
| `case30_queue` | 30 | N-1 SCDCOPF | 20 | exhaustive `2^20` |
| `activsg2000_v12_lev0a3` … `lev4a3` | 2000 | N-1 SCDCOPF | 20 | exhaustive `2^20` |
| `activsg2000_v12_levfull` | 2000 | N-1 SCDCOPF | 195 | MCMC (unenumerable) |

The paper's results come from `activsg2000_v12_lev3a3` (the enumerable leveraged-queue
build-out) and `activsg2000_v12_levfull` (the full 195-candidate queue).  The `lev*a3` cases
are five independent leveraged draws on the same 2000-bus network, fleet, and snapshots; the
two IEEE cases are small, fast examples of the identical pipeline.  The registry lists a few
more cases than ship with committed inputs; those are not included here.

## Running a case end to end

The project is an editable install rooted at `src/`, so every module imports by name and runs
with `-m` from anywhere.

```bash
# 1. the sweep's two inputs: the relief bus set, then the contingencies that bind
uv run python -m src.cases.prepare_case        --case case30_queue

# 2. exhaustive ground truth -> results/<case>/mu_cache/
uv run python -m src.sweeps.enumerate_configs  --case case30_queue --workers 16

# 3. reduce to the common caches every consumer reads
uv run python -m src.sweeps.build_target_cache --case case30_queue
uv run python -m src.sweeps.build_dual_cache   --case case30_queue

# 4. estimate and draw
uv run python -m src.sampling.run_experiments               --case case30_queue
uv run python -m src.figures.mu_distributions.line_mean_error --case case30_queue --transpose
```

The IEEE cases (`case14_queue`, `case30_queue`) are built from committed inputs by
`cases.build_ieee_queue` + `cases.calibrate_ratings` if you want to regenerate them; the
ACTIVSg2000 queues and network are committed under `data/`.  Step 1 is idempotent and skips
whichever of its two outputs already exists.  Step 2 skips parts that already exist, so an
interrupted sweep resumes; run one sweep job per case at a time, because a screened sweep
rewrites the contingency set as it goes.  There is no separate audit step: the sweep checks
every solve against the contingencies screening dropped, so the cache is exact when written.

The full-queue case `activsg2000_v12_levfull` cannot be enumerated, so it replaces step 2–3
with an MCMC sweep and its reducers:

```bash
uv run python -m src.cases.prepare_case              --case activsg2000_v12_levfull
uv run python -m src.sampling.mcmc_sweep             --case activsg2000_v12_levfull --law bernoulli --screened
uv run python -m src.sampling.consolidate_mcmc_duals --case activsg2000_v12_levfull --law bernoulli --rounds 200
uv run python -m src.sampling.solve_modal_config     --case activsg2000_v12_levfull --law bernoulli
uv run python -m src.figures.mu_distributions.line_sample_distribution --case activsg2000_v12_levfull --law bernoulli
```

## Running a sweep on the cluster

`src/sweeps/submit_cluster.py` opens with a settings block — which cases, the partition,
cores, walltimes, and switches (`SCREENED`, `DRY_RUN`, `TEST`).  Edit it and run it;
`src/sweeps/vacc.slurm` beside it is the batch script it submits.

```bash
uv sync                                          # login node, once
uv run python -m src.sweeps.submit_cluster       # with TEST = True first, ~5 min
uv run python -m src.sweeps.submit_cluster       # then TEST = False
```

Each case becomes a chain of dependent jobs: `prepare_case -> sweep x LINKS ->
build_target_cache`.  The preparation job is submitted only when one of its two outputs is
missing.  The sweep links are chained with `--dependency=afterany` so a job stopped at the
walltime is continued by the next; cached chunks are skipped.  There is one sweep job per case
at a time, never an array: a screened sweep rewrites `data/<case>/screened.json` as it runs,
and concurrent jobs would race for it.  A node is filled by the evaluator's worker pool
instead.

### How a screened sweep stays exact

Screening keeps the union of contingencies that bind across a spread of stress
configurations, so a configuration outside them can bind a contingency screening dropped —
and then its duals are wrong, not merely imprecise.  The sweep closes that gap itself:

1. **Every solve checks its own dispatch.**  A worker recomputes the post-outage flows of
   every dropped contingency at the point it just solved and returns the violated names with
   the duals.  No second solve — the dispatch is already in hand.
2. **The chunk boundary combines the workers.**  The parent unions the violated names as it
   drains the results, so a contingency any worker needs is discovered once for all of them.
3. **A discovery enlarges the set and redoes the chunk.**  The additions are written into
   `data/<case>/screened.json`, the workers are rebuilt against it, and the chunk is re-run
   before any parquet is written.
4. **Earlier chunks stand.**  A chunk that came back clean is already optimal for the full
   problem, so only the discovering chunk is redone.

The set grows monotonically inside the case's full N-1 list, so discoveries taper and the loop
terminates.  Verified on `case14_queue`: with the screened set cut to two contingencies, a
286-config sweep discovered 11 more in one round and reproduced a full-19-contingency sweep to
1.1e-13 on every one of the 1082 duals.

## The data contract between stages

Every case follows the same layout, which is what lets a stage consume the previous one's
output with no per-case glue:

```
data/<case>/queue.json        queue candidates and their build probabilities
data/<case>/laws/<name>.json  the stored nominal build-probability models (bernoulli, mrf_*)
data/<case>/screened.json     contingency set filtered to the binders        (built by prepare_case)
data/<case>/relief_set.json   slim ORDC relief buses                          (built by prepare_case)
data/activsg2000.json         the shared ACTIVSg2000 network (flat)
data/activsg2000_load_collection.json   the shared load snapshots (flat)

results/<case>/
  mu_cache/part_*.parquet       long   id, n_on, constraint, mu          [the raw sweep]
  snapmax_cache/part_*.parquet  long   per (config, snapshot): peak mu, worst overload, shed
  dual_cache.parquet            wide   id x (group, element); the `mu` group is the per-line table
  target_cache.parquet          wide   max_mu sum_mu one_mu one_lmp n_on max_mu_single driver
  mcmc/<law>/                   the MCMC draws and their reductions, for a sample-only case
  experiment_{samples,curves}/  the direct Monte Carlo runs scored against truth
```

Configuration ids are MSB-first over the candidate order in the queue file, and every
per-configuration frame is indexed by `id`.  Everything except `mu_cache` (and the MCMC
draws) is derived and rebuildable from it.

## Package notes

- **`cases/`** — `registry.py`, the IEEE-queue builder (`build_ieee_queue` +
  `calibrate_ratings`), the generic contingency screener, and `prepare_case` (relief set +
  screening).  `cases/texas/` is the one-time construction of the ACTIVSg2000 cases from the
  downloaded model and ERCOT sources (network, snapshots, leveraged queues), plus the relief-set
  selector `prepare_case` uses; see its README.
- **`model/`** — the only place the physical model lives.  `scdcopf.build_persistent` builds
  the model once with every candidate present and `PersistentModel.solve` applies a
  configuration by toggling bounds and warm re-solving, which is what makes `2^20` affordable.
  `evaluator.for_case` picks the parallel axis: config-parallel for the IEEE cases,
  snapshot-parallel (averaging mu over load snapshots) for ACTIVSg2000.  See `model/README.md`
  for the recipe (emergency factor, ORDC relief, shadow-price cap).
- **`sweeps/`** — `enumerate_configs` for both parallel axes; the `build_*_cache` reducers;
  `submit_cluster` + `vacc.slurm` for the cluster.
- **`sampling/`** — `problem.py` exposes the nominal law only as a weight vector, a pmf
  lookup, and a categorical sampler, so the estimator never assumes it is a product law;
  `run_experiments.py` is the direct Monte Carlo driver; `mcmc_sweep.py` and the
  `consolidate_*`/`solve_modal_config` reducers handle the unenumerable full queue.  See
  `sampling/README.md` for the build-probability models and the figure list.
- **`figures/`** — nothing here computes a result; each module reads a cache and renders it.
- **`ercot/`** — a small ERCOT public-API client (`ercot_auth`, `fetch_ercot`, `data_paths`)
  that the `cases/texas` snapshot fetchers use to pull real ERCOT load and renewable output.

## Reproducibility note

The SCDCOPF sweeps are **not bit-reproducible**.  Under lazy PTDF a constraint carries a dual
only once it has been seen violated, so which constraints are resident depends on the order a
warm worker happened to see configurations in; and where the active constraint gradients are
linearly dependent, the dual vector is not unique.  Neither moves the dispatch, the cost, or
the scalar targets `max_mu`/`sum_mu` by more than solver tolerance — but a per-line study off
`dual_cache` that cares about a specific constraint on a specific contingency can see a few
rows differ between two runs (`case30_queue` is the most affected, being the most cornered).
Nothing about this is a property of the solver.  Treat a committed `mu_cache` as *the* ground
truth for a case rather than regenerating it casually.
