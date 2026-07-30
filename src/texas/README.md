# ACTIVSg2000 congestion-dual tail study

This package studies the distribution of transmission **congestion dual variables (μ)**
over the binary space of interconnection-queue generators on the ACTIVSg2000 synthetic
Texas grid (2000 buses, 3206 branches, 544 base generators). 
A *configuration* `x` is a length-20 bit vector: each queue generator is either online or not.
The nominal law is a product of independent Bernoullis (each candidate online with its 
calibrated build probability `p`). The scalar quantity of interest is

```
S(x) = max over monitored constraints of ( load-averaged μ_constraint )
```

the peak load-averaged congestion shadow price a configuration produces, averaged over a
fixed set of 8 high-congestion ERCOT load snapshots. The research question is the high
tail of `S` under the nominal law — which rare queue outcomes drive extreme congestion —
and how well rare-event samplers recover that tail versus the exact `2^20` enumeration.

## Physical model

Every evaluation is one **N-1 security-constrained DC optimal power flow (SCDCOPF)**
solved with `egret.models.scopf.create_scopf_model` (lazy PTDF/LODF contingency
constraints, `gurobi_persistent`). Congestion μ is the dual of a thermal limit. Three
ingredients make the μ distribution realistic rather than degenerate — see
`notes/ercot_price_formation.md` and the memory notes for the derivation:

1. **Emergency factor 1.2** (`set_emergency_factor`): the post-contingency thermal limit
   is `1.2 × long_term rating`. At factor 1.0 the duals increase to millions; ~1.2 is the 
   usable band where a handful of constraints bind at realistic magnitudes. An emergency 
   factor is used in practice, allowing modest overloads in response to an outage that can 
   be resolved by corrective action. The case JSONs all carry `rating_emergency = 1.5 ×
   long_term` (baked in by `convert_model.EMERGENCY_FACTOR`); the 1.2 is applied at
   model-build time by `set_emergency_factor`, so it never appears in the data files. Any
   screening or diagnostic that reads a case JSON without that override is solving at 1.5.
2. **ORDC-shaped nodal relief** (`add_lostload_generators_ordc`, `$500 → $5000`, 8 blocks):
   a rising piecewise relief price at load buses, mirroring ERCOT's ORDC scarcity adder,
   so shallow shed is charged near the floor and μ spreads over a range instead of pinning
   at a single VOLL spike.
3. **$3500 shadow-price cap** (`soft_limits.enable_soft_limits`): ERCOT enforces a Max
   Shadow Price — a constraint whose dual would exceed the cap is allowed to *overload*
   (soft slack, penalty `cap × baseMVA`) rather than pay unbounded redispatch. This bounds
   every reported μ at exactly `$3500`. Price ordering is `overload $3500 < load-shed $5000
   < system slack $9000`, so the model overloads a line before shedding load before the
   last-resort slack.

**Slim relief.** A full relief generator at every load bus enlarges the LP ~5× (an
inactive relief column is still factorized on every warm re-solve). Only 9 of 1125 load
buses ever shed, so relief is restricted to those. The set is a sweep *input*: it lives in
`data/<case>_relief_set.json`, named by the case's `CASE_REGISTRY` entry and loaded by
default. The full sweep additionally records per configuration the worst
line overload and the system-wide shed (`snapmax_cache`), so any configuration that needed
a bus outside the 9 would show up as a large overload or nonzero shed for a full-relief
recheck. Over the full `2^20` sweep the max overload was 5.64 MW and no configuration shed
— the slim set is proven complete.

## Recipe constants and knobs

All in `sample_harness.py`:

| constant | value | meaning |
|---|---|---|
| `EMERGENCY_FACTOR` | 1.2 | post-contingency limit = factor × long_term rating |
| `ORDC_C_MIN`, `ORDC_C_MAX` | 500, 5000 | relief price floor/ceiling ($/MWh) |
| `ORDC_BLOCKS` | 8 | relief piecewise blocks (`HARNESS_ORDC_BLOCKS`) |
| `SHADOW_PRICE_CAP` | 3500 | μ cap ($/MWh) |
| `SOLVER_THREADS` | 0 (Gurobi default) | Gurobi threads per worker (`HARNESS_THREADS`) |
| `RELIEF_BUSES` | case's `data/<case>_relief_set.json` | slim relief bus set (`HARNESS_RELIEF_SET` overrides with another file, or `all` for every load bus) |

The harness pins 8 worker processes, one per load snapshot; each builds its SCDCOPF once
with all 20 candidates present and toggles generator bounds per configuration (warm
start). With 8 workers on 20 cores, set `HARNESS_THREADS` to avoid oversubscription.

## Data flow

```
                         (one-time model + data prep)
  MATPOWER + .aux ──convert_model──►  data/activsg2000.json
                                            │
                        screen_contingencies (13 binding N-1)
                                            ▼
                                data/activsg2000_screened.json ─────┐
  ERCOT API ─fetch_load_snapshots─► ..._load_snapshots.csv          │
                     └build_load_snapshots─► ..._load_collection.json (8 snapshots)
  ERCOT queue ─build_queue_distribution─► ..._queue20.json (20 candidates + p)
                                            │                       │
        ┌───────────────────────────────────┴───────────────────────┘
        ▼   (model machinery: imported, not run directly)
   load_averaged_target.py   soft_limits.py   sample_harness.py::ParallelTarget
        │
        ▼   (one-time calibration / selection)
   check_relief_curve_mu   sweep_capped_target   select_relief_set ─► data/<case>_relief_set.json
        │
        ▼   (exhaustive ground truth — the expensive step, ~5.6 h)
   enumerate_configs --mode full ─► results/activsg2000/mu_cache/       (id,n_on,constraint,μ)
                                    results/activsg2000/snapmax_cache/  (per-snapshot peak μ, overload, shed)
        │
        ▼   (derived datasets — rebuilt with consolidate/reconstruct/score below)
   consolidate_s_cache ─► S_cache.parquet          (id → n_on, S, driver, driver_mu)
   reconstruct_lmp     ─► lmp_cache.parquet         (S_cache + mu_cache → PTDF congestion-LMP spread)
   score_samplers      ─► tail_samples.parquet, tail_guiding_curves.parquet  (samplers vs exact S)

  (independent live-solve path — the method for cases too large to enumerate)
   sample_tail ─► tail_sample.npz  ──plot_tail──►  tail plot
```

## How to run

### One-time model and data prep
These are already done (their outputs are committed under `data/`); rerun only to rebuild
inputs.
```bash
uv run python src/texas/convert_model.py           # -> data/activsg2000.json
uv run python src/texas/screen_contingencies.py    # -> data/activsg2000_screened.json
# Screen at the recipe the sweep solves (factor 1.2 + ORDC + cap) instead of the JSON's
# native 1.5 -- required for any case a sweep runs on:
HARNESS_CASE=2025 uv run python src/texas/screen_harness_recipe.py
                                                   # -> data/activsg2000_2025_screened.json
source gridstatus_api_key.sh                        # ERCOT token; see src/grid_data/ercot_auth.py
uv run python src/texas/fetch_load_snapshots.py --n 8
uv run python src/texas/build_load_snapshots.py    # -> data/activsg2000_load_collection.json
uv run python src/texas/build_queue_distribution.py  # -> data/activsg2000_queue20.json
```

### Calibration / relief selection (one-time)
```bash
uv run python src/texas/check_relief_curve_mu.py   # flat vs ORDC relief, emergency-factor regime
uv run python src/texas/sweep_capped_target.py     # confirm the capped target is a good testbed
HARNESS_CASE=2025 uv run python src/texas/select_relief_set.py
                                                   # -> data/activsg2000_2025_relief_set.json
```

### The full sweep (exhaustive ground truth, ~5.6 h)
```bash
HARNESS_CASE=2025 HARNESS_THREADS=2 \
  uv run python src/texas/enumerate_configs.py --mode full
# -> results/activsg2000/{mu_cache,snapmax_cache}/  (resumable: existing part pairs are skipped)
```
Subset modes for sanity checks (no full enumeration): `--mode random --n 30000`,
`--mode leq_k --k 5`.

On a cluster (UVM VACC), `sbatch src/texas/sweep_vacc.slurm`. The job derives
`HARNESS_THREADS` from `--cpus-per-task`, stops itself just before the walltime, and
resubmits so the next link resumes from the parts already written. Measured footprint on
the 104-contingency 2025 case: ~2.8 GB resident (8 workers + manager), ~180 s per 2000-config
chunk at 2 threads/worker. **Gurobi licensing is the blocker to settle first** — the harness
has no non-Gurobi path, and a node-locked workstation license will not validate on a compute
node.

### Rebuild the derived datasets (fast; run in this order after any sweep)
```bash
uv run python -m texas.consolidate_s_cache   # mu_cache      -> S_cache.parquet
uv run python -m texas.reconstruct_lmp       # mu_cache + S  -> lmp_cache.parquet
uv run python -m texas.score_samplers        # S_cache       -> tail_samples.parquet, tail_guiding_curves.parquet
```
`consolidate_s_cache` and `score_samplers` are pure table work (seconds/minutes);
`reconstruct_lmp` factorizes the PTDF once (a few minutes).

### Rare-event sampling by live solves
This is the actual research method for the regime where `2^20` is too large to enumerate;
it drives the same `ParallelTarget`, no cache.
```bash
uv run python src/texas/sample_tail.py --budget 4000          # -> tail_sample.npz
uv run python src/texas/sample_tail.py --budget 4000 --audit  # certify top configs vs dropped contingencies
uv run python src/texas/plot_tail.py                          # plot tail_sample.npz
```

## Files

**Data prep**
- `convert_model.py` — MATPOWER + PowerWorld `.aux` → EGRET JSON with coordinates, ERCOT weather zones, emergency ratings, N-1 contingency set.
- `screen_contingencies.py` — reduce the full N-1 set to the ~13 that ever bind; drop-in screened case. Screens through `snapshot_mu`, i.e. at the JSON's native 1.5 rating with no relief and no cap.
- `screen_harness_recipe.py` — the same screen driven through `ParallelTarget`, so the kept set matches the recipe the sweep solves (factor 1.2 + ORDC relief on the case's own relief buses + $3500 cap). Case-parametric via `HARNESS_CASE`; supersedes `screen_original_v12.py`.
- `fetch_load_snapshots.py` — pull ERCOT actual zonal load for the high-congestion hours (needs an ERCOT token).
- `build_load_snapshots.py` — turn those into per-zone load scale factors (8 snapshots).
- `build_queue_distribution.py` — select 20 ERCOT queue projects, match to buses geographically, set each candidate's Bernoulli `p`.

**Model machinery (imported by the drivers)**
- `load_averaged_target.py` — `add_queue_generators`, `scale_loads_by_zone`, `add_lostload_generators_ordc`, `ordc_relief_curve`.
- `soft_limits.py` — `enable_soft_limits` (the μ cap via EGRET's lazy violation slacks), `thermal_violations` (overload MW).
- `sample_harness.py` — `ParallelTarget`: 8 snapshot-pinned warm-started SCDCOPF workers; `build_persistent`, `worker`, `evaluate`/`evaluate_detailed`, `scalar_target`, the recipe constants and env knobs, and `build_audit_ptdf`.

**Calibration / selection**
- `check_relief_curve_mu.py` — flat vs ORDC relief and the emergency-factor regime, all-off diagnostic.
- `sweep_capped_target.py` — small capped-target sweep confirming `S` is a good combinatorial testbed.
- `select_relief_set.py` — collect the buses that ever shed under full relief, validate the slim set reproduces `S` on held-out configs → `data/<case>_relief_set.json`. (`identify_relief_set.py` is the earlier timing/certification variant → `relief_set.json`.)

**Sweep and derived datasets**
- `enumerate_configs.py` — drive configs through the workers; `--mode full` streams `mu_cache` + `snapmax_cache`; subset modes cache `config_cache.parquet`.
- `consolidate_s_cache.py` — reduce `mu_cache` to `S_cache.parquet` (peak load-averaged μ per config + its driving constraint).
- `reconstruct_lmp.py` — project the cached μ through the PTDF/LODF to per-config congestion-LMP spread → `lmp_cache.parquet` (no new solves).
- `score_samplers.py` — run the reusable `tail_sampling/` samplers as table lookups against the exact `S_cache` and score their tail estimates → `tail_samples.parquet`, `tail_guiding_curves.parquet`.

**Live sampling and utilities**
- `sample_tail.py` — subset-simulation tail estimate by live solves → `tail_sample.npz`.
- `plot_tail.py` — plot the estimated exceedance curve.
- `time_dcopf.py` — single-solve timing and congestion summary.

## Case inputs (`data/`)

A case is one `CASE_REGISTRY` entry in `sample_harness.py`, selected by `HARNESS_CASE`.
Every file below is an input: tracked in git, and all a fresh checkout needs to sweep.

| registry key | file | contents |
|---|---|---|
| `screened` | `<case>_screened.json` | the SCDCOPF the sweep solves (contingencies filtered to the kept set) |
| `full` | `<case>.json` (or `activsg2000.json`) | all 2756 contingencies, for `audit()` and for re-screening |
| `relief` | `<case>_relief_set.json` | the slim relief bus set, loaded by default |
| — | `<case>_contingencies_screened.json` | provenance: kept names + the recipe they were screened at |

Shared across cases: `activsg2000_queue20.json` (the 20 queue candidates and their
probabilities) and `activsg2000_load_collection.json` (the 8 load/renewable snapshots).

| case | fleet | contingencies | relief buses |
|---|---|---|---|
| `2025` | current (CDR additions through 2025) | 104 | 3 |
| `original_v12` | original ~2016 | 33 | 9 |

## Artifacts (`results/activsg2000/`)

| file | produced by | contents |
|---|---|---|
| `mu_cache/` | `enumerate_configs --mode full` | long-format load-averaged μ per config (id, n_on, constraint, μ) — **ground truth** |
| `snapmax_cache/` | same | per (config, snapshot) peak μ + line, worst overload MW, load shed MW (the missing-relief safety net) |
| `S_cache.parquet` | `consolidate_s_cache` | per config: `S`, `driver` constraint, `driver_mu`, `n_on` |
| `lmp_cache.parquet` | `reconstruct_lmp` | `S_cache` + congestion-LMP spread (`lmp_range`, `lmp_std`, `lmp_maxabs`) |
| `tail_samples.parquet` | `score_samplers` | sampler outputs (bits, weights, S) per method/replicate |
| `tail_guiding_curves.parquet` | `score_samplers` | tidy exceedance curves, estimated vs exact |
| `tail_sample.npz` | `sample_tail` | live subset-simulation exceedance curve + weighted sample |
| `original_case_source/` | — | pre-cap (factor-1.5, uncapped) `mu_cache` + `lmp_cache_old` — the live pinned source for the `activsg2000_original` case (do not delete; distinct setup from the top-level capped/tighter-factor sweep) |

Current sweep: all `2^20 = 1048576` configs, `S ∈ [125, 1531] $/MWh` (median 512, q99 1488),
0 at the cap; drivers `ctg:68:18` (3/4 of configs) and `ctg:329:330` (1/4).

