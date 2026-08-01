# tail_sampling

Rare-event sampling of the congestion-dual tail: estimate `P[S >= b]` for a target
`S(x)` (e.g. `S(x) = max_b |mu_b|`) over generator-availability configurations `x`,
where high-`S` configs are rare under the nominal law. Each testbed has cached duals
for all `2^ngen` configs, so the nominal-weighted exceedance curve is known exactly by
enumeration and every sampler is scored against that ground truth.

The nominal law is **general**, not committed to iid Bernoulli. It is supplied as a
correlated pairwise **Markov random field (MRF)** (`mrf.py`, `distributions.py`); the
samplers never see it directly — `Problem`/`TargetProblem` exposes the law only as a
normalized weight vector over configs, an exact pmf lookup (`log_nominal_pmf`), and an
exact categorical sampler (`draw_nominal`), so **no method assumes the law has any
particular form**. Bernoulli-specific machinery (the `ascent_is` mixture *proposal*,
`cross_entropy`'s product fit) is one ansatz among several, not a claim about the law:
the IS estimators still query the true law generically, so unbiasedness does not depend
on the proposal's form. With coupling `J = 0` the MRF reduces exactly to iid Bernoulli.

## The experiment matrix

`run_experiments.py` is the single driver. For each case it sweeps the cross-product

    nominal law  x  sampler  x  ascent move  x  target  x  snapshot  x  budget

runs every cell `--n-rep` times, scores each weighted sample against the exact
enumerated truth, and writes two tidy parquet frames per case (see Outputs). The case
is **not** baked into the library: it is selected by name from the `CASES` registry in
`problem_setup.py`, and everything runs as table lookups against the cached enumeration.

| axis | values |
|------|--------|
| case | `case14_gen_per_bus`, `case30_gen_per_bus`, `activsg2000_original`, `activsg2000_2025`, `activsg2000_original_v12` |
| nominal law (`dist`) | `bernoulli` (J=0), `mrf_a` (J~0.04, ferromagnetic), `mrf_b` (J~N(0,0.08), mixed-sign), `mrf_c` (J~N(0,0.15), mixed-sign) |
| target | `max_mu`, `sum_mu`, `one_mu`, `one_lmp` (whichever the case's cache carries). `one_mu` is a single element, named per case in `build_target_cache.CASES` (table below); unnamed cases fall back to the globally most-congested element |
| snapshot | `multi` (averaged/nominal); `single` runs `max_mu` on the per-snapshot peak (Texas only) |
| ascent move | `all` always; `best` / `top5` only where a sensitivity proxy exists — in practice case14 only, since case30 has no `sensitivity_cache.parquet` (use `--proxy live` to get them there) |
| budget | solver-call budgets, e.g. `640,2560` — one budget every sampler honors |

### The `one_mu` element

The single-element target is chosen so the tail is *hard but resolvable*: the mu level
exceeded with probability 1e-2 should be high, and the curve should stay graded down to
1e-4 rather than collapsing to a cliff. Levels below are exact under each case's nominal
Bernoulli law (`p = 0.65`; calibrated queue marginals for the Texas cases).

| case | `one_mu` | `P[mu > 0]` | b @ 1e-2 | b @ 1e-3 | b @ 1e-4 |
|------|----------|-------------|----------|----------|----------|
| case14 | line `6` | 0.058 | 232 | 236.1 | 236.1 |
| case30 | line `27` | 0.170 | 39.8 | 222.9 | 259.9 |
| TX original | `base:9` | 1.0 | 58.3 | 140.8 | 152.5 |
| TX 2025 | `ctg:464:461` | 1.0 | 295.1 | 297.4 | 298.0 |
| TX original v12 | `ctg:68:18` | 1.0 | 1554.9 | 1815.5 | 1996.3 |

case14 line 6 is the one the tail rarity comes from *occurrence*: mu is zero for 94% of
configs and ~213-236 whenever it binds, so its curve is a shelf ending in a cliff and
everything below `P ~ 5e-3` sits within 1e-6 of 236.0873. The other four bind far more
often (always, on the Texas cases) and their rarity is in the *magnitude* instead.

## The samplers

Each lives in its own file and is preserved (never overwritten); all are compared at a
matched solver-call budget. The key is `(method, ascent_move)`.

| method | file | idea |
|--------|------|------|
| `direct` | `run_experiments.py` | Plain Monte Carlo from the nominal law (baseline; budget iid draws, uniform weights). |
| `subset_sim` | `subset_simulation.py` | Subset simulation (Au & Beck): nested conditional levels with a Metropolis kernel that accepts by the joint nominal pmf ratio (correct for a correlated law). |
| `ascent_is` | `ascent_importance_sampling.py` | Greedy single-flip ascent finds high-`S` modes; sample a defensive IS mixture seeded on them. Move `best`/`top5` rank flips by the sensitivity proxy. |
| `smc_tilt` | `smc_tilting.py` | SMC along the tilting path `q ∝ p_nom·exp(θS)`; stochastic bit-flip kernel; temperatures pooled by the balance heuristic. |
| `ascent_exact` | `ascent_exact_mass.py` | Same ascent + mixture as `ascent_is`, but exact nominal mass over the explored set, IS only on the unexplored remainder. |
| `cross_entropy` | `cross_entropy.py` | Cross-entropy IS fitting a product (Bernoulli) proposal to the tail; runs under the Bernoulli law only. |

## Supporting files

| file | role |
|------|------|
| `problem_setup.py` | `Problem` / `TargetProblem` for one cached case: loads `S`, the nominal law (as a generic weight vector), exact ground truth, per-config targets, bit/id helpers; mints `Evaluator`s (call counter). Holds the `CASES` registry. |
| `distributions.py` | Named nominal laws over `ngen` bits (`make_dist`, `DIST_NAMES`); `queue_p` for the calibrated Texas queue marginals. |
| `mrf.py` | `MarkovRandomField` (pairwise binary MRF) — the only file that knows the law's form. |
| `sensitivity_proxy.py` | Barrier-sensitivity proxy `s_k(x)` ranking which generator flip to evaluate; drives the `best`/`top5` ascent moves (`--proxy cached` or `live`). |
| `build_target_cache.py` | Reduce every testbed to one common per-config `target_cache.parquet`. |
| `build_dual_cache.py` | Build a compact wide `dual_cache.parquet` for a Texas case from its long `mu_cache`. |
| `plot_style.py` | Shared plotting parameters (`COLORS`, `LABELS`) and the median + 10–90 replicate `band` helper. |

## Plotting

All read a case's parquet outputs under `results/<case>/`.

| file | shows |
|------|-------|
| `plot_inverse_cdf.py` | Single-axis exceedance curve `P[target >= b]` per sampler vs truth, for one `(dist, target, snapshot, budget)` slice. |
| `plot_target_grid.py` | Grid of the same overlay: rows = distribution, cols = target. |
| `plot_per_line_survivor.py` | Grid of per-line congestion-dual survivor curves `P[|mu_l| > mu_hat]` (the quantity of interest; the target is only the sampling driver). `--kind qq` draws the per-line QQ view. |
| `plot_line_exceed_strip.py` | Per-line jittered strip of the tail probability `P[|mu_l| > c]` across the matrix. |
| `plot_target_line_strip.py` | The same strip for the single `one_mu` line only, with the four nominal laws as panels: per-replicate `P[|mu_l| > c]` by sampler against the exact truth. `c` defaults to the level that line exceeds with probability `--at` (default `1e-2`), so it auto-scales across cases. Sized 3.4 in wide for a single document column / slide at 1:1. |

## Outputs

`run_experiments.py` writes, per case, into `results/<case>/`:

- `experiment_curves.parquet` — tidy inverse-CDF frame, columns `case, dist, target,
  snapshot, budget, method, ascent_move, rep, threshold, estimate` (method `"truth"`
  carries the exact curve). Read by `plot_inverse_cdf`, `plot_target_grid`.
- `experiment_samples.parquet` — the raw weighted samples behind those curves, so
  evaluation can score any per-config quantity (e.g. per-line `mu`) from one run. Read
  by `plot_per_line_survivor`, `plot_line_exceed_strip`.

## Running

The project is an editable install rooted at `src/`, so modules import by name from any
directory. Run the module form (`-m`):

```bash
uv run python -m tail_sampling.run_experiments --case case14_gen_per_bus
uv run python -m tail_sampling.run_experiments --case case30_gen_per_bus --proxy live
uv run python -m tail_sampling.run_experiments --case all --n-rep 40 --budgets 640,2560

uv run python -m tail_sampling.plot_inverse_cdf --case case14_gen_per_bus
uv run python -m tail_sampling.plot_per_line_survivor --case case14_gen_per_bus
```
