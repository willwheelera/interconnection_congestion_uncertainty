# sampling

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
`problem.py`, and everything runs as table lookups against the cached enumeration.

| axis | values |
|------|--------|
| case | `case14_gen_per_bus`, `case30_gen_per_bus`, `activsg2000_original`, `activsg2000_2025`, `activsg2000_original_v12` |
| nominal law (`dist`) | `DIST_NAMES` = `bernoulli` (J=0), `mrf_a` (J~N(0,0.15), unstructured mixed-sign), and the two structured laws `mrf_cobuild` and `mrf_substitute` (below), which `STRUCTURED_NAMES` marks as needing the queue's fuel groups and so skips on a case without a queue. Every law's field is solved for the same marginals, so they differ in coupling alone |
| target | `max_mu`, `sum_mu`, `one_mu`, `one_lmp` (whichever the case's cache carries). `one_mu` is a single element, named per case in `cases.registry` (table below); unnamed cases fall back to the globally most-congested element |
| snapshot | `multi` (averaged/nominal); `single` runs `max_mu` on the per-snapshot peak (Texas only) |
| ascent move | `all` always; `best` / `top5` only where a sensitivity proxy exists — in practice case14 only, since case30 has no `sensitivity_cache.parquet` (use `--proxy live` to get them there) |
| ascent-IS beta | `--betas` (default `0.2`, the weight the figures draw) — the defensive weight of the ascent-IS mixture; each value is written as its own method, `ascent_is_b0.5`, and costs one ascent-IS run |
| budget | solver-call budgets, e.g. `640,2560` — one budget every sampler honors |

### The structured laws (`mrf_cobuild`, `mrf_substitute`)

`mrf_a` barely moves the mu distribution: its coupling is an unstructured Gaussian draw
spread thin over all `ngen*(ngen-1)/2` pairs, so the induced bit correlations stay small
(mean 0.010, max 0.046 on `activsg2000_original_v12`) and none of them line up with the
handful of candidates a congested element responds to. The two structured laws instead
couple strongly *within a technology group* (`Case.candidate_groups()`, one block per
fuel) at `GROUP_COUPLING = 2.0`, reaching mean |corr| 0.08 with a max near 0.7:

| law | coupling | story |
|-----|----------|-------|
| `mrf_cobuild` | `+c` inside each fuel block | one technology's candidates build together or not at all (a technology-wide cost or policy shock) |
| `mrf_substitute` | `-c` from each block's most probable candidate to the rest, `+c` among the rest | the near-certain project takes the interconnection capacity its class competes for; the speculative projects clear as a block only when it does not |

On `activsg2000_original_v12` these two put the same probability on every individual
candidate as Bernoulli and still separate `P[S > 1600]` by a factor of 117
(`2.8e-4` vs `3.3e-2`, against Bernoulli's `4.6e-3`); on the `one_mu` element `ctg:68:18`
at its 1e-3 level they separate by 135. `diagnostics/check_law_coupling.py` prints the
table and `figures/mu_distributions/law_tail_compare.py` draws the curves.

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
| `subset_sim` | `methods/subset_simulation.py` | Subset simulation (Au & Beck): nested conditional levels with a Metropolis kernel that accepts by the joint nominal pmf ratio (correct for a correlated law). |
| `ascent_is_b<beta>` | `methods/ascent_importance_sampling.py` | Greedy single-flip ascent finds high-`S` modes; sample a defensive IS mixture seeded on them, the nominal law carrying weight `beta`. Move `best`/`top5` rank flips by the sensitivity proxy. One method per `--betas` value: large `beta` caps the weights at `1/beta` but spends most of the sample on plain nominal draws, small `beta` leans on the modes. |
| `smc_tilt` | `methods/smc_tilting.py` | SMC along the tilting path `q ∝ p_nom·exp(θS)`; stochastic bit-flip kernel; temperatures pooled by the balance heuristic. |
| `ascent_exact` | `methods/ascent_exact_mass.py` | Same ascent + mixture as `ascent_is`, but exact nominal mass over the explored set, IS only on the unexplored remainder. |
| `cross_entropy` | `methods/cross_entropy.py` | Cross-entropy IS fitting a product (Bernoulli) proposal to the tail; runs under the Bernoulli law only. |

## Supporting files

| file | role |
|------|------|
| `problem.py` | `Problem` / `TargetProblem` for one cached case: loads `S`, the nominal law (as a generic weight vector), exact ground truth, per-config targets, bit/id helpers; mints `Evaluator`s (call counter). Reads the case list from `cases.registry`. |
| `distributions.py` | Named nominal laws over `ngen` bits. `marginal_vector` fixes the marginals, then `make_dist` lays the coupling keyed by name on top (`DIST_NAMES`, `STRUCTURED_NAMES`, `P_STD`); `case_dist` returns a registered case's stored law and `case_law_names` filters a requested list down to what a case can have. |
| `build_laws.py` | Builds each named law at a case's marginals and technology groups and stores it — the only place a field is ever solved. Every coupled law's field comes from `mrf.calibrate_field`, so its marginals match Bernoulli's to solver precision. |
| `law_store.py` | Reads and writes a law's full definition, tracked `data/<case>/<name>.json`: `h`, `J`, target and realized marginals, groups, coupling. |
| `mrf.py` | `MarkovRandomField` (pairwise binary MRF) — the only file that knows the law's form; `calibrate_field` solves for the field that reproduces given marginals under a given coupling. |
| `diagnostics.check_law_coupling` | Per-law marginal error, bit correlations, distance to Bernoulli, and exact mu exceedance — the check that two laws differ in correlation only. |
| `sensitivity.proxy` | Barrier-sensitivity proxy `s_k(x)` ranking which generator flip to evaluate; drives the `best`/`top5` ascent moves (`--proxy cached` or `live`). |
| `sweeps.build_target_cache` | Reduce every testbed to one common per-config `target_cache.parquet`. |
| `sweeps.build_dual_cache` | Build a compact wide `dual_cache.parquet` for a Texas case from its long `mu_cache`. |
| `figures.style` | Shared plotting parameters (`COLORS`, `LABELS`, the ascent-IS `BETA_COLORS` ramp), the method-key helpers (`parse_method`, `method_color`, `method_label`, `select_methods`), the shared `--methods` selection, and the median + 10–90 replicate `band` helper. |

## Plotting

All read a case's parquet outputs under `results/<case>/`.

Every comparison figure takes one selection option, `--methods`, whose values are the
method names as they appear in the data (`figures.style.DEFAULT_METHODS` is what it draws
without it):

| `--methods` | draws |
|-------------|-------|
| *(omitted)* | `direct` and ascent-IS at beta 0.2 |
| `all` | every method the sweep wrote |
| `direct,ascent_is` | ascent-IS at every beta present, plus direct |
| `ascent_is_b0.5` | that one beta |
| `direct,subset_sim,smc_tilt,cross_entropy` | any explicit set |

A name is a full method key, a sampler family covering its betas, or `all`; a name matching
nothing in the data raises rather than silently dropping a curve. Anything but the default
is tagged into the output filename, so a restricted figure never overwrites the default one.

| file | shows |
|------|-------|
| `figures/mu_distributions/inverse_cdf.py` | Single-axis exceedance curve `P[target >= b]` per sampler vs truth, for one `(dist, target, snapshot, budget)` slice. |
| `figures/mu_distributions/target_grid.py` | Grid of the same overlay: rows = distribution, cols = target. |
| `figures/mu_distributions/per_line_survivor.py` | Grid of per-line congestion-dual survivor curves `P[|mu_l| > mu_hat]` (the quantity of interest; the target is only the sampling driver). `--kind qq` draws the per-line QQ view. |
| `figures/mu_distributions/line_exceed_strip.py` | Per-line jittered strip of the tail probability `P[|mu_l| > c]` across the matrix. |
| `figures/mu_distributions/target_line_strip.py` | The same strip for the single `one_mu` line only, with the four nominal laws as panels: per-replicate `P[|mu_l| > c]` by sampler against the exact truth. `c` defaults to the level that line exceeds with probability `--at` (default `1e-2`), so it auto-scales across cases. Sized 3.4 in wide for a single document column / slide at 1:1. |

## Outputs

`run_experiments.py` writes, per case, into `results/<case>/`:

- `experiment_curves.parquet` — tidy inverse-CDF frame, columns `case, dist, target,
  snapshot, budget, method, ascent_move, rep, threshold, estimate` (method `"truth"`
  carries the exact curve). Read by `figures.mu_distributions.inverse_cdf`, `figures.mu_distributions.target_grid`.
- `experiment_samples.parquet` — the raw weighted samples behind those curves, so
  evaluation can score any per-config quantity (e.g. per-line `mu`) from one run. Read
  by `figures.mu_distributions.per_line_survivor`, `figures.mu_distributions.line_exceed_strip`.

## Running

The project is an editable install rooted at `src/`, so modules import by name from any
directory. Run the module form (`-m`):

A case's nominal laws are built and stored once, before anything samples or plots under
them; `case_dist` then only reads the tracked `data/<case>/<name>.json`, and a law that
has not been built is an error rather than a silent rebuild.

```bash
uv run python -m src.sampling.build_laws --case all

uv run python -m src.sampling.run_experiments --case case14_gen_per_bus
uv run python -m src.sampling.run_experiments --case case30_gen_per_bus --proxy live
uv run python -m src.sampling.run_experiments --case all --n-rep 40 --budgets 640,2560

# extend an existing sweep with more ascent-IS defensive weights (tops the shards up)
uv run python -m src.sampling.run_experiments --case activsg2000_original_v12 --betas 0.5,0.8

uv run python -m src.figures.mu_distributions.inverse_cdf --case case14_gen_per_bus
uv run python -m src.figures.mu_distributions.per_line_survivor --case case14_gen_per_bus
uv run python -m src.figures.mu_distributions.target_grid --case activsg2000_original_v12 \
    --methods direct,ascent_is
```
