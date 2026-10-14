# sampling

Estimate the distribution of the congestion duals `μ` over generator build-out
configurations `x` (each queued project on/off), drawn from a nominal build-probability law.
For an enumerable case every one of the `2^ngen` configurations has cached duals, so the
nominal-weighted truth is known exactly and every estimate is scored against it.  For the
full queue (`ngen ≈ 195`) the configurations are drawn by MCMC instead.

This is **not** a rare-event problem.  Over the worst load hours of the year congestion is a
common outcome across the distribution of build-outs, so plain direct Monte Carlo — budget iid
draws from the nominal law, uniform weights — estimates the per-line mean, exceedance
probability, and full distribution well.  The driver is `run_experiments.py`.

## The build-probability (nominal) laws

The nominal law is **general**, not committed to iid Bernoulli.  It is supplied as a
correlated pairwise **Markov random field (MRF)** (`mrf.py`, `distributions.py`); the
estimator never sees it directly — `Problem`/`TargetProblem` exposes the law only as a
normalized weight vector over configs, an exact pmf lookup, and an exact categorical sampler,
so no method assumes the law has a particular form.  With coupling `J = 0` the MRF reduces
exactly to iid Bernoulli.

Four laws are built per case (`build_laws.py`, stored in `data/<case>/laws/<name>.json`), all
solved to the **same marginals**, so they differ in correlation alone:

| law | coupling | story |
|-----|----------|-------|
| `bernoulli` | none (`J = 0`) | each project independent at its marginal build probability |
| `mrf_a` | dense Gaussian, any sign | unstructured correlation spread thin over all pairs |
| `mrf_cobuild` | `+c` within a technology/location group | a group's projects build together or not at all |
| `mrf_substitute` | `-c` from a group's most probable project to the rest, `+c` among the rest | the near-certain project takes the capacity its class competes for |

The paper's finding is that these put nearly the same probability on each per-line quantity in
the body of the distribution, so **standard Bernoulli suffices** in most cases, with the
structured laws separating only deep in the tail.

## Estimating an enumerable case

`run_experiments.py` sweeps, per case, the cross-product

    nominal law  x  target  x  snapshot  x  budget

running direct Monte Carlo `--n-rep` times per cell and scoring each weighted sample against
the exact enumerated truth.  The case is selected by name from the registry; everything runs
as table lookups against the cached enumeration.  It writes two tidy parquet frames per case:

- `experiment_curves/` — the inverse-CDF frame (`method "truth"` carries the exact curve).
- `experiment_samples/` — the raw weighted draws, so any per-config quantity (e.g. per-line
  `μ`) can be scored from one run.

```bash
uv run python -m src.sampling.build_laws       --case case30_queue        # build the four laws (once)
uv run python -m src.sampling.run_experiments  --case case30_queue
uv run python -m src.sampling.run_experiments  --case all --n-rep 40 --budgets 640,2560
```

## Estimating the full queue by MCMC

The full leveraged queue (`activsg2000_v12_levfull`, `ngen ≈ 195`) has no exact reference, so
its distribution is estimated by drawing configurations from the nominal law and solving each.

- `build_laws_mcmc.py` builds the laws without enumeration (the structured laws calibrate
  exactly by a count collapse; see `structured_calibration.py` / `structured_sampling.py`).
- `mcmc_sweep.py` draws and solves resumably, through the same worker pool and
  screened-contingency augmentation the enumeration uses.  Output under `results/<case>/mcmc/<law>/`.
- `consolidate_mcmc_duals.py` scatters the draws into the wide per-line `μ` table the figures
  read; `consolidate_mcmc.py` reduces to the per-config target frame; `solve_modal_config.py`
  adds the single most-probable Bernoulli configuration (never drawn, so solved directly).

## Supporting files

| file | role |
|------|------|
| `problem.py` | `Problem` / `TargetProblem` for one cached case: loads `S`, the nominal law (as a generic weight vector), exact truth, per-config targets, id/bit helpers; mints call-counting `Evaluator`s.  Reads the case list from `cases.registry`. |
| `distributions.py` | Named nominal laws over `ngen` bits.  `marginal_vector` fixes the marginals, `make_dist` lays the coupling keyed by name on top; `case_dist` returns a case's stored law. |
| `build_laws.py` / `build_laws_mcmc.py` | Build each named law at a case's marginals and groups and store it — enumerated, or enumeration-free for the full queue. |
| `law_store.py` | Reads/writes a law's full definition, tracked at `data/<case>/laws/<name>.json`. |
| `mrf.py` | `MarkovRandomField` (pairwise binary MRF); `calibrate_field` solves for the field reproducing given marginals under a given coupling. |
| `structured_calibration.py` / `structured_sampling.py` | Exact calibration and exact iid sampling of the structured laws via the within-group count collapse. |

## Figures

All read a case's parquet outputs under `results/<case>/`.  The figures that appear in the
paper:

| file | shows |
|------|------|
| `figures/mu_distributions/top_config_mu.py` | per-line `|μ|` of the few most probable configurations, all laws in one panel |
| `figures/mu_distributions/top_config_estimate.py` | per-line statistic estimated from only the top-`k` configurations, vs the exact truth |
| `figures/mu_distributions/line_mean_error.py` | direct estimate of per-line `E[|μ_l|]` with standard error, converging on truth with budget |
| `figures/mu_distributions/line_exceed_error.py` | direct estimate of per-line `P[|μ_l| > c]` with error, vs truth |
| `figures/mu_distributions/mu_distribution_difference.py` | per-line `|μ|` distribution under each law, baseline vs correlated |
| `figures/mu_distributions/direct_distribution_difference.py` | the direct-sample per-line `|μ|` distribution vs the exact one |
| `figures/mu_distributions/line_sample_distribution.py` | full-queue (`levfull`) per-line `|μ|` distribution from the MCMC draws |
| `figures/mu_distributions/line_sample_error.py` | full-queue per-line estimate with its sampling error |

The per-line grids share one line selection (`_line_stats.select_lines`: the top lines by
`E[|μ_l|]` under the baseline law), so a line sits in the same panel position across figures.
The `--transpose` option puts the (long) line ids down the rows; the paper uses the
transposed layout.
