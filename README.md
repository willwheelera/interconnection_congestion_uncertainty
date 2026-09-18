# AMPS — Electricity Market Sensitivity Analysis

Research project studying congestion dual variables (shadow prices of line thermal limits) over different generator availability in DC optimal power flow and unit commitment problems.

## Research goal

The overall goal is estimating distribution of locational marginal prices (LMPs) given a probability function for every combination of generators (available or not available) — useful for market analysis and generator siting.
Our study focuses on predicting transmission congestion, the main driver of high LMPs.
Specifically, given a transmission network and a probability function for generator configurations, we ask: "what is the probability that the congestion shadow price of line l exceeds a threshold value?"
The space of generator configurations is too large to enumerate, necessitating a reliable sampling scheme.
Since congestion is relatively rare and difficult to predict, importance sampling is likely needed.

A candidate importance function is the prior probability times the sum of shadow prices across the network for each generator configuration.
The normalization for this function is unknown.
A local scheme for biasing acceptance probabilities and move proposals in exchange for importance weights is preferable.
"Local" means informed by neighbors, i.e., configurations that differ by one generator (on/off).
The increase or decrease in congestion prices from a configuration to its neighbor can inform move proposal and acceptance.

The main point of interest is very large values of mu, which are rare.
Finding the probability of rare events is difficult using random sampling.
We consider approaches that use an ascent strategy to locate local maxima, and calculate corresponding weights to determine probabilities.

To save on computation, we propose estimating the change in congestion through sensitivity.
For linear cost, congestion (mu) is piecewise constant.
In practice, the costs have small quadratic terms, leading to piecewise linear mu; but critically, they are still discontinuous, making derivatives useless.
We aim for a proxy based on barrier methods (such as IPOPT) and assess its usefulness in predicting the actual change in mu.

## Current focus: sampling the congestion-dual tail

The active line of work estimates the high tail of the congestion target

```
S(x) = sum_l |mu_l(x)|
```

over the discrete space of generator-availability configurations `x` (each generator on/off).
High-`S` configurations are rare under the nominal law, so we compare rare-event sampling schemes against an **exact ground truth** obtained by enumerating every config and caching its duals.

The nominal law is **general**, not fixed to iid Bernoulli. The iid-Bernoulli law in the current testbeds is a temporary, convenient choice; the methods must handle correlated discrete laws, in particular a **Markov random field (MRF)** form (`sampling/fit_mrf.py` fits pairwise MRF parameters to the dual landscape). Likewise, cross-entropy is one option among several for choosing an importance distribution — its closed-form update only carries an advantage for product/Bernoulli families, so for an MRF or other correlated law we stay **flexible on the importance metric** (variance minimization, exact-mass, tilting, etc.) rather than privileging cross-entropy.

**Pipeline.** `model/` solves one configuration and extracts per-constraint μ; `sweeps/enumerate_configs.py` runs all `2^ngen` configurations and `sweeps/build_target_cache.py` reduces the result to a common per-configuration frame; `sampling/problem.py` defines a `Problem` class that loads that cache and provides `S`, the nominal law, the call-counting `Evaluator`, and the exact exceedance curve `P[S>=b]` used as truth.  The runner (`sampling/run_experiments.py`) constructs the `Problem` for a case named on the command line and passes it into every sampler, so no case is hardcoded in the library.

**Samplers** (all live in `src/sampling/methods/`, each preserved in its own file, compared cumulatively at a matched solver-call budget via `sampling/run_experiments.py`; see `src/sampling/README.md`):

| key | file | sampler |
|-----|------|---------|
| `direct` / `subset_sim` | `subset_simulation.py` | direct nominal sampling baseline + subset simulation (Au & Beck) |
| `ascent_is` | `ascent_importance_sampling.py` | greedy single-flip ascent seeds a defensive IS mixture (currently Bernoulli, an ansatz — not a commitment to a Bernoulli nominal law) |
| `smc_tilt` | `smc_tilting.py` | SMC along the exponential-tilting path `q ~ p_nom * exp(theta*S)` (no ansatz fit) |
| `ascent_exact` | `ascent_exact_mass.py` | ascent-IS but exact mass over the explored set + IS on the unexplored remainder |

**Testbeds.** `case14_gen_per_bus` (14 bits, fully enumerated) was the first model but proved too easy — its deep tail has only ~26 configs, so a sampler covering a few percent of the space nearly enumerates it.
`case30_gen_per_bus` (built by `cases/build_gen_per_bus.py`: IEEE-30 topology, a generator at each of the 20 load buses, `2^20` configs) is the harder replacement: `diagnostics/check_congestion_landscape.py` confirms a genuinely multimodal tail (24/41 lines bind, ~310 distinct binding-line signatures and 21 distinct peak-μ corridors in the high-`S` tail).
The exhaustive `case30` ground truth is now solved: every one of the `2^20` generator-availability configs has been run through `model/dcopf.py` on Gurobi (1,048,575 viable solutions in `results/case30_gen_per_bus/`; only the all-off config is infeasible).
The enumerated `S = sum_l |mu_l|` distribution spans `p50 = 220`, `p90 = 474`, `p99 = 1144`, `max = 2040`, with up to 13 lines binding at once.

These two gen-per-bus cases have since been superseded by the five-case incumbent-fleet matrix in `notes/paper_plan.md` §3; they are kept for continuity and not re-run.

> Solver note: Gurobi is licensed and is the solver used for the `case30` enumeration; **HiGHS** (`solver="highs"`) remains available as an open-source fallback that returns the same LP duals.

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd AMPS
./setup.sh               # clones EGRET into Egret/ and installs all dependencies
```

Solvers:
- **Gurobi** — preferred for unit commitment (MIP) and DCOPF; requires a valid licence
- **HiGHS** — open-source fallback for LP/DCOPF (`--solver highs`)
- **Ipopt** — required for AC OPF (not currently used in main scripts)

## Project layout

`src/` is one package per workflow stage; see `src/README.md` for the full map and the case
registry that ties them together.

```
src/
  cases/         define a testbed and screen its N-1 set   ->  data/<case>*.json
    registry.py    THE case registry: every stage reads case facts from here
  model/         one configuration -> congestion duals mu  (the recipe lives here)
  sweeps/        many configurations -> caches             ->  results/<case>/*
  sampling/      estimate the tail from a sample, score against exact truth
  descriptors/   try to predict the tail from one solve instead (it fails)
  figures/       every plot and table; nothing here computes a result
  sensitivity/   barrier-regularised dmu/dP and dLMP/dP
  ercot/         empirical ERCOT congestion study (motivation, independent of the pipeline)
  diagnostics/   one-off probes and audits
  utils/         low-level helpers
  _archive/      superseded code, kept readable, imported by nothing
Egret/           read-only EGRET reference (runtime EGRET pinned in pyproject.toml)
data/            network JSON cases, queue files, and fetched ERCOT datasets
results/         output caches and figures, one directory per case
notes/           research notes and the paper plan
scripts/         cluster job scripts
```

## Typical workflow

Every entry point takes `--case <name>` and reads its paths, queue, and modeling recipe from
`cases.registry`, so the same commands run any of the eight testbeds.

```bash
# build a testbed and screen its contingency set
uv run python -m src.cases.build_ieee_queue     --case case30_queue
uv run python -m src.cases.calibrate_ratings    --case case30_queue
uv run python -m src.cases.screen_contingencies --case case30_queue

# exhaustive ground truth, then the common caches every consumer reads
uv run python -m src.sweeps.enumerate_configs   --case case30_queue --workers 16
uv run python -m src.sweeps.build_target_cache  --case case30_queue
uv run python -m src.sweeps.build_dual_cache    --case case30_queue

# estimate, describe, draw
uv run python -m src.sampling.run_experiments        --case case30_queue
uv run python -m src.figures.mu_distributions.per_line_survivor  --case case30_queue
uv run python -m src.figures.descriptors
```

The sensitivity thread runs separately:

```bash
uv run python -m src.sensitivity.barrier --dir case14_gen_per_bus        # barrier dmu/dP, dLMP/dP
uv run python -m src.sensitivity.finite_difference                       # finite-difference cross-check
uv run python -m src.figures.sensitivity.sensitivity results/sensitivity_barrier_<case>.json
```

## Methods

### Barrier-regularised sensitivity (`sensitivity/barrier.py`)

Works for any cost structure (LP or QP).  Replaces the singular bordered-Hessian of the LP case with a regularised saddle-point system:

```
[ H + Y + D_w    J^T ] [ dx/dP ]   [ 0  ]
[ J             -D_c ] [ dν/dP ] = [ e_j]
```

`Y` is built from all inequality constraints using barrier weights `y_k = (z_k + ε) / (s_k + ε)` evaluated at the optimum.  Congestion dual sensitivities are recovered from the differentiated complementarity condition.

See `docs/narrative/kkt_sensitivity.rst` for the full derivation.

### KKT / bordered-Hessian

For QP costs (quadratic generator cost curves) only.  Differentiates the KKT stationarity conditions directly; no re-solving required.  Returns zero and exits early for LP problems.

### Finite-difference (`sensitivity/finite_difference.py`)

Re-solves DCOPF with a small load perturbation at each bus.  Slow but assumption-free; used to validate the analytical methods.

## EGRET

[EGRET](https://github.com/grid-parity-exchange/Egret) is a Pyomo-based library for grid optimisation.  It is installed as a local editable package from `Egret/`.  Changes to files under `Egret/egret/` take effect immediately without reinstalling.

Key classes and functions:
- `egret.data.model_data.ModelData` — central data structure for network topology and results
- `egret.models.dcopf.solve_dcopf` — DC optimal power flow
- `egret.models.unit_commitment.solve_unit_commitment` — security-constrained unit commitment

## Notes on LMPs and unit commitment

`solve_unit_commitment` solves a MIP; MIP solvers do not produce meaningful dual variables, so **LMPs are not available in a standard UC solution**.  LMPs require either:
- solving the LP relaxation (`relaxed=True`), or
- fixing commitment decisions and re-solving as DCOPF.

The sensitivity scripts use LMP differences across buses as proxies for congestion dual values when constructing barrier weights.  If LMPs are absent (as in a UC result file), the barrier weights fall back to the smoothing parameter `eps`.

## Docs

```bash
cd docs
make html
# open _build/html/index.html
```
