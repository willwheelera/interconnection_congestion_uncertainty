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

```
code/                   scripts (one idea per file)
  uc_example.py             solve unit commitment, save result
  sensitivity_analysis.py   barrier-regularised dμ/dP and dLMP/dP
  kkt_sensitivity.py        analytical KKT sensitivity (QP costs only)
  finite_difference_sensitivity.py  finite-difference cross-check
  plot_sensitivity.py       plot sensitivity matrices
  plot_uc_results.py        plot UC dispatch results
Egret/                  local editable install of EGRET
results/                output files (JSON, PDF)
docs/                   Sphinx documentation
wholesale_data/         ICE price data and plotting script
proposal_figures/       research proposal figure scripts
```

## Typical workflow

**1. Solve unit commitment and save the result**

```bash
uv run python code/uc_example.py
# writes results/uc_test_scuc_masked.json
```

**2. Compute sensitivity from the saved solution**

```bash
# barrier method (works for LP and QP costs; computes dμ/dP and dLMP/dP)
uv run python code/sensitivity_analysis.py
uv run python code/sensitivity_analysis.py --solution results/uc_test_scuc_masked.json --hour 12

# KKT / bordered-Hessian method (QP costs only; faster, exact)
uv run python code/kkt_sensitivity.py

# finite-difference cross-check (slow: re-solves DCOPF once per bus)
uv run python code/finite_difference_sensitivity.py
```

Both barrier and KKT scripts write JSON to `results/`.

**3. Plot results**

```bash
# single method
uv run python code/plot_sensitivity.py results/sensitivity_barrier_<case>.json

# side-by-side comparison
uv run python code/plot_sensitivity.py \
    results/sensitivity_kkt_<case>.json \
    results/sensitivity_fd_<case>.json
```

## Methods

### Barrier-regularised sensitivity (`sensitivity_analysis.py`)

Works for any cost structure (LP or QP).  Replaces the singular bordered-Hessian of the LP case with a regularised saddle-point system:

```
[ H + Y + D_w    J^T ] [ dx/dP ]   [ 0  ]
[ J             -D_c ] [ dν/dP ] = [ e_j]
```

`Y` is built from all inequality constraints using barrier weights `y_k = (z_k + ε) / (s_k + ε)` evaluated at the optimum.  Congestion dual sensitivities are recovered from the differentiated complementarity condition.

See `docs/narrative/kkt_sensitivity.rst` for the full derivation.

### KKT / bordered-Hessian (`kkt_sensitivity.py`)

For QP costs (quadratic generator cost curves) only.  Differentiates the KKT stationarity conditions directly; no re-solving required.  Returns zero and exits early for LP problems.

### Finite-difference (`finite_difference_sensitivity.py`)

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
