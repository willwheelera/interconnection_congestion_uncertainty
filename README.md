# AMPS — Congestion price uncertainty from the interconnection queue

This repository estimates how uncertain generation-interconnection outcomes move wholesale
**congestion prices**, by Monte Carlo sampling.

A generator interconnection queue is a set of proposed projects, each of which may or may not
be built.  Which ones are built shifts power flows and therefore the congestion shadow prices
(the dual variables `μ` of transmission limits) that set locational price differences.  On the
planning timescale this is a source of price uncertainty that single-scenario studies miss.

Given a probability distribution over which queued projects are built, we:

1. draw configurations of the queue (each project on/off) from that distribution,
2. solve a security-constrained DC optimal power flow for each one, averaged over a few
   high-load snapshots of the year, and
3. estimate, for every line and contingency, the full distribution of `μ`, its mean, and the
   probability it exceeds a price threshold.

The central result is that **plain direct Monte Carlo is enough**: a small number of samples
estimates these quantities well, and does better than picking the one or few most likely
build-outs.  This is not a rare-event problem — over the worst hours of the year congestion is
a common outcome across the distribution of build-outs — so no importance sampling is needed.
The size of the uncertainty problem is set by the queue, not the network, and the sampling
cost is decoupled from the optimization (warm starts make each extra sample cheap).

The method and empirical results on a 2000-bus synthetic ERCOT network are described in the
accompanying paper, *Congestion price uncertainty from interconnection queue by Monte Carlo
sampling*.

## What the repository computes

The work is one pipeline, run per testbed **case**:

```
define a case ─▶ solve one build-out ─▶ sweep many build-outs ─▶ estimate the distribution ─▶ draw
  (cases/)          (model/)              (sweeps/, sampling/)       (sampling/)            (figures/)
```

A **case** is a network, a queue of candidate projects with their build probabilities, and
the modeling recipe it is solved under.  Every command takes `--case <name>` and reads
everything else from the case registry (`src/cases/registry.py`), so the same commands run any
case.  The cases that reproduce the paper are the ACTIVSg2000 (synthetic ERCOT) build-outs
`activsg2000_v12_lev3a3` (an enumerable 20-project queue) and `activsg2000_v12_levfull` (the
full 195-project queue, estimated by MCMC).  Two small IEEE cases, `case14_queue` and
`case30_queue`, are included as fast runnable examples of the same pipeline.

`src/README.md` is the full map of the pipeline and the data layout.

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd AMPS
./setup.sh          # fetches EGRET and installs all dependencies with `uv sync`
```

Solvers — the models are linear programs, so either works:

- **Gurobi** (`gurobipy`) — preferred, needs a license (free for academics).
- **HiGHS** (`highspy`) — open-source fallback; returns the same LP duals.

## Running the calculation

### A small example, end to end (runs on a laptop)

`case14_queue` enumerates all `2^14` build-outs in well under a minute:

```bash
uv run python -m src.sampling.build_laws        --case case14_queue        # the four build-probability models
uv run python -m src.sweeps.enumerate_configs   --case case14_queue        # ground-truth duals
uv run python -m src.sweeps.build_target_cache  --case case14_queue        # per-config summary
uv run python -m src.sweeps.build_dual_cache    --case case14_queue        # per-line |μ| table
uv run python -m src.sampling.run_experiments   --case case14_queue        # direct Monte Carlo
uv run python -m src.figures.mu_distributions.line_mean_error --case case14_queue --transpose
```

Each stage writes to `results/<case>/` and the next stage reads it; `src/cases/README.md` and
`src/sampling/README.md` describe the caches.  `case30_queue` runs the same way (its `2^20`
enumeration takes a couple of hours on ~16 workers).

### Reproducing the paper — enumerable case (`activsg2000_v12_lev3a3`)

This is a 2000-bus case swept over eight load snapshots, so the enumeration is large and is
meant for a cluster (`src/sweeps/submit_cluster.py` and `src/sweeps/vacc.slurm` run it as a
SLURM job chain; see `src/README.md`).  The stages are the same:

```bash
uv run python -m src.cases.prepare_case        --case activsg2000_v12_lev3a3   # relief set + screened contingencies
uv run python -m src.sweeps.enumerate_configs  --case activsg2000_v12_lev3a3 --screened
uv run python -m src.sweeps.build_target_cache --case activsg2000_v12_lev3a3
uv run python -m src.sweeps.build_dual_cache   --case activsg2000_v12_lev3a3
uv run python -m src.sampling.run_experiments  --case activsg2000_v12_lev3a3

# paper figures
uv run python -m src.figures.mu_distributions.top_config_mu                   --case activsg2000_v12_lev3a3
uv run python -m src.figures.mu_distributions.top_config_estimate            --case activsg2000_v12_lev3a3 --transpose
uv run python -m src.figures.mu_distributions.line_mean_error                --case activsg2000_v12_lev3a3 --transpose
uv run python -m src.figures.mu_distributions.line_exceed_error              --case activsg2000_v12_lev3a3 --transpose
uv run python -m src.figures.mu_distributions.mu_distribution_difference     --case activsg2000_v12_lev3a3
uv run python -m src.figures.mu_distributions.direct_distribution_difference --case activsg2000_v12_lev3a3
```

The queue and the four build probability models (`bernoulli` and the correlated `mrf_a`,
`mrf_cobuild`, `mrf_substitute`) are committed under `data/activsg2000_v12_lev3a3/`.

### Reproducing the paper — full-queue case (`activsg2000_v12_levfull`)

The 195-project queue has `2^195` build-outs, too many to enumerate, so its distribution is
estimated by drawing configurations with MCMC and solving each (`src/sampling/mcmc_sweep.py`).
This is a multi-day cluster run per probability model:

```bash
uv run python -m src.cases.prepare_case  --case activsg2000_v12_levfull
uv run python -m src.sampling.mcmc_sweep --case activsg2000_v12_levfull --law bernoulli --screened
uv run python -m src.sampling.consolidate_mcmc_duals --case activsg2000_v12_levfull --law bernoulli --rounds 200
uv run python -m src.sampling.solve_modal_config     --case activsg2000_v12_levfull --law bernoulli

uv run python -m src.figures.mu_distributions.line_sample_distribution --case activsg2000_v12_levfull --law bernoulli
uv run python -m src.figures.mu_distributions.line_sample_error        --case activsg2000_v12_levfull --law bernoulli
```

Repeat the `mcmc_sweep`/`consolidate`/figure steps for `--law mrf_cobuild` and
`--law mrf_substitute` to compare build probability models.

### Rebuilding the ACTIVSg2000 inputs from source (optional)

The ACTIVSg2000 inputs ship committed under `data/`, so this is only needed to regenerate them
with `src/cases/texas/` (see that README for the full command sequence).  Three external sources
are involved:

- **ACTIVSg2000 model** — download the case from
  [Texas A&M](https://electricgrids.engr.tamu.edu/electric-grid-test-cases/activsg2000/) into
  `texas_model/ACTIVSg2000/`; `convert_model` turns it into `data/activsg2000.json`.
- **ERCOT Public API** (for the load and renewable snapshots) — register for an ERCOT Public API
  account to get a username and subscription key, set them as `ERCOT_API_USERNAME` and
  `ERCOT_PUBLIC_API_SUBSCRIPTION_KEY`, then run `uv run python -m src.ercot.ercot_auth` to
  exchange your password for a ~1-hour token (cached to `~/.ercot_token.json`); the `fetch_*`
  steps read it.
- **ERCOT CDR report** — download the Capacity, Demand and Reserves workbook into
  `data/ercot_cdr/` for the per-fuel capacity factors.

## Repository layout

```
src/            the pipeline, one package per stage (see src/README.md)
data/           committed case inputs: the network, load snapshots, queues, and build models
texas_model/    ERCOT interconnection-queue data the ACTIVSg2000 construction reads
results/        output caches and figures, one directory per case (regenerated, not tracked)
notes/          research notes
Egret/          local EGRET reference, fetched by setup.sh (not tracked)
```

`results/` is derived from `data/` and is not committed; the commands above regenerate it.  The
raw ACTIVSg2000 model and ERCOT workbooks that `src/cases/texas/` reads are bulk downloads and
are not tracked either; the committed `data/activsg2000*` files are their processed outputs.

## EGRET

[EGRET](https://github.com/grid-parity-exchange/Egret) is a Pyomo-based grid-optimization
library; the security-constrained DCOPF is built on it.  `setup.sh` fetches it, and `uv`
installs it as a dependency.
