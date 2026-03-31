# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

This project uses `uv` for dependency management (Python 3.14).

```bash
uv sync              # Install all dependencies
uv run python script.py  # Run a script in the project environment
uv add <package>     # Add a new dependency
```

## Project Structure

This is a research project focused on electricity market optimization and analysis.

### Research Goal

The primary quantity of interest is the **sensitivity of congestion dual variables (μ) to power injections** — e.g. how does the shadow price of a line's thermal limit change when a generator at a given bus produces more power. Important note: we are interested in all dual variables, not just LMPs. When implementing or discussing sensitivity analysis, always compute both dλ/dP and dμ/dP.

- **`code/`** — All project code using EGRET goes here (optimization models, experiments, analysis scripts).
- **`Egret/`** — Local editable install of [EGRET](https://github.com/grid-parity-exchange/Egret) (`gridx-egret`), a Pyomo-based library for electrical grid optimization. Changes here affect the installed package directly.
- **`wholesale_data/`** — ICE electricity price Excel data (2022–2025) and a plotting script (`plot_wholesale_prices.py`).
- **`proposal_figures/`** — Scripts to generate research proposal figures (MCMC sampling visualization, circle problem illustration). Run scripts from within that directory so PDFs are saved correctly.

### Code Style
- Each file implements one idea. Different approaches are implemented in different files. Calculation, post-processing, and plotting are all implemented in separate files.
- Functions are preferred to be around 50 lines long. Longer functions are discouraged, when possible.
- Do not plan for every possible case. If arguments do not conform to expectations, allow the program to error out instead of providing backup options. This makes code more direct, concise, and readable.
- Never ask for interactive inputs in python scripts

## EGRET Library Overview

EGRET (`Egret/egret/`) provides:

- **`egret.data.model_data.ModelData`** — Central data structure used across all model formulations. Grid topology, generator parameters, etc. are stored here.
- **`egret.models`** — Ready-to-run optimization models: `unit_commitment`, `dcopf`, `acopf`, `copperplate_dispatch`, etc.
- **`egret.model_library`** — Declarative building blocks for constructing Pyomo models (power balance, ramping limits, startup costs, etc.). Used to compose models in `egret.models`.
- **`egret.parsers`** — Data parsers including RTS-GMLC (`egret.parsers.rts_gmlc`) and MatPower formats.
- **`egret.common.solver_interface`** — Solver invocation wrapper for Pyomo models.

### Solvers

- Gurobi is the preferred solver in this repository
- **HiGHS** (`highspy`) is available as an LP/MIP solver via Pyomo — preferred open-source option for unit commitment and DCOPF.
- **Ipopt** is required for NLP problems (ACOPF, AC relaxations).
- Commercial solvers (Gurobi, CPLEX) enable the full EGRET test suite.

## Development Testing

When testing code during development: 
- Do not run commands that contain a quoted newline followed by a #-prefixed line
- Temporarily remove commented lines from the code snippet for these tests, to avoid unnecessary "This command requires approval" bottlenecks


## Running EGRET Tests

```bash
# From the EGRET models/tests directory:
cd Egret/egret/models/tests
uv run pytest test_unit_commitment.py
uv run pytest test_dcopf.py

# Run all EGRET tests:
cd Egret
uv run pytest
```
