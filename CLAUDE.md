# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

This project uses `uv` for dependency management (Python 3.14).

```bash
uv sync              # Install all dependencies
uv run python script.py  # Run a script in the project environment
uv add <package>     # Add a new dependency
```

Specific dependencies where the codebase is a helpful reference:
The EGRET source is installed at .venv/lib/python3.14/site-packages/egret/ — consult it for the real API.
The gridstatusio source is installed at .venv/lib/python3.14/site-packages/gridstatusio/ — consult it for the real API.

## Project Structure

This is a research project focused on price uncertainty in the wholesale electricity market.

### Research Goal

The primary quantity of interest is the **sensitivity of congestion dual variables (μ) to power injections** — e.g. how does the shadow price of a line's thermal limit change when a generator at a given bus produces more power. Important note: we are interested in all dual variables, not just LMPs. When implementing or discussing sensitivity analysis, always compute both dλ/dP and dμ/dP.

- **`code/`** — All project code using EGRET goes here (optimization models, experiments, analysis scripts).
- **`wholesale_data/`** — ICE electricity price Excel data (2022–2025) and a plotting script (`plot_wholesale_prices.py`).
- **`proposal_figures/`** — Scripts to generate figures from the proposal (MCMC sampling visualization, circle problem illustration). Run scripts from within that directory so PDFs are saved correctly.

### Code Style
- Each file implements one idea. Different approaches are implemented in different files. Calculation, post-processing, and plotting are all implemented in separate files.
- Functions are preferred to be around 50 lines long. Longer functions are discouraged, when possible.
- Do not plan for every possible case. If arguments do not conform to expectations, allow the program to error out instead of providing backup options. This makes code more direct, concise, and readable.
- Never ask for interactive inputs in python scripts
- If a python file has a main() function, it should be concise and contain exclusively organizational logic of the program. All specific tasks, such as plot formatting, should be implemented in other functions dedicated to those purposes.
- plotting files should contain `plt.show()`
- For loops are preferred over while loops. Especially avoid `while True:` construction.

### Plot design
- Figures should have 3-inch width to fit in a single column of a two-column document. Wider figures (e.g., more than three panels wide) should be at most 6 inches wide to fit within margins.

### Code Execution Rules
- **Temporary Snippets**: When running code via `python3 -c`, `bash -c`, or similar inline execution commands, **NEVER include comments**. 
- **Permanent Files**: When writing code to actual files (e.g., `src/`, `tests/`, `config/`), include concise comments and explanations as normal.
- **Distinction**: Comments are forbidden only in transient, inline code blocks passed directly to shell interpreters.   

### Writing style

Use American English conventions (not British).

Avoid non-standard compound words in writing. Examples include (do not use these words):
- pushforward
- copperplate

Variable and function names in code are not restricted.

### LaTeX coding style:
- Each sentence is on its own line.
- Where appropriate sentence parts can be split across lines (colons, semicolons, and lists, may fall into this category when the parts are long).
- Line length is unlimited. There is no need to add line breaks to keep lines under a character limit.
- Separated equations are preferred over in-line equations, unless they are very short (such as $s_k \ge 0$).

## EGRET Library Overview

EGRET (`.venv/lib/python3.14/site-packages/egret/`) provides:

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

## Language and phrases

These words and phrases are overused and should be avoided:
- caveat
- say the word
- smoke-test
- honest
- worth flagging
- things worth flagging
- flag
- gotcha

Confusing words to avoid:
- adder

## Allowed commands
Sometimes Claude needs permission to run a command. These commands are always allowed:
- `source gridstatus_api_key.sh`
- `$((SECONDS+520))`
