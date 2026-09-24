# `data/` directory layout

How case inputs are organized on disk, and the convention new code follows.

## Convention (as of 2026-09-24)

Each case keeps its **own** inputs together in a per-case directory `data/<case>/`:

```
data/<case>/
  queue.json            # the candidate queue (candidates_file)
  screened.json         # the screened case (screened_file)
  contingencies.json    # kept contingency names + recipe (contingency_file)
  relief_set.json       # the ORDC relief bus set (relief_file)
  laws/<name>.json      # nominal laws: bernoulli, mrf_cobuild, mrf_substitute, mrf_b
```

The **shared physical model** stays flat in `data/`, because many cases sit on it:

- `activsg2000.json` — the ACTIVSg2000 network (14 cases share it)
- `activsg2000_load_collection.json` — the load/renewable snapshot collection (all Texas cases)
- `activsg2000_queue20.json` — the queue shared by the three incumbent-fleet `_texas` cases
- `activsg2000_2025.json`, `case*_queue.json`, `case*_gen_per_bus.json` — 1:1 networks, kept flat (see objective below)
- `activsg2000_screened.json`, `activsg2000_contingencies_screened.json` — the `original` case's legacy-named screened set

Outputs (regenerable) stay under the git-ignored `results/<case>/`.

`src/cases/registry.py` is the single source: the `Case` path properties resolve every
input from `data_dir = data/<case>/`, with `paths` overrides for the shared/legacy files
above.  All reads and writes go through those properties, so nothing hardcodes a path.

## Organization objective (not yet done)

Several older scripts still **hardcode** `data/...` paths instead of resolving them through
the registry.  They are not broken -- they all reference files that stayed flat -- but they
do not follow the pattern the new code uses, and they are why the 1:1 networks
(`case*_gen_per_bus.json`, `case*_queue.json`, `activsg2000_2025.json`) and the `original`
case's screened set could not be moved into their case directories without editing ~20 call
sites.

Future cleanup: route these through `registry.get(...)` and the `Case` properties, then move
their remaining 1:1 inputs into the case directories too.

Scripts with hardcoded `data/` paths to convert:

- `src/diagnostics/check_congestion_landscape.py`, `check_relief_curve_mu.py`,
  `time_dcopf.py`, `identify_relief_set.py`, `check_original_feasibility.py`
- `src/figures/line_limits.py`, `tables.py`,
  `src/figures/sensitivity/{sensitivity_vs_flip,top_sensitivity_vs_flip}.py`
- `src/sensitivity/build_cache.py`, `scdcopf.py`, `tests/test_vs_finite_difference.py`
- builders that write the shared files (fine to leave, but could take their output path from
  the registry): `src/cases/texas/{convert_model,add_cdr_resources,build_load_snapshots,build_renewable_availability,build_queue_distribution}.py`,
  `src/cases/build_gen_per_bus.py`
