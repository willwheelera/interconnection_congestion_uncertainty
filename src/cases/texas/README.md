# cases/texas — ACTIVSg2000 relief selection

The ACTIVSg2000 (synthetic ERCOT) cases are built from the shared network
`data/activsg2000.json`, the load snapshots `data/activsg2000_load_collection.json`, and each
case's own leveraged queue `data/<case>/queue.json` — all committed.  On this branch, the only
construction step that still runs as part of preparing a case is choosing the slim relief set:

| file | role |
|---|---|
| `select_relief_set.py` | Collect the buses that ever shed load under full ORDC relief, validate that restricting relief to that set reproduces the target `S` on held-out configurations, and write `data/<case>/relief_set.json`.  Called by `cases.prepare_case`. |

The synthetic ACTIVSg2000 grid is 2000 buses, 3206 branches, and 544 base generators.  A
*configuration* is a bit vector over ERCOT interconnection-queue candidates (20 for the
`lev*a3` cases, 195 for `levfull`), each either built or not.

## Why a slim relief set

A full relief generator at every load bus enlarges the LP about 5x — an inactive relief
column is still factorized on every warm re-solve — while only a handful of ACTIVSg2000's load
buses ever shed.  The relief set is a sweep **input**: it lives in `data/<case>/relief_set.json`
and `Case.relief_buses()` falls back to every load bus until one is selected.  The recipe it
supports (emergency factor, ORDC-shaped nodal relief, shadow-price cap) is documented in
`src/model/README.md`.

```bash
uv run python -m src.cases.prepare_case --case activsg2000_v12_lev3a3   # runs select_relief_set, then screening
```

## The full construction, on `archive`

Building the ACTIVSg2000 cases from source — the MATPOWER/PowerWorld network conversion, the
ERCOT CDR fleet additions, the ERCOT load and renewable snapshot pulls, and the leveraged-queue
construction — is a one-time process whose committed outputs are the `data/activsg2000*` files
the registry points at.  Those construction scripts (`convert_model`, `add_cdr_resources`,
`fetch_*`, `build_leveraged_queue`, `screen_bus_leverage`, …) live on the `archive` branch;
they are not needed to reproduce the paper from the committed inputs.
