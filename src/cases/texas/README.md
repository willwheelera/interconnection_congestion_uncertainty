# cases/texas — building the ACTIVSg2000 testbeds from source

One-time construction of the ACTIVSg2000 (synthetic ERCOT) inputs that `cases.registry` points
at: the network, the load/renewable snapshots, and the leveraged queues.  Their outputs are the
committed `data/activsg2000*` and `data/activsg2000_v12_*/queue.json` files, so rerun these only
to rebuild an input.  The raw sources they read — the ACTIVSg2000 MATPOWER/PowerWorld model and
the ERCOT CDR workbook — are bulk downloads and are not tracked; the two `fetch_*` steps also
need an ERCOT API token (`ercot_auth.py`).

The synthetic ACTIVSg2000 grid is 2000 buses, 3206 branches, and 544 base generators.  A
*configuration* is a bit vector over ERCOT interconnection-queue candidates (20 for the
`lev*a3` cases, 195 for `levfull`), each either built or not.  The shipped cases are built on the
base `activsg2000.json` fleet; only the queue on top differs.

## Files

| file | role |
|---|---|
| `convert_model.py` | MATPOWER + PowerWorld `.aux` → EGRET JSON with coordinates, ERCOT weather zones, emergency ratings, and the N-1 contingency set.  **This is the loader that turns the downloaded ACTIVSg2000 model into the `data/activsg2000.json` the registry uses.** |
| `compile_resource_additions.py` | Read the ERCOT CDR unit-detail tables (fuel groups and capacity factors used by the renewable step). |
| `fetch_load_snapshots.py` | Pull ERCOT actual zonal load for a set of high-congestion hours (needs an ERCOT token). |
| `build_load_snapshots.py` | Turn those into per-zone load scale factors → `data/activsg2000_load_collection.json`. |
| `fetch_renewable_snapshots.py` | Pull ERCOT actual wind and solar output for the same hours (needs an ERCOT token). |
| `build_renewable_availability.py` | Attach a per-fuel capacity factor to each snapshot in the collection. |
| `build_queue_distribution.py` | Select ERCOT queue projects, match them to buses geographically, set each candidate's build probability → a base queue. |
| `screen_bus_leverage.py` | Rank buses by how much power they can push onto the constraints that price congestion (a PTDF/shift-factor screen through `model.scdcopf`). |
| `build_leveraged_queue.py` | Draw a queue restricted to high-leverage sites → `data/<case>/queue.json` for the `lev*` cases. |
| `select_relief_set.py` | Collect the buses that ever shed under full ORDC relief, validate the slim set reproduces the target `S` on held-out configurations → `data/<case>/relief_set.json`.  Called by `cases.prepare_case`. |

## Pipeline

```bash
# network
uv run python -m src.cases.texas.convert_model              # ACTIVSg2000 model -> data/activsg2000.json

# load and renewable snapshots (needs an ERCOT token; see src/ercot/ercot_auth.py)
uv run python -m src.cases.texas.fetch_load_snapshots
uv run python -m src.cases.texas.build_load_snapshots       # -> data/activsg2000_load_collection.json
uv run python -m src.cases.texas.fetch_renewable_snapshots
uv run python -m src.cases.texas.build_renewable_availability

# queue: a leveraged draw at the high-shift-factor buses
uv run python -m src.cases.texas.build_queue_distribution
uv run python -m src.cases.texas.screen_bus_leverage  --case activsg2000_v12_lev3a3
uv run python -m src.cases.texas.build_leveraged_queue --case activsg2000_v12_lev3a3

# relief set + screened contingencies, at the recipe the sweep solves
uv run python -m src.cases.prepare_case               --case activsg2000_v12_lev3a3
```

Then `sweeps.enumerate_configs` (or `sampling.mcmc_sweep` for `levfull`) for the distribution;
see `src/README.md`.

## Notes

**Candidate cost comes from fuel.** The ERCOT queue file records a project's fuel rather than a
marginal cost, so `cases.registry.FUEL_COST` maps fuel → $/MWh when normalizing candidates.
Renewables and storage discharge are zero-cost; thermal costs sit above the existing fleet so
new thermal runs only when cheaper units are exhausted.

**Leverage-first queue construction.** `screen_bus_leverage` ranks buses by their shift-factor
leverage on the priced constraints, and `build_leveraged_queue` draws real ERCOT projects only
at those buses — the siting that produces a queue whose build-out actually moves congestion.
The slim relief set and the recipe it supports are documented in `src/model/README.md`.
