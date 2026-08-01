# cases/texas — building the ACTIVSg2000 testbeds

One-time construction of the three Texas cases from MATPOWER, PowerWorld, and ERCOT
sources.  Their outputs are the `data/activsg2000*` files that `cases.registry` points at,
and they are already committed — rerun these only to rebuild an input.

The synthetic ACTIVSg2000 grid is 2000 buses, 3206 branches, and 544 base generators.  A
*configuration* is a length-20 bit vector over ERCOT interconnection-queue candidates, each
either built or not.  The three registered cases differ only in the incumbent fleet:

| case | fleet | notes |
|---|---|---|
| `activsg2000_original` | the model's own ~2016 fleet | historical run at the JSON's native 1.5 emergency factor; kept for its enumerated cache |
| `activsg2000_original_v12` | the same ~2016 fleet | re-solved under the current factor-1.2 soft-cap recipe |
| `activsg2000_2025` | 2025 fleet (CDR additions 2017–2025) | the primary Texas case |

The physical model and the recipe they are solved under are documented in
`src/model/README.md`, not here.

## Pipeline

```bash
# network
uv run python -m cases.texas.convert_model              # MATPOWER + .aux -> data/activsg2000.json
uv run python -m cases.texas.compile_resource_additions # ERCOT CDR unit tables -> additions CSV
uv run python -m cases.texas.add_cdr_resources          # -> data/activsg2000_2025.json

# load and renewable snapshots (needs an ERCOT token)
source gridstatus_api_key.sh                            # see src/ercot/ercot_auth.py
uv run python -m cases.texas.fetch_load_snapshots --n 8
uv run python -m cases.texas.build_load_snapshots       # -> data/activsg2000_load_collection.json
uv run python -m cases.texas.fetch_renewable_snapshots
uv run python -m cases.texas.build_renewable_availability   # per-fuel capacity factor per snapshot

# queue
uv run python -m cases.texas.build_queue_distribution   # -> data/activsg2000_queue20.json

# contingency screening, at the recipe the sweep solves
uv run python -m cases.screen_contingencies --case activsg2000_2025

# slim nodal relief set (see src/model/README.md)
uv run python -m cases.texas.select_relief_set --case activsg2000_2025
```

Then `sweeps.enumerate_configs` for the ground truth; see `src/README.md`.

## Files

| file | role |
|---|---|
| `convert_model.py` | MATPOWER + PowerWorld `.aux` → EGRET JSON with coordinates, ERCOT weather zones, emergency ratings, and the N-1 contingency set. |
| `compile_resource_additions.py` | Compile ERCOT resources added 2017–2025 from the CDR unit-detail tables. |
| `add_cdr_resources.py` | Add those additions to ACTIVSg2000 as new generators → the 2025 fleet. |
| `fetch_load_snapshots.py` | Pull ERCOT actual zonal load for the high-congestion hours. |
| `build_load_snapshots.py` | Turn those into per-zone load scale factors (8 snapshots). |
| `fetch_renewable_snapshots.py` | Pull ERCOT actual wind and solar output for the same hours. |
| `build_renewable_availability.py` | Attach a per-fuel capacity factor to each snapshot. |
| `build_queue_distribution.py` | Select 20 ERCOT queue projects, match them to buses geographically, set each candidate's Bernoulli `p`. |
| `select_relief_set.py` | Collect the buses that ever shed under full relief, validate the slim set reproduces `S` on held-out configurations → `data/<case>_relief_set.json`. |

`diagnostics/identify_relief_set.py` is the earlier timing/certification variant of the
last one, kept as a diagnostic.

## Notes

**The candidate queue is shared.** All three cases use `data/activsg2000_queue20.json` and
`data/activsg2000_load_collection.json`; only the incumbent fleet differs.  That is what
makes the two fleets a controlled comparison — same network, same queue, same recipe — and
it is the basis of the descriptor-instability result in `notes/paper_plan.md` §5.

**Candidate cost comes from fuel.** The ERCOT queue file records a project's fuel rather
than a marginal cost, so `cases.registry.FUEL_COST` maps fuel → $/MWh when normalizing
candidates.  Renewables and storage discharge are zero-cost; thermal costs sit above the
existing fleet's ~$17–19 so new thermal runs only when cheaper units are exhausted.

**FarWest stays capped at 1.5.** The zone-scale cap was left where it is rather than
rebuilding the network: this case is a proof of concept, not a claim to be "2025 ERCOT."
2025 congestion in the real system is rural 115 kV (Llano, Val Verde, Washington, Kenedy,
Brazoria), not FarWest.
