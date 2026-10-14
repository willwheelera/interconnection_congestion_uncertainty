# cases — testbed definitions

`registry.py` is the single case registry; see `src/README.md` for what a `Case` record
carries and how downstream stages read it.  The rest of this package builds the testbeds
that the registry points at.

| file | role |
|---|---|
| `registry.py` | Every case: model, paths, recipe, queue.  Read by every stage; `--case` accepts a name, a name substring, or `all`. |
| `build_ieee_queue.py` | Builds `case14/30/118_queue`: incumbents, candidate siting/cost/capacity/build probability, provisional ratings, N-1 contingency set. |
| `calibrate_ratings.py` | Relaxes ratings round by round until the incumbent-only grid is N-1 secure. |
| `prepare_case.py` | The two sweep inputs in order: the slim relief set, then the screened contingencies, at the case's own recipe. |
| `screen_contingencies.py` | Drops contingencies that never bind across stress configurations, at the case's own recipe.  Approximate — for fast runs, not for ground truth. |
| `texas/select_relief_set.py` | Chooses the slim ORDC relief bus set for the ACTIVSg2000 cases (used by `prepare_case`).  The one-time ACTIVSg2000 construction from ERCOT and MATPOWER sources lives on the `archive` branch. |

## The incumbent-fleet IEEE testbeds

`case14_queue`, `case30_queue`, and `case118_queue` are small, fast examples of the same setup
as the ACTIVSg2000 cases: the network's own generators stay permanently online and the queue
is added on top (an incumbent fleet that the build-out perturbs), and all three are N-1
security-constrained DCOPF under the same recipe as the Texas cases.  This makes them
enumerable stand-ins that exercise the whole pipeline in minutes.  `case14_queue` and
`case30_queue` ship with committed inputs on this branch; `case118_queue` is buildable from
`build_ieee_queue` (its committed inputs are on `archive`).

| case | buses | branches | contingencies | incumbents | candidates | configurations |
|---|---|---|---|---|---|---|
| `case14_queue` | 14 | 20 | 19 | 5 | 14 | 2^14 = 16,384 |
| `case30_queue` | 30 | 41 | 38 | 6 | 20 | 2^20 = 1,048,576 |
| `case118_queue` | 118 | 186 | 177 | 54 | 20 | 2^20 = 1,048,576 |

## Pipeline

```bash
uv run python -m src.cases.build_ieee_queue     --case case30_queue   # data/<case>.json + candidates
uv run python -m src.cases.calibrate_ratings    --case case30_queue   # rewrites data/<case>.json
uv run python -m src.cases.screen_contingencies --case case30_queue   # optional, fast runs only
```

Then `sweeps.enumerate_configs` for the ground truth; see `src/README.md`.

## Design decisions worth knowing

**Ratings are a security criterion, not a congestion criterion.** Each line starts rated at
its incumbent-only flow plus 5% headroom, then `calibrate_ratings` relaxes only lines that
are overloaded or priced above `SECURE_FRAC` of the shadow-price cap.  Relaxing *every*
binding line instead inflates exactly the corridors that should stay tight, and the finished
case then shows no congestion under buildout either — this was observed and is the reason
for the threshold.

**The incumbent grid is secure but not uncongested.** Driving all-off congestion to exactly
zero costs so much rating headroom that buildout stops congesting.  The cases settle with
all-off max mu in the tens of $/MWh, well under the $3500 cap, leaving the whole range above
that for the queue to drive.

**The cost band is wide on purpose.** A congestion dual is bounded by the redispatch cost
differential available to relieve the line.  A narrow fleet band (the IEEE cases' native
spread) caps mu at a few $/MWh regardless of the network.  The `$10–150/MWh` band is both
realistic for a real merit order and wide enough for a graded mu distribution.

**Screening is approximate, so the ground-truth sweep does not use it.** The kept set is a
union over stress configurations, and a configuration outside them can bind a dropped
contingency: on case14_queue, one configuration in 62 does, and its max mu comes out 2% low.
The full set costs only ~12% more on case30 and 2.3x on case118, so `enumerate_configs`
defaults to it and screens only with `--screened`.  Interactive solves still prefer the
screened case when one exists, which is what makes them fast.

**Screen at the recipe you solve at.** Which contingencies bind depends on the emergency
factor, the relief available, and the cap, so `screen_contingencies` drives its stress
configurations through the same `model.scdcopf` path a sweep uses.  A set screened at a case
JSON's native (looser) rating omits binders the tightened factor needs.  Screening draws a
random spread of stress configurations, so re-running will not reproduce an earlier set
exactly — the sweep's own augmentation (see `src/README.md`) is what makes the ground truth
exact regardless.

**Candidate capacity and siting.** Each candidate is 6–14% of system load (total queue ≈ 2x
load, in line with real interconnection queues), sited by greedy farthest-point sampling on
the reactance-weighted graph seeded from the incumbent generator buses — remote from existing
generation and spread out, which is the export-corridor geometry that produces distinct
congestion modes.

## Enumeration cost

Warm persistent workers dominate: once a worker has built its model, a configuration is a
bound change plus a warm re-solve.  On 18 workers against the full contingency set,
`case14_queue` (2^14) finishes in well under a minute and `case30_queue` (2^20) in roughly
two hours; `case118_queue` runs about 2.3x slower per solve than case30.

Don't extrapolate from a short run — a 10-configuration timing is almost entirely model build
and reads ~200x too pessimistic.
