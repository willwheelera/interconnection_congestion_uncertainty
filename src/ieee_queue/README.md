# ieee_queue

Incumbent-fleet interconnection-queue testbeds on the IEEE 14, 30, and 118-bus networks,
with exhaustively enumerated N-1 SCDCOPF ground truth.

These exist to remove two confounds in the older `case14_gen_per_bus` / `case30_gen_per_bus`
testbeds (see `notes/paper_plan.md` §3):

1. **No incumbent fleet.** There, every generator is a queue candidate, so the all-off
   configuration is an empty grid served entirely by priced load shed. Here the IEEE
   network's own generators stay permanently online and the queue is added on top, matching
   the Texas cases where a queue perturbs a real fleet.
2. **DCOPF vs SCDCOPF.** There the model is plain DCOPF while Texas is N-1
   security-constrained, so comparisons confound network size with physical model. Here all
   three cases are N-1 SCDCOPF under the same recipe as Texas: emergency factor 1.2,
   ORDC-shaped nodal shed ($500 → $5000), and a $3500 shadow-price cap via soft thermal
   limits.

## The cases

| case | buses | branches | contingencies | incumbents | candidates | configurations |
|---|---|---|---|---|---|---|
| `case14_queue` | 14 | 20 | 19 | 5 | 14 | 2^14 = 16,384 |
| `case30_queue` | 30 | 41 | 38 | 6 | 20 | 2^20 = 1,048,576 |
| `case118_queue` | 118 | 186 | 177 | 54 | 20 | 2^20 = 1,048,576 |

Every candidate has a linear cost drawn from the same `$10–150/MWh` band as the incumbent
fleet, so a build changes the merit order rather than sitting idle, and which projects are
built determines which corridors congest.

## Pipeline

Run in order. Steps 1–3 are quick; step 4 is the long one.

```bash
# 1. build the case and its queue (data/<case>.json, data/<case>_candidates.json)
uv run python -m ieee_queue.build_queue_case

# 2. relax ratings until the incumbent-only grid is N-1 secure (rewrites data/<case>.json)
uv run python -m ieee_queue.calibrate_ratings

# 3. screen the contingency set (optional: fast diagnostic runs only, see below)
uv run python -m ieee_queue.screen_contingencies

# 4. exhaustive ground truth -> results/<case>/mu_cache/part_*.parquet
uv run python -m ieee_queue.enumerate_configs --case case14_queue  --workers 8
uv run python -m ieee_queue.enumerate_configs --case case30_queue  --workers 16
uv run python -m ieee_queue.enumerate_configs --case case118_queue --workers 16

# 5. reduce to the common caches the samplers read
uv run python -m tail_sampling.build_target_cache --case case30_queue
uv run python -m tail_sampling.build_dual_cache   --case case30_queue

# 6. run the sampling experiments
uv run python -m tail_sampling.run_experiments --case case30_queue
```

Steps 1–3 are re-runnable and idempotent. Step 4 skips parts that already exist, so an
interrupted sweep resumes where it stopped.

## Files

| file | role |
|---|---|
| `build_queue_case.py` | Builds all three cases: incumbents, candidate siting/cost/capacity/build probability, provisional ratings, N-1 contingency set. |
| `calibrate_ratings.py` | Relaxes ratings round by round until the incumbent-only grid needs no overload or shed and prices nothing above `SECURE_MU`. |
| `screen_contingencies.py` | Drops contingencies that never bind across stress configurations. Approximate — for fast diagnostic runs, not for ground truth. |
| `sample_harness.py` | Config-parallel warm-started SCDCOPF evaluator; the single place the modeling recipe lives. |
| `enumerate_configs.py` | Sweeps every configuration, streaming duals to a resumable long-format `mu_cache`. |

## Design decisions worth knowing

**Ratings are a security criterion, not a congestion criterion.** Each line starts rated at
its incumbent-only flow plus 5% headroom, then `calibrate_ratings` relaxes only lines that
are overloaded or priced above `SECURE_MU` (5% of the cap). Relaxing *every* binding line
instead inflates exactly the corridors that should stay tight, and the finished case then
shows no congestion under buildout either — this was observed and is the reason for the
threshold.

**The incumbent grid is secure but not uncongested.** Driving all-off congestion to exactly
zero costs so much rating headroom that buildout stops congesting. The cases settle with
all-off max mu in the tens of $/MWh, well under the $3500 cap, leaving the whole range above
that for the queue to drive.

**The cost band is wide on purpose.** A congestion dual is bounded by the redispatch cost
differential available to relieve the line. A narrow fleet band (the IEEE cases' native
spread) caps mu at a few $/MWh regardless of the network. The `$10–150/MWh` band is both
realistic for a real merit order and wide enough for a graded mu distribution.

**Screening is approximate, so the ground-truth sweep does not use it.** The kept set is a
union over stress configurations, and a configuration outside them can bind a dropped
contingency: on case14, one configuration in 62 does, and its max mu comes out 2% low. The
full set costs only ~12% more on case30 and 2.3x on case118, so `enumerate_configs` defaults
to it and screens only with `--screened`. `sample_harness` still prefers the screened case
when one exists, which is what makes interactive checks fast.

**Candidate capacity and siting.** Each candidate is 6–14% of system load (total queue ≈ 2x
load, in line with real interconnection queues), sited by greedy farthest-point sampling on
the reactance-weighted graph seeded from the incumbent generator buses — remote from
existing generation and spread out, which is the export-corridor geometry that produces
distinct congestion modes.

## Enumeration cost

Warm persistent workers dominate: once a worker has built its model, a configuration is a
bound change plus a warm re-solve. On 18 workers against the full contingency set,
`case14_queue` (2^14) finishes in well under a minute and `case30_queue` (2^20) in roughly
two hours; `case118_queue` runs about 2.3x slower per solve than case30.

Don't extrapolate from a short run — a 10-config timing is almost entirely model build and
reads ~200x too pessimistic.

`enumerate_configs` still writes resumable parts, so an interrupted 2^20 sweep picks up
where it stopped.
