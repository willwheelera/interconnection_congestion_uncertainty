# model — the physical model

One generator configuration in, congestion duals out.  Everything above this package treats
a configuration as a bit vector and gets back `{constraint_id: mu}`; everything below it is
EGRET.

| file | role |
|---|---|
| `scdcopf.py` | Builds the N-1 security-constrained DCOPF once and warm re-solves it per configuration.  `build_persistent`, `PersistentModel.solve`, `extract_mu`, `build_audit_ptdf`. |
| `relief.py` | The ModelData edits the recipe is made of: emergency ratings, queue generators, ORDC-shaped nodal relief, zone/renewable scaling. |
| `soft_limits.py` | `enable_soft_limits` — the mu cap via EGRET's lazy violation slacks; `thermal_violations` — the overload it bought, in MW. |
| `evaluator.py` | `ConfigParallel` / `SnapshotParallel` over one shared worker body; `for_case` picks by the case's parallel axis. |
| `dcopf.py` | Plain DCOPF for the legacy `gen_per_bus` testbeds. |

Constraint ids are `base:{line}` for normal thermal limits and `ctg:{contingency}:{line}`
for post-contingency ones, in both case families.

## The recipe

Three ingredients make the mu distribution realistic rather than degenerate.  They live in
`cases.registry.Recipe`, per case, so no environment variable selects them and no script
has its own copy.  The derivation is in `notes/ercot_price_formation.md`.

| constant | value | meaning |
|---|---|---|
| `emergency_factor` | 1.2 | post-contingency limit = factor x long-term rating |
| `ordc_min`, `ordc_max` | 500, 5000 | nodal relief price floor / ceiling ($/MWh) |
| `ordc_blocks` | 8 | relief piecewise blocks |
| `shadow_price_cap` | 3500 | mu cap ($/MWh) |
| `slack_cost` | 9000 | EGRET system-wide slack, above the shed curve |

**1. Emergency factor 1.2.** At factor 1.0 the duals increase to millions; ~1.2 is the
usable band where a handful of constraints bind at realistic magnitudes.  An emergency
factor is used in practice, allowing modest overloads in response to an outage that can be
resolved by corrective action.  The ACTIVSg2000 case JSONs carry `rating_emergency = 1.5 x
long_term` baked in by `cases.texas.convert_model`; the 1.2 is applied at model-build time
by `relief.set_emergency_factor`, so it never appears in the data files.  **Any screening or
diagnostic that reads a case JSON without that override is solving at 1.5.**

**2. ORDC-shaped nodal relief.** EGRET's single system-wide slack cannot represent a pocket
that is unservable only under one contingency; without a local outlet such a case goes
infeasible instead of pricing.  A relief generator at each load bus can inject up to that
bus's demand, so the pocket is relieved locally and the relief price becomes its ceiling.
Pricing that relief on a rising `$500 → $5000` curve rather than one flat VOLL is what
makes mu spread over a range instead of pinning at a single level: shallow shed is charged
near the floor, and the marginal block that clears sets the local price.

**3. $3500 shadow-price cap.** ERCOT enforces a Max Shadow Price — a constraint whose dual
would exceed the cap is allowed to *overload* (soft slack, penalty `cap x baseMVA`) rather
than pay unbounded redispatch.  This bounds every reported mu at exactly `$3500`.  Price
ordering is overload `$3500` < load shed `$5000` < system slack `$9000`, so the model
overloads a line before shedding load before the last-resort slack.

## Slim relief

A full relief generator at every load bus enlarges the LP ~5x — an inactive relief column is
still factorized on every warm re-solve.  Only 9 of ACTIVSg2000's 1125 load buses ever shed,
so relief is restricted to those.

The set is a sweep **input**, not an output: it lives in `data/<case>_relief_set.json` and is
loaded by `Case.relief_buses()`, which returns `relief.ALL_LOAD_BUSES` for a case that has no
set yet.  `relief_buses` is required — no default and no sentinel — at both `build_persistent`
and `evaluator.for_case`, because it decides which model gets built and every caller already
has the `Case` in hand.  All of them pass `case.relief_buses()` except
`cases.texas.select_relief_set`, which exists to choose the set and so passes the one under
test, or `ALL_LOAD_BUSES` for the full-relief baseline it measures against.
A sweep additionally records per configuration the worst line overload and the system-wide
shed (`snapmax_cache` / `stress_cache`), so any configuration needing a bus outside the set
would show up as a large overload or nonzero shed.  Over the full `2^20` Texas sweep the max
overload was 5.64 MW and no configuration shed — the slim set is proven complete.

## Warm re-solves and the parallel axis

Building the SCDCOPF is expensive; solving a configuration against an already-built model is
cheap.  So `build_persistent` builds once with **every** candidate present and
`PersistentModel.solve` applies a configuration by setting each candidate's upper bound to
its capacity or to zero and warm re-solving.  That carries the basis, the PTDF
factorization, and the accumulated lazy contingency constraints across configurations, which
is what makes a `2^20` sweep affordable rather than a multi-day job.

`evaluator.for_case` then picks the axis from the case record:

- **config-parallel** (`case.parallel == "config"`): one load level, so `n_workers`
  identical models take whole configurations off a shared queue.  The IEEE queue cases.
- **snapshot-parallel** (`"snapshot"`): mu is averaged over a collection of load snapshots,
  so there is one worker per snapshot and each configuration is broadcast to all of them.
  Each worker's model is pinned to its snapshot's scaled loads and derated renewables.  The
  Texas cases, 8 workers for 8 snapshots.

With 8 snapshot workers on 20 cores, cap Gurobi threads per worker (`--threads`) to avoid
oversubscription.  A worker draws only ~115% CPU anyway — warm-started dual simplex is
largely sequential, so extra threads buy little and sharding the configuration range across
jobs beats making one job bigger.

## Reproducibility

Duals here are frequently degenerate, and under lazy PTDF a constraint only carries a dual
once it has been added to the model — which depends on the order a warm worker happened to
see configurations in.  Two identical runs therefore differ slightly.  See the note at the
end of `src/README.md` before regenerating a committed `mu_cache`.
