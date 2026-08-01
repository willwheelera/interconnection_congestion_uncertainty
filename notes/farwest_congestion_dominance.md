# FarWest congestion dominance — investigation notes (paused)

**Status:** open, paused 2026-07-10. Diagnosis complete; no fix applied yet.

**One line:** the scalar target `S(x)` over the 2^20 queue hypercube is effectively
single-mechanism — one FarWest internal contingency (`ctg:68:18`) dominates the entire
congested set — which does not reflect the multi-mechanism congestion seen in real ERCOT
data. This note records why, and what a fix would (and would not) take.

## The concern

`S(x) = max over constraints of the load-averaged μ` should expose a rich landscape of
competing congestion mechanisms if it is to be a faithful testbed for tail samplers meant
to generalize to real, many-contingency grids. It does not: only two constraints ever set
the max, one of them strictly dominated.

## Diagnosis (evidence)

Data: `results/activsg2000/S_cache.parquet` (full contiguous 2^20 sweep), the underlying
`results/activsg2000/mu_cache/`, `results/activsg2000/snapmax_cache/`, and
`data/activsg2000_queue20.json` / `data/activsg2000_screened.json` /
`data/activsg2000_load_collection.json`.

1. **Only 2 drivers over all 2^20 configs.** `ctg:68:18` drives 786,432 configs (75%,
   S up to $1531); `ctg:329:330` drives 262,144 (25%, S in ~$153–215, strictly dominated,
   never in the tail). S itself is not two-valued (~6500 distinct values); the *mechanism*
   behind the max is binary.

2. **Not a `max` artifact.** In the full μ vector (mu_cache part 0, 2000 configs) only 17
   constraints ever carry any dual and only 3 ever exceed $50: `ctg:68:18` (dominant, up to
   ~$1192), `ctg:329:330` (near-constant ~$213), `ctg:464:461` ($206–304, never wins). The
   network physically congests ~3 lines under this queue; one dominates.

3. **Tail is one subcube plateau.** Top 0.1% of S = 320 configs, all `ctg:68:18`; ~7 gens
   pinned ON, ~4 pinned OFF, ~9 indifferent (ON-fraction ≈ 0.5). 25 basins, top ~10 all
   peak ≈$1531 — many hill-climb entry points onto one ridge, not distinct mechanisms.
   Effective dimension ≈ 8, not 20.

4. **`ctg:68:18` is entirely inside FarWest.** Monitored branch 18 = bus 1008→1032
   (FarWest), limit 102.9 MW; contingency 68 outages branch 68 = bus 1082→1038 (FarWest).
   LODF(18←68) = +0.728 — near-parallel FarWest paths, so tripping 68 dumps 73% onto 18.

5. **FarWest is clamped identically in every snapshot.** `build_load_snapshots.py` caps
   per-zone scale at `ZONE_SCALE_CAP = 1.5`; FarWest's true 2025 ratio is 4.9×–6.3×
   (actual 6426–8216 MW vs 1307 MW nominal), so it is pinned at 1.5 in all 8 snapshots.
   The one binding zone is maxed the same way regardless of snapshot.

6. **Snapshots are diverse in reality, collapsed in the model.** The 8 snapshots carry 7
   distinct real ERCOT constraints (E_PASP, HMLTN, EDGWD_MR2H, PALE_ESS_TLINE, DPWGBY69,
   138_WBT_WBN, HI_LONE). In the model they nearly all bind `ctg:68:18`; different loading
   changes its *severity* (μ ranges $11 → $32 → $2528 → $3500 across snapshots for all-off)
   not its *identity*. Load-averaging over 8 snapshots is close to a no-op on mechanism.

7. **Queue gens are geographically spread, not clustered.** 20 gens span 978 km of latitude
   vs the grid's 1101 km, median pairwise separation 446 km (Cameron border → Deaf Smith
   panhandle). Geographic spread is not the cause.

8. **Sensitivity is redispatch-driven, not geometric.** Per-gen main effect on S
   (mean S|on − mean S|off over the full sweep): only 8 of 20 gens move S by >$10; 7
   aggravate, 1 relieves. The relief giant is gen 14 (solar, bus 8038, East) at −$1078.
   PTDF/LODF leverage of the queue buses on branch 18|68 does **not** match: the highest-
   leverage gen (gen 5, Wind, bus 1083, FarWest, PTDFc = −0.109) has ~zero effect (−$0.6),
   while gen 14 (PTDFc = −0.003) sets the target. So dμ/dP here runs through economic
   redispatch of cheap (≈$0) renewables, not through the injecting gen's own shift factor —
   consistent with the LP sign/reduced-cost mechanism in `project_texas_relief_curve_ordc`.

9. **The "second mechanism" is a 2-bit switch, not independent congestion.** `ctg:329:330`
   is the driver exactly when `gen14 ON AND gen13 OFF` (precisely 25%: P(329|gen14 on)=0.5,
   P(329|gen13 off)=0.5, both zero otherwise). It is a constant ~$213 floor exposed only
   when gen 14 suppresses the FarWest constraint and gen 13 is not there to counteract.
   The whole landscape reduces to roughly `S ≈ max(f(gens 1,2,3,6,7,11,12,13,14), 213)`.

## Root cause

One dominant network bottleneck (a thin FarWest internal path) sits in the one zone that is
clamped at its maximum scale in every snapshot. Neither the geographically spread queue nor
the 8 real snapshots can diversify the binding constraint, because (a) target sensitivity is
economic-redispatch not location, and (b) the binding zone is stressed identically everywhere.
The 13-of-2756 contingency screen finishes the collapse.

## Candidate fixes and their feasibility

- **(1) Un-clamp / reinforce FarWest to 2025 levels.** The load *target* is free — it is in
  `data/activsg2000_load_snapshots.csv` (4.9×–6.3× nominal). The *network* fix is ill-posed:
  the 2018 FarWest topology (36 loads, 1307 MW, branch limits ~103 MW) cannot import 6× load,
  so removing the clamp just sheds load and pins S at lost-load cost. Reinforcing means
  arbitrary rating bumps, or adding FarWest generation (which overlaps the queue perturbation
  and would itself set the congestion), or sourcing a newer synthetic ERCOT case with the
  Permian buildout. No principled per-line fix exists (synthetic buses ≠ real substations).

- **(2) Sub-zonal (nodal) load.** Not supported by the data: the ERCOT source is 8 weather
  zones + total, nothing finer; load elements carry only `p_load`/`bus`. The current method
  already preserves the model's intra-zone pattern. Per-snapshot within-zone variation would
  need an external space-and-time key — bus `lat`/`lon` → gridded weather (temperature) is the
  only option the case supports, and requires fetching ERA5/NOAA. Moot until (1), since
  reallocating load inside Coast/NorthC cannot move a constraint that lives inside FarWest.

- **(3) Loosen screening.** Keep more than 13 contingencies (see `screen_contingencies.py`).
  Necessary but not sufficient on its own — the collapse is driven by (1).

## Cheapest well-posed next step (resume here)

Test the case's latent bottleneck structure directly — no new data, no arbitrary surgery:

1. Use the **full unscreened** contingency set (`data/activsg2000.json`, not `_screened`).
2. Set **FarWest scale back to 1.0** (remove its always-tightest status).
3. Stress **one different zone to its true ratio** (Coast or NorthC), for a handful of snapshots.
4. Check whether the binding constraint moves **off `ctg:68:18`** onto a Coast/NorthC line.

Outcomes: if it moves, the grid *can* produce multiple mechanisms and the fix lives entirely
in load construction (un-clamp + broaden screen); if it still binds `ctg:68:18` with FarWest
at nominal and another zone maxed, the grid has a single dominant bottleneck and multi-mechanism
congestion requires a different/updated synthetic case — i.e. (1) is unavoidable.

## Pointers

- Pipeline: `build_load_snapshots.py` (ZONE_SCALE_CAP), `screen_contingencies.py` (13/2756),
  `load_averaged_target.py` (add_queue_generators, snapshot_mu), `consolidate_s_cache.py`.
- Related memory: `project_texas_queue_study`, `project_texas_relief_curve_ordc`,
  `project_ercot_congestion_concentration`, `project_ercot_tail_event_definition`.
