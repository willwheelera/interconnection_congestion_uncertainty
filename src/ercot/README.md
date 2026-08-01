# ERCOT data: goals and plan

This directory holds the tooling to pull real ERCOT market data and the plan for
what we do with it. It exists to **validate and refine our congestion/LMP model
against reality**.

## Background: where this fits

So far the project has worked synthetically: sample a distribution of grid
conditions, run optimization (UC/OPF) on a model system, and estimate the
resulting distributions of **congestion dual variables (μ)** and **LMPs (λ)**.
Those distributions depend entirely on the model system and its parameters.

The open question is whether that synthetic picture resembles the real market.
ERCOT publishes the same quantities (LMPs, shadow prices on binding constraints,
settlement point prices) plus the unit-commitment state. This data lets us check
the model against ground truth instead of trusting it by construction.

## Core research questions

1. **Do the tails match?** Compare the distributions of real ERCOT LMPs and
   congestion/shadow prices against our model's. If the tails are similar, the
   current model is worth pursuing further. If they differ, the differences tell
   us how to refine or replace it.

2. **Is high congestion the main driver of high LMPs?** We assume the high-price
   moments on the grid are driven primarily by congestion (the μ term), not by
   the energy/marginal-cost term (λ). Validate this directly in the real data.

3. **Are high-congestion events alike or all different?** When congestion is
   high, is it (nearly) always the same cause — same constraint binding, similar
   load/generation pattern — or is each instance distinct? Equivalently: can a
   **small number of scenarios** capture most high-congestion events, or do
   load/generation configurations vary too much to condense that way? This is the
   crux for whether scenario reduction is viable in the model.

4. **How do we replicate the effect in a model?** Whatever structure we find in
   the real high-congestion events, the end goal is to reproduce it in our model
   system.

## Data strategy: scan, then drill down

**Historical, not real-time.** At this stage — developing models and strategies
— we work from historical data, not the live realtime feeds. We want data that is
**complete and static** so that analysis and testing are consistent and
reproducible: the same query returns the same rows every time, and nothing is
still being revised. This also unlocks the 60-day disclosure datasets (unit
commitment, resource-level dispatch) that are only published on a lag. Realtime
fetching is reserved for later, once the modeling approach is settled.

We do **not** need full, complete data for every interval across the year. Two
phases:

**Phase 1 — Scan (cheap, wide).** Pull the one scalar congestion signal across a
long time span and find where it is extreme:
- Shadow prices on binding transmission constraints (`np6-86`, congestion) —
  small enough to pull for a whole year in one call.
`select_high_congestion_periods.py` flags the **high-congestion moments** (hourly
max shadow price over a threshold, default \$300) and writes them to
`data/ercot/high_congestion_{hours,days}_{base,contingency}.csv` — the candidate
list for Phase 2. LMPs / settlement-point prices are *not* scanned year-wide: at
~1,111 settlement points (or ~19,000 electrical buses) they are drill-down scale,
pulled only for the days Phase 1 flags.

**What the 2025 scan found.** Two distinct congestion regimes, treated separately
throughout this analysis:
- **Base case** (no assumed outage). Extreme congestion is *rare and
  concentrated*: only 67 hours / 33 days clear \$300, and one constraint (E_PASP)
  sets the max in over half of them. This is the scenario-reducible regime, and it
  holds the very highest prices (up to \$8082).
- **Contingency** (constraint monitored under an assumed N-1 outage). Extreme
  congestion is *pervasive*: 4,767 hours / 361 days clear \$300, and it is mostly
  genuine — only ~2% pinned at the \$3500 shadow-price cap.
Whether to prioritize base or contingency for *modeling* is deferred to the
Phase 2 settled-price drill-down. The open empirical question: do contingency
shadow prices actually drive settled LMPs, or do they wash out? If they drive
prices, a faithful tail model needs contingency-scenario sampling (a
security-constrained / N-1 formulation), not just base-case loading scenarios.

**Phase 2 — Drill down (rich, narrow).** For a handful of critical days flagged in
Phase 1 — chosen as *separate, non-contiguous delivery dates* that span both
regimes (some base-driven, some contingency-driven, some mixed) so a settled-price
spike can be attributed to one regime — download fuller context to explain *why*
congestion was high. In priority order:
- **Settled vs. SCED prices first (cheap).** Node/zone/hub LMP (`np6-788`, 5-min
  SCED, ~1,111 settlement points) and settlement-point prices (`np6-905`, 15-min)
  for the flagged days. This is the validation step — does a high shadow price show
  up as a high settled price, and under which regime? The 5-min LMP aligns in time
  with the SCED shadow prices; SPP is the "what actually settled" cross-check.
  `select_high_congestion_periods.py` writes `sample_days.csv` — days chosen to
  isolate the regimes (base-driven, contingency-driven, and mixed, labeled at the
  hour level) — and `fetch_sample_days.py` downloads these two feeds for those days.
- **Per-bus spatial gradient (expensive, only when needed).** LMP by electrical
  bus (`np6-787`, ~19,329 buses, ~17× the node/zone/hub feed) for the spatial price
  structure *around a specific binding constraint*. Pull only for the flagged hours
  of a few targeted days, never whole days across all buses.
- Load (actual system load by zone: `np6-345`, `np6-346`).
- Generation / unit commitment (which units were committed/online each period —
  `np3-965` SCED gen resource data, `np3-966` DAM gen resource data, COP
  snapshots) for the on/off context our model needs.
- Network topology context to locate the binding constraints geographically.

### Phase 2 progress: how settled prices relate to base vs contingency

These are early results on the 12 high-congestion sample days; they are suggestive,
not settled conclusions — the questions below stay open.

- **Settled prices carry the same congestion signal as SCED.**
  `correlate_settled_congestion.py` (5-min LMP) and `settlement_regime_split.py`
  (15-min settlement-point prices, aligned to the shadow prices) measure the
  congestion footprint as cross-node price spread (congestion separates nodal
  prices; the energy/scarcity term lifts them together). The SPP and LMP spreads
  correlate at 0.99, so the actual settled price tracks the SCED price for
  congestion purposes.
- **On these tail days, settled separation almost always coincides with a
  contingency binding.** Tagging each congested 15-min interval by which regime is
  binding gives contingency-only ≈ 72%, both ≈ 27%, base-only ≈ 1.5%, neither = 0%
  (`results/settlement_regime_when.png`). The base-only intervals cluster in midday
  hours on a single day. The `neither = 0` is a useful consistency check: no settled
  separation appears without some constraint binding.
- **Base-case is sparse, not absent.** A naive LMP-spread vs base-μ correlation is
  weak (ρ ≈ 0.11) because base constraints bind in a minority of intervals; restricted
  to intervals where base actually binds it rises to ≈ 0.30. The large spreads at
  zero base-μ are explained by contingency μ (SCED is security-constrained, so
  contingency shadow prices enter the LMP), not by missing data.
- **DC price formation holds empirically.** `recover_shift_factors.py` fits the
  hub-differenced LMP on the μ vector; per-day this reaches R² ≈ 0.99, confirming
  LMP_n − hub ≈ −Σ SF·μ. A pooled effective shift-factor matrix is partly
  recoverable but limited by collinearity (co-binding constraints are not separately
  identified) and transfers cleanly only to the high-congestion days.

Taken together these *suggest* that contingencies matter for the high-price tail and
are worth modeling explicitly — but we are not closing the base-vs-contingency
question on this evidence.

**Next steps:**
1. **Toward reproducing the effect in a model.** Reproducing the observed congestion
   needs a security-constrained (N-1) DCOPF, which needs a network model (incidence +
   susceptances). The monitored-constraint data alone is not enough: its lines form a
   fragmented forest on *substation* nodes that do not name-match the settlement
   price nodes. The missing link is a settlement-point ↔ electrical-bus mapping (with
   the `np6-787` bus list) plus a fuller branch list.
2. **Characterize the contingencies.** Map each `ContingencyName` to the element it
   outages, find which contingencies recur in the tail, and describe their flow
   (`Value`) and price signatures — the input to modeling them in test cases.
3. **Probe the shadow-price cap.** Many contingency intervals pin at the $3500 cap;
   check whether the settled spread keeps rising past it or saturates (tail censoring).

## Analysis and outputs

- Statistical plots of the **tails** of LMP and congestion-price distributions
  (real vs. model), and their **correlations** (e.g. LMP vs. shadow price).
- Place results in the context of **network topology** — where on the grid the
  congestion concentrates.
- Characterize high-congestion events: cluster them and ask whether a few
  representative scenarios reproduce the bulk of the tail.

## Endpoints currently fetched

`fetch_ercot.py` pulls four real-time datasets (see its docstring for the SCED
vs. delivery-date filtering and timezone handling):

| Save name            | Endpoint                                | Content                                  | Cadence |
|----------------------|-----------------------------------------|------------------------------------------|---------|
| `lmp_electrical_bus` | `/np6-787-cd/lmp_electrical_bus`        | LMP by electrical bus                    | 5-min   |
| `lmp_node_zone_hub`  | `/np6-788-cd/lmp_node_zone_hub`         | LMP by node/zone/hub                     | 5-min   |
| `shadow_prices`      | `/np6-86-cd/shdw_prices_bnd_trns_const` | Shadow prices on binding constraints     | SCED    |
| `spp_node_zone_hub`  | `/np6-905-cd/spp_node_zone_hub`         | Settlement point prices                  | 15-min  |

`fetch_ercot.py` defaults to the Phase 1 year scan (`scan_year`, shadow prices
only); its `example()` pulls a one-hour window across all four endpoints to
exercise the plumbing. `select_high_congestion_periods.py` then turns the scan
into the Phase 2 candidate-day CSVs and `sample_days.csv`, and
`fetch_sample_days.py` downloads the node/zone/hub LMP + SPP feeds for those days
(reusing `fetch_ercot.py`'s general `fetch_days` / `PHASE2_DATASETS`).
Three analysis scripts consume those feeds plus the shadow prices:
`correlate_settled_congestion.py` (LMP-spread vs base/contingency μ),
`settlement_regime_split.py` (15-min SPP, regime tagging and timing), and
`recover_shift_factors.py` (empirical shift-factor fit and out-of-sample test). Not
yet wired: the per-bus drill-down (`np6-787`) and the load / generation /
commitment endpoints — including the 60-day disclosure datasets — noted above.

## Auth

`ercot_auth.py` handles login via getpass and caches a short-lived bearer token
to `~/.ercot_token.json` (chmod 600). `fetch_ercot.py` uses only that token — the
password is never stored in the environment or on disk.
