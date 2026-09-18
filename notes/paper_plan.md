# Short paper: planning notes

Working notes for a short paper. Claims, outline, evidence needed, and the task list to produce
the missing data. Section 5 is a separate running list of things to explore — nothing there is
committed to the paper.

## 1. The claim set

**Setting.** An interconnection queue defines a probability distribution over which candidate
generators are built. Each outcome is a bit vector `x` over `ngen` candidates. Solving the
network at `x` gives congestion duals `mu_l(x)` on every line constraint. The quantity of
interest is a per-element tail probability

    P_l = P[ mu_l > c ]   (probability over queue outcomes)

i.e. for each line, how likely is it that queue buildout leaves that line severely congested.

**Claim 1 (positive, and deliberately unsurprising).** Direct Monte Carlo over queue outcomes
gives usable per-line congestion-risk probabilities at a modest number of OPF solves, across
networks from 14 buses to 2000 buses with N-1. It is the correct baseline and it works.

**Claim 2 (negative, the main technical content).** If Claim 1 is trivial then the probabilities
should be obtainable without sampling, from one solve plus network structure. They are not.
Static descriptors from a single base configuration — proximity to limit (`slack = rating -
|flow|`) and generator-to-line PTDF swing (`sigma`, `z = slack/sigma`) — fail to predict the
sampled per-line statistics. Specifically: which descriptor ranks lines is not stable across
cases (proximity on case30 and TX-2025; PTDF swing on TX-v12, *same network and same queue*,
only the incumbent fleet differs), apparent fits are driven by single outliers, and no descriptor
gives magnitude. The failure mode is that `mu` is set by combinatorial active-set structure that
a single configuration cannot see.

**One-line abstract.** Queue-outcome sampling gives per-line congestion-risk probabilities
cheaply, and single-configuration network descriptors cannot replace it.

**Not a claim.** The rare-event samplers (subset simulation, ascent-seeded importance sampling,
SMC tilting, cross-entropy) do not beat direct sampling at matched solver-call budget. That is a
null result about known methods, not an advance; it stays out of the paper. It only justifies, in
one sentence if needed, why direct sampling is the method used.

## 2. What already exists

| artifact | where | status |
|---|---|---|
| tail-sampling harness (6 samplers, 4 distributions, exact enumerated truth) | `src/sampling/` | done |
| exhaustive `2^20` duals, TX-2025 and TX-v12 (N-1 SCDCOPF) | `results/activsg2000_2025`, `results/activsg2000_original_v12` | done |
| exhaustive duals, case14 (`2^14`) and case30 (`2^20`) DCOPF | `results/case14_gen_per_bus`, `results/case30_gen_per_bus` | done |
| descriptor study, 4 cases x {mean, quantile, exceed} targets | `src/descriptors/` | done |
| per-line survivor / exceedance-strip plots | `src/figures/mu_distributions/` | done |
| ERCOT empirical congestion study (concentration, contingency dominance) | `src/ercot/` | done, motivation only |
| N-1 contingency screening | `src/cases/screen_contingencies.py` | done for TX and the IEEE queue cases |
| incumbent-fleet IEEE queue testbeds (build, calibrate, screen, SCDCOPF harness, sweep) | `src/cases/`, `src/model/`, `src/sweeps/` | built; see its README |
| exhaustive duals, case14-queue (`2^14`, N-1 SCDCOPF) | `results/case14_queue` | done |

## 3. Test case matrix

Two problems with the current cases:

1. **case14 / case30 have no incumbent fleet.** Every generator is a queue candidate, so the
   all-off configuration is an empty grid served entirely by load shed at a uniform price. The
   low-`n_on` part of the space is a shed-priced artifact, not a grid. TX has a 544-generator
   incumbent fleet that the queue perturbs — the real setting.
2. **case14 / case30 are plain DCOPF, TX is N-1 SCDCOPF.** The comparison across cases confounds
   network size with physical model. ERCOT evidence says contingency constraints dominate real
   congestion, so the IEEE cases should be security-constrained too.

| case | buses | incumbent fleet | queue | model | ground truth | status |
|---|---|---|---|---|---|---|
| case14-queue | 14 | yes (5 original IEEE gens) | 14 | SCDCOPF, full N-1 (19) | exhaustive `2^14` | **done** |
| case30-queue | 30 | yes (6 original IEEE gens) | 20 | SCDCOPF, full N-1 (38) | exhaustive `2^20` | **done** |
| case118-queue | 118 | yes (54 original IEEE gens) | 20 | SCDCOPF, full N-1 (177) | exhaustive `2^20` | **done** |
| ACTIVSg2000-2025 | 2000 | yes (2025 fleet) | 20 | SCDCOPF | exhaustive `2^20` | done |
| ACTIVSg2000-v12 | 2000 | yes (2016 fleet) | 20 | SCDCOPF | exhaustive `2^20` | done |

Built by `src/cases/` (see its README for the pipeline and the design decisions). The
rules the three new cases follow:
- the original case's generators are the incumbent fleet, permanently online; queue candidates
  are added on top;
- candidates are sited by farthest-point sampling on the reactance-weighted graph seeded from
  the incumbent generator buses — remote from existing generation and spread out, so different
  configurations load different corridors;
- candidate costs are linear and spread over the same `$10-150/MWh` band as the incumbents, so a
  build changes the merit order rather than sitting idle. The band is wide on purpose: a
  congestion dual is bounded by the redispatch cost differential available, so the IEEE cases'
  native narrow cost spread caps `mu` at a few $/MWh no matter what the network does;
- ratings are a *security* criterion, not a congestion one: each line starts at its
  incumbent-only flow plus 5% and is then relaxed until the incumbent-only grid needs no
  overload or shed and prices nothing above 5% of the cap. Relaxing every binding line instead
  inflates exactly the corridors that should stay tight and leaves the case uncongested under
  buildout too;
- same recipe as TX so the five cases are comparable: emergency factor 1.2, ORDC-shaped nodal
  shed ($500 to $5000), $3500 shadow-price cap via soft thermal limits, system slack above both;
- `ngen` 14 / 20 / 20, so exhaustive enumeration stays feasible and every estimate is scored
  against exact truth.

Keep the old case14/case30 (no incumbents, DCOPF) as at most an appendix line; do not re-run them.

## 4. Outline

**Title** (working): *Sampling interconnection-queue outcomes for per-line congestion risk, and
why single-configuration descriptors cannot replace it*

**I. Introduction.** Queue uncertainty as the driver of future congestion. Planners want per-line
risk, not one deterministic case. Contributions: (a) the sampling formulation and that direct
Monte Carlo suffices; (b) evidence that cheap structural predictors fail.

**II. Problem formulation.**
- Queue outcome space and the build-probability distribution (independent Bernoulli with
  calibrated probabilities; MRF coupling as a robustness check).
- Physical model: N-1 security-constrained DCOPF, emergency factor, shadow-price cap, priced load
  shed. Reproducing contingency `mu` needs *one simultaneous* SC-DCOPF, not independent
  per-contingency DCOPFs.
- Target: per-element `P[mu_l > c]`.

**III. Test systems.** The five-case table; construction, contingency screening, enumeration cost.

**IV. Direct sampling works.** Per-line exceedance estimates vs exact enumeration, all five cases;
error vs solve budget.

**V. Static descriptors do not predict the sampled statistics.** Rank and value figures across
cases; the instability result (which descriptor ranks best flips between the two TX fleets); the
outlier-driven fit; interpretation via active sets.

**VI. Conclusion.** Sampling is the tool that works; structural intuition does not substitute for
it.

**Figures (~5).**
1. Test systems, queue candidates marked.
2. Per-line exceedance: sampled vs exact, grid over cases.
3. Estimate error vs solve budget.
4. Descriptor rank-rank scatter grid (exists: `mu_static_descriptors_*_rank.png`).
5. Descriptor raw-value grid (exists: `..._values.png`).

## 5. Things to explore (not paper content unless a result earns it)

A running list. Each entry: what to try, what to compute, how to judge whether it is interesting.

**5.1 Descriptors combined with sampling.** Descriptors fail as standalone predictors (Claim 2).
Open question: do they carry any usable information *given* a sample, so that a fixed number of
OPF solves yields a better per-line estimate.
- First measurement, before building anything: on the enumerated caches, per line, how strongly
  does `mu_l > c` co-occur with cheap quantities available without solving — `slack_l`, `sigma_l`,
  and a no-solve flow estimate that assumes a fixed dispatch response (each candidate injecting
  its capacity, incumbents backing down in proportion to their base dispatch), giving line flow as
  a linear function of the bits.
- Judge on that alone: if the per-line correlation is near zero on most cases, stop — there is
  nothing to exploit and no estimator will help.
- If the correlation is substantial, then build an estimator that uses it and measure error at
  matched solve budget against exact truth. Only a clear and consistent reduction across cases is
  worth writing up.

**5.2 More descriptors.** The negative result rests on three descriptors. Cheap additions:
`mu` at the base configuration itself; LODF-weighted exposure for contingency constraints; the
count of candidates that can load a line at all. Judge: do any of them rank lines consistently
across all five cases. Consistency across cases is the bar — a descriptor that works on one case
is what we already have.

**5.3 Queue size scaling.** Does descriptor failure worsen as the queue grows past 20? Needs a
case too large to enumerate and a sampling-based reference instead of exact truth.

**5.4 Load snapshots on the IEEE cases.** TX averages 8 ERCOT snapshots. Whether the IEEE cases
need more than a single peak snapshot to produce diverse congestion is untested.

**5.5 Distribution robustness.** The MRF-coupled distributions are already swept but never
analyzed for this question. Judge: does any conclusion change, or is it a robustness paragraph.

## 6. Open questions to settle before writing

- **Venue and length.** IEEE PES Letters (4 pages) vs PSCC / PMAPS / HICSS (6-8). PMAPS fits the
  probabilistic framing. Decide — it fixes the figure budget.
- **Is `c` fixed per line or per case?** Current plots use both a global threshold (`thr100`) and
  a per-line auto-scaled level (`--at 1e-2`). Pick one convention; the per-line quantile level is
  more comparable across cases.
- **Headline metric.** "Per-line probability" is a technique. The planner-facing version is
  "which lines are most likely to be severely congested, and how many solves does it take to rank
  them correctly." Top-k rank recovery may be the better headline, and it is the fairest arena for
  the descriptor comparison.

## 7. Task list

Ordered by dependency. `[ ]` = not started.

### A. New test cases (blocking for everything downstream)

- [x] **A1.** Generic N-1 screening — `src/cases/screen_contingencies.py`. Keeps 7 of 19
      (case14), 22 of 38 (case30), 21 of 177 (case118).
- [x] **A2.** Case builder — `src/cases/build_ieee_queue.py`, one file, three cases.
- [x] **A3.** Calibration — `src/cases/calibrate_ratings.py` drives each case to an N-1
      secure incumbent grid; the recipe constants live in `cases/registry.py::Recipe`. Resulting all-off
      `mu` is $140 / $63 / $153 per case, well under the $3500 cap, with the range above that left
      for buildout to drive.
- [~] **A4.** Testbed criteria. Checked on case14 (exhaustive, full contingency set): 21
      constraints bind, per-line `P[mu_l > 0]` spans 1.0 down to ~0, `P[mu_l > 100]` is nonzero on
      13 of them and spans 0.41 down to 0, and `max_mu` is graded — $316 at `P = 0.1`, $514 at
      `1e-2`, $730 at `1e-3`, $738 max. Many modes, no single dominator. Sampled checks on
      case30 (exhaustive): all 25 constraints in its `dual_cache` have nonzero `P[mu_l > 100]`,
      per-line `P[mu_l > 0]` spans 0.002 to 0.50, `max_mu` graded $144 / $295 / $450 at
      `P = 0.1 / 1e-2 / 1e-3`. Sampled check on case118 (26 of 31 constraints vary with the
      build). Redo on the finished case118 sweep, and run `src/diagnostics/characterize_modes.py`. Note
      case118's *scalar* `max_mu` is nearly degenerate even though its per-line structure is not
      — pick its `one_mu` element from the per-line tail, not the default most-congested rule.
- [x] **A5.** Single load level for the IEEE cases, not a snapshot collection: the queue itself
      supplies the mode diversity (see A4), and it keeps the physical model one variable simpler
      than TX. Revisit only if 5.4 gives a reason.

### B. Ground truth

- [x] **B1.** SCDCOPF solve path — `src/model/scdcopf.py` + `src/model/evaluator.py`. Config-parallel warm-started
      workers (the IEEE cases have one load level, so parallelism is over configurations rather
      than TX's snapshot-pinned workers), reusing `src/model/soft_limits.py` and the ORDC relief.
- [x] **B2.** Exhaustive enumeration, against the **full** contingency set — screening is a union
      over stress configurations and is not exact (one case14 config in 62 binds a screened-out
      contingency and comes out 2% low), while the full set costs only ~12% more on case30 and
      2.3x on case118. All three done: case14 (`2^14`, 21 binding constraints), case30 (`2^20` in
      12 minutes on 18 workers — warm persistent workers make these sweeps cheap, not the days
      first estimated from cold solves), case118 (`2^20`, finished 2026-07-29). Parts are
      resumable. Note the sweeps are not bit-reproducible: degenerate duals plus lazy PTDF mean a
      re-run changes ~0.2% of per-configuration `max_mu` values, so a committed `mu_cache` is the
      ground truth for its case rather than something to regenerate (see `src/README.md`).
- [x] **B3.** Registered in `cases/registry.py` -- one record per case now carries the queue file, the
      long/wide ingest, the cache paths, and the `one_mu` element, replacing the five separate
      case tables that used to be kept in sync by hand. `target_cache` / `dual_cache` built for
      all three; case118's `one_mu` is still unset — choose it from the enumerated per-line tail,
      not the default most-congested rule (see A4).

### C. Sampling results (Section IV)

- [ ] **C1.** Run `sampling.run_experiments` on the three new cases (direct sampling is what
      the paper needs; the other samplers come free from the same run).
- [ ] **C2.** Error-vs-budget figure: per-line error and top-k rank recovery vs solve count, all
      five cases. New plotting file.

### D. Descriptor results (Section V)

- [ ] **D1.** Extend `src/descriptors/study.py` to the three new cases — needs a descriptor
      source for security-constrained IEEE cases, now done: `descriptors/compute.py` dispatches on the case family, so a security-constrained
      IEEE case takes the same `base:`/`ctg:` path Texas does.
- [ ] **D2.** Re-run the descriptor figures with the queue-empty grid as the base configuration
      (the TX convention), now that every case has an incumbent fleet.
- [ ] **D3.** Restate the instability result over the five-case matrix as a table of Spearman by
      (case, descriptor).
- [ ] **D4.** Add the extra descriptors from 5.2 so the negative result is not under-powered.

### E. Writing

- [ ] **E1.** Choose venue and page limit.
- [ ] **E2.** Settle the threshold convention and headline metric (Section 6).
- [ ] **E3.** Related work: Misra/Roald/Ng active-set learning
      (`notes/misra_active_constraints_summary.md`) is the closest neighbor — their DiscoverMass is
      essentially our direct sampler. Also interconnection-queue planning under uncertainty and
      congestion-screening practice.
- [ ] **E4.** Draft in outline order; every figure from a committed script.

### F. Optional

- [ ] **F1.** ERCOT empirical motivation: congestion is concentrated, contingency dominated, capped
      at $3500 — supports the modeling recipe. One paragraph, possibly one panel from
      `results/congestion_concentration*.png`.

## 8. Immediate next step

Finish B2 (the case30 and case118 sweeps), then B3 for those two, then C1 — run
`sampling.run_experiments` across the five-case matrix. D1 is the first thing after that
which needs new code: the descriptor study has no path for security-constrained IEEE cases,
whose constraints are `base:` / `ctg:` pairs like TX's rather than plain lines.
