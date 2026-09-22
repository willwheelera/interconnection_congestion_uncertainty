# Full-queue MCMC sampling setup (`activsg2000_v12_levfull`)

Status note for resuming the setup of the large sampled testbed.
The goal is to estimate the congestion-dual (mu) distribution and P[large mu] over generator-availability
configurations on a large queue with **no exhaustive reference**: 2^195 cannot be enumerated, so the
distribution is estimated by drawing configurations from a nominal law and solving each one.

## The case

- `activsg2000_v12_levfull`: the v12 network/fleet/snapshots with the **full leveraged pool** on top.
- Queue: `data/activsg2000_queue_levfull.json` — 195 candidates, blocks Wind 25 / Solar 51 / Battery 95 / Gas 24, 44 buses.
- Registered `sample_only=True` (guards the `1<<ngen` enumeration paths); ngen = 195.
- Built by `build_leveraged_queue.py --full` (keep every candidate clearing the shift-factor screen, no per-fuel quota, no aggregation).

## Decisions already made

- **Queue**: full leveraged pool (~195), no aggregation, leverage screen kept. (user)
- **Structured-law coupling**: scaled by block size (`distributions.block_c`, `ref_block=5`, `coupling=2.0`) so a large fuel block does not saturate. (user)
- **Laws to sample**: `bernoulli`, `mrf_cobuild`, `mrf_substitute`, **and `mrf_b`** (dense, any-sign pair coupling — added by user for comparison).
- **Calibration/sampling split**:
  - Calibrate the field h to the target marginals **exactly** for the structured laws (`cobuild`, `substitute`), by MCMC for `mrf_b`.
  - **Sampling (revised 2026-09-22).** The structured laws are drawn as **exact iid** samples by the same count collapse (`structured_sampling.sample_structured`), *not* by Gibbs. At the coupling `block_c` leaves ((m-1)cp = 8), a fuel block has a metastable all-on mode a systematic-scan Gibbs sweep cannot cross: on the m=95 cobuild block Gibbs from all-off is trapped at mean count 12.5 vs the exact 14.85 (marginal error 2.7e-2, no improvement from 200 to 4000 sweeps), and from all-on it freezes at 90.8. So Gibbs does not reproduce the calibrated marginals and would miss the correlated high-count tail these laws exist to create. `bernoulli` (J = 0, a Gibbs sweep is already iid) and the dense `mrf_b` (no count summary) still draw by Gibbs.

Why exact calibration is possible for the structured laws: their within-block coupling is **uniform** (`cobuild`: +c' on every pair; `substitute`: anchor -c', rest +c'), so the interaction energy depends only on the count K of a fuel's on-candidates — `(c'/2)(K^2 - K)` for cobuild. The sum over 2^(block size) configurations collapses to a sum over K = 0..block_size, computed exactly with elementary symmetric polynomials. Block-diagonality alone does **not** give this — it only makes the blocks independent; the collapse needs the uniform coupling. `mrf_b` is dense with no such summary, so it genuinely needs MCMC.

## Open decisions (resume here)

1. **`mrf_b` coupling scale on 195 nodes.** `COUPLINGS["mrf_b"] = (0, 2.0)` is calibrated to the enumerated 20-candidate cases. On 195 nodes a dense N(0, 2^2) coupling gives each node an incoming field of std ~ sigma*sqrt(n/2) ~ 2*sqrt(97) ~ 20, dwarfing h (order 0-2): a strongly frustrated spin glass. That makes calibration ill-conditioned and Gibbs mixing slow (glassy) — for both the calibration and the cluster sampling. Needs a global scale (dense analogue of `block_c`); the spin-glass choice sigma ∝ 1/sqrt(n) matching the n=20 regime gives **sigma ≈ 2*sqrt(20/195) ≈ 0.64**. Decide the target before building `mrf_b`.
2. **Calibration algorithm for `mrf_b` (first-order vs Newton).** `src/sampling/field_calibration.py` implements first-order stochastic gradient. Alternative: damped Newton `h += (Cov + lambda I)^-1 (p - m)` using the sample covariance (= exact Jacobian dm/dh) — ~10 iterations vs a few hundred, at the cost of estimating/regularizing a 195x195 covariance each step. User suspected the approach may be inefficient and asked to discuss before running; open question is whether they have a specific method in mind.

## What is built and verified

- **`src/cases/registry.py`**: `activsg2000_v12_levfull` entry (ngen 195, `sample_only=True`); `_leveraged(..., sample_only=...)` factory.
- **`src/sampling/mcmc_sweep.py`**: the resumable sweep. Ensemble Gibbs (`--chains` chains advanced `--thin` sweeps between solves), solved through the same worker pool + screened-contingency augmentation the enumeration uses (`enumerate_configs.solve_chunk`). Config identity is a **packed bit vector** (int64 id overflows past ngen 63). Checkpoints `{X chain states, RNG state, next round}` to `state.json` after each round; per-round part files rewritten idempotently. Output `results/<case>/mcmc/<law>/{state.json, duals/, snapmax/, configs/}`. Chain/checkpoint/resume unit-tested (offline, fake solves — test lived in scratchpad, not committed).
- **`src/sweeps/submit_cluster.py`**: `sample_only` path wired and dry-run-verified. Submits `prep -> mcmc(law) x LINKS` per law, laws chained sequentially (they share the one screened contingency set that augmentation rewrites). Settings: `PREP_ONLY=False`, `SAMPLE_LAWS`, `MCMC_MAX_HOURS=29.0`, `LINKS=2`, `SWEEP_TIME=30h`. `vacc.slurm` forwards `--law/--max-hours/--screened` via `"$@"`. **TODO: add `mrf_b` to `SAMPLE_LAWS`.**
- **Law storage**: `law_store.law_file` routes `sample_only` laws to **tracked** `data/<case>_law_<name>.json` (not git-ignored `results/`), because the cluster reads the law to draw configs — it is a sweep input, like the relief set. Enumerated cases keep laws under `results/<case>/laws/`.
- **`src/sampling/structured_calibration.py`** (new, **done + verified 2026-09-18, hardened 2026-09-22**): the exact, enumeration-free calibration for `cobuild`/`substitute`. The uniform within-block coupling makes a block's law depend only on the on-count K, so the 2^m block sum collapses to the Poisson-binomial pmf tilted by the count energy, giving exact per-node marginals in O(m^2). The field is solved block by block by Newton (finite-difference Jacobian of the exact marginals) with **coupling continuation** — ramp `cp` from 0 (where `h = logit(p)` is exact) to the target in ~0.5 per-node increments, warm-starting each solve. Continuation is required: `block_c` leaves a near-bistable ferromagnet at `(m-1)cp = 8`, where Newton started cold diverges at `m = 95`. The per-step solve is an **adaptive Levenberg** step (raise damping until the 2-norm residual drops, lower it on success): the real levfull solar block has 31 candidates at the identical target 0.05, which makes the block Jacobian near-singular (a permutation null space) — the old fixed 1e-12 regularization took a runaway step there and drove h to -1e10 (substitute m=51). All four blocks of both laws now calibrate to <= 5e-14. `structured_marginals` returns the analytic exact marginals of a calibrated field (the calibration's own check). Wired into `build_laws_mcmc.build_law`.
- **`src/sampling/structured_sampling.py`** (new, **done + verified 2026-09-22**): the exact iid sampler that the same count collapse permits. Per block: draw the count K from its exact pmf (Poisson-binomial tilted by the count energy), then the size-K on-set as a conditional Bernoulli (a size-K subset with probability proportional to the product of the on odds, sampled sequentially from the suffix Poisson-binomial pmfs); `substitute` draws the anchor bit first, then the rest as a cobuild block with fields shifted by -cp when the anchor is on. Block kind, sign, and anchor are read straight off the stored coupling J, so it draws exactly the stored law. Verified against enumeration (marginals and second moments to the Monte Carlo floor at m <= 8) and at full scale (target marginals reproduced to the 20k-sample floor ~7e-3, within-block |corr| mean ~0.33 max ~0.57). Wired into `build_laws_mcmc.build_law` (the correlation check) and `mcmc_sweep.run` (the cluster draw for the structured laws).
- **`src/sampling/field_calibration.py`** (the corrected MCMC calibrator; **kept for `mrf_b` only, not wired into any current law**): mean-field warm start (`h = logit(p) - J p`), true-gradient stochastic ascent `h += step*(p - m)` on persistent chains, Polyak-Ruppert averaging of the tail iterates. Replaces the now-**deleted** biased `mrf.calibrate_field_mcmc` (which stepped in logit space, used one noisy sweep per step, and returned the last iterate — the source of the 2.1% marginal error on cobuild). Verified unbiased: ~6e-4 exact-marginal error on a dense n=12 case, versus 2.1e-2 from the old one.

## Still to do

1. ~~Write the exact structured calibration~~ **DONE** — `src/sampling/structured_calibration.py` (count collapse via the Poisson-binomial tilt, per-block Newton with coupling continuation); wired into `build_laws_mcmc`.
2. ~~Rebuild the stored laws~~ **DONE (2026-09-22)** — `data/activsg2000_v12_levfull_law_{bernoulli,cobuild,substitute}.json` regenerated at machine precision (marginal error 0 / 2e-15 / 5e-14; within-block |corr| mean 0.33/0.35). The stored `p_exact` is now the analytic count-collapse marginal, not a Gibbs estimate. The laws are drawn by the exact sampler (item revised under Decisions), so they are faithfully sampleable — the earlier Gibbs 2.8e-2 was the sampler's metastability, not calibration.
3. **Settle open decision 1** (mrf_b coupling scale), decide open decision 2 (mrf_b first-order vs Newton), then add `mrf_b` to `build_laws_mcmc.LAWS` and build it with `field_calibration.calibrate_field_mcmc`. `mrf_b` is the only law left on Gibbs, so `build_laws_mcmc.py` need not be renamed.
4. **Add `mrf_b` to `submit_cluster.SAMPLE_LAWS`.**
6. **Sync check before launch**: the tracked `data/*_law_*.json` and the queue file must reach the cluster; laws must be rebuilt first.

## Related

- q28 testbed (`activsg2000_v12_q28_lev0a3`, 2^28) is a separate, deferred prep-only case — swap `CASES` and set `PREP_ONLY=True` to launch its prep.
- `notes/ercot_price_formation.md` (recipe), `src/sampling/distributions.py` (`block_c`, couplings), `src/sampling/mrf.py` (Gibbs).
