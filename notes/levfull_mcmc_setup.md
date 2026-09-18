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
- **Calibration/sampling split** (user approved):
  - Calibrate the field h to the target marginals **exactly** for the structured laws (`cobuild`, `substitute`), by MCMC for `mrf_b`.
  - **Sample every law by Gibbs MCMC on the cluster.** The Gibbs sweeps are ~1% of a round next to the OPF solves, so MCMC adds negligible overhead vs iid sampling, and Gibbs is the only option for the dense `mrf_b`.

Why exact calibration is possible for the structured laws: their within-block coupling is **uniform** (`cobuild`: +c' on every pair; `substitute`: anchor -c', rest +c'), so the interaction energy depends only on the count K of a fuel's on-candidates — `(c'/2)(K^2 - K)` for cobuild. The sum over 2^(block size) configurations collapses to a sum over K = 0..block_size, computed exactly with elementary symmetric polynomials. Block-diagonality alone does **not** give this — it only makes the blocks independent; the collapse needs the uniform coupling. `mrf_b` is dense with no such summary, so it genuinely needs MCMC.

## Open decisions (resume here)

1. **`mrf_b` coupling scale on 195 nodes.** `COUPLINGS["mrf_b"] = (0, 2.0)` is calibrated to the enumerated 20-candidate cases. On 195 nodes a dense N(0, 2^2) coupling gives each node an incoming field of std ~ sigma*sqrt(n/2) ~ 2*sqrt(97) ~ 20, dwarfing h (order 0-2): a strongly frustrated spin glass. That makes calibration ill-conditioned and Gibbs mixing slow (glassy) — for both the calibration and the cluster sampling. Needs a global scale (dense analogue of `block_c`); the spin-glass choice sigma ∝ 1/sqrt(n) matching the n=20 regime gives **sigma ≈ 2*sqrt(20/195) ≈ 0.64**. Decide the target before building `mrf_b`.
2. **Calibration algorithm for `mrf_b` (first-order vs Newton).** `src/sampling/field_calibration.py` implements first-order stochastic gradient. Alternative: damped Newton `h += (Cov + lambda I)^-1 (p - m)` using the sample covariance (= exact Jacobian dm/dh) — ~10 iterations vs a few hundred, at the cost of estimating/regularizing a 195x195 covariance each step. User suspected the approach may be inefficient and asked to discuss before running; open question is whether they have a specific method in mind.

## What is built and verified

- **`src/cases/registry.py`**: `activsg2000_v12_levfull` entry (ngen 195, `sample_only=True`); `_leveraged(..., sample_only=...)` factory.
- **`src/sampling/mcmc_sweep.py`**: the resumable sweep. Ensemble Gibbs (`--chains` chains advanced `--thin` sweeps between solves), solved through the same worker pool + screened-contingency augmentation the enumeration uses (`enumerate_configs.solve_chunk`). Config identity is a **packed bit vector** (int64 id overflows past ngen 63). Checkpoints `{X chain states, RNG state, next round}` to `state.json` after each round; per-round part files rewritten idempotently. Output `results/<case>/mcmc/<law>/{state.json, duals/, snapmax/, configs/}`. Chain/checkpoint/resume unit-tested (offline, fake solves — test lived in scratchpad, not committed).
- **`src/sweeps/submit_cluster.py`**: `sample_only` path wired and dry-run-verified. Submits `prep -> mcmc(law) x LINKS` per law, laws chained sequentially (they share the one screened contingency set that augmentation rewrites). Settings: `PREP_ONLY=False`, `SAMPLE_LAWS`, `MCMC_MAX_HOURS=29.0`, `LINKS=2`, `SWEEP_TIME=30h`. `vacc.slurm` forwards `--law/--max-hours/--screened` via `"$@"`. **TODO: add `mrf_b` to `SAMPLE_LAWS`.**
- **Law storage**: `law_store.law_file` routes `sample_only` laws to **tracked** `data/<case>_law_<name>.json` (not git-ignored `results/`), because the cluster reads the law to draw configs — it is a sweep input, like the relief set. Enumerated cases keep laws under `results/<case>/laws/`.
- **`src/sampling/field_calibration.py`** (new, **not wired, not run**): MCMC field calibration — mean-field warm start (`h = logit(p) - J p`), true-gradient stochastic ascent `h += step*(p - m)` on persistent chains, Polyak-Ruppert averaging of the tail iterates. Replaces the biased `mrf.calibrate_field_mcmc` (which stepped in logit space, used one noisy sweep per step, and returned the last iterate — the source of the 2.1% marginal error on cobuild).

## Still to do

1. **Write the exact structured calibration** (count-collapse via elementary symmetric polynomials): per block, build e_k of a_i = exp(h_i), form P(K=k) ∝ e_k f(K), get marginals by leave-one-out, calibrate h by Newton with the exact block covariance. Handle `cobuild` (tilt in K) and `substitute` (condition on the anchor, tilt in the rest count). Exact, O(block^2), machine precision.
2. **Settle open decisions 1-2**, then finish `field_calibration.py` (add Newton if chosen).
3. **Rework `build_laws_mcmc.py`**: use exact calibration for `cobuild`/`substitute`, `field_calibration.calibrate_field_mcmc` for `mrf_b`; add `mrf_b` (with the chosen scale) to its law list. Rename off "mcmc" since most laws are no longer MCMC-calibrated.
4. **Rebuild the stored laws.** Current `data/activsg2000_v12_levfull_law_*.json` are **provisional**: `bernoulli` is exact (1.3e-3), but `cobuild` (2.1e-2) and `substitute` were built with the biased MCMC calibration and must be rebuilt with the exact method; `mrf_b` not built yet. Verify marginals to machine precision and within-block correlations after.
5. **Add `mrf_b` to `submit_cluster.SAMPLE_LAWS`.**
6. **Sync check before launch**: the tracked `data/*_law_*.json` and the queue file must reach the cluster; laws must be rebuilt (step 4) first.

## Related

- q28 testbed (`activsg2000_v12_q28_lev0a3`, 2^28) is a separate, deferred prep-only case — swap `CASES` and set `PREP_ONLY=True` to launch its prep.
- `notes/ercot_price_formation.md` (recipe), `src/sampling/distributions.py` (`block_c`, couplings), `src/sampling/mrf.py` (Gibbs).
