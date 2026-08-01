"""Exact-explored-mass importance sampling for the tail of S(x) = max_b |mu_b|.

Ascent-seeded importance sampling (ascent_importance_sampling) spends ~40% of its
budget evaluating S at high-S configurations during the ascent, then throws those
evaluations away --- they only seed the mixture centers.  But every configuration we
ever evaluate has a *known, exact* nominal probability (an exact lookup into the
enumerated nominal weights, whatever the law's form), so its contribution to
P[S >= b] needs no estimation at all.

This approach keeps the same ascent and the same defensive mixture proposal, but
splits the exceedance probability into an exact part over the explored set E and an
importance-sampling estimate of the unexplored remainder:

    P[S >= b] = sum_{x in E} p_nom(x) 1{S(x) >= b}            (exact)
              + sum_{x not in E} p_nom(x) 1{S(x) >= b}        (IS-estimated).

E is fixed before drawing the fresh importance samples, and those samples have their
contribution zeroed on E (already counted exactly), so the estimator stays unbiased
while the heavy, high-S explored configs --- the dominant variance contributors ---
are counted at their true weight instead of being estimated.

Usage (vs the other samplers):
    uv run python -m tail_sampling.run_experiments
"""

import numpy as np

import tail_sampling.ascent_importance_sampling as ais


def sample(problem, ev, rng, budget, move="all", score_lookup=None):
    """Exact-explored-mass + IS-remainder weighted sample (ids, weights).

    move / score_lookup pick the ascent's per-step neighbor evaluations (all / best /
    top5).  Explored configs carry their exact nominal probability; fresh draws outside
    the explored set carry IS weights and are zeroed on it (already counted exactly), so
    sum_i w_i g(x_i) ~ E_p[g] for any g."""
    mode_ids = ais.discover_modes(problem, ev, rng, int(ais.ASCENT_FRAC * budget),
                                  move, score_lookup)
    explored = np.array(sorted(ev.seen))
    p_explored = np.exp(problem.log_nominal_pmf(problem.to_binary(explored)))

    R, omega = ais.build_mixture(problem, mode_ids)
    n_samp = max(1, budget - ev.calls)
    bits = ais.sample_mixture(problem, R, omega, n_samp, rng)
    ids = problem.bits_to_id(bits)
    ev(ids)
    w = np.exp(problem.log_nominal_pmf(bits) - ais.mixture_logpmf(problem, bits, R, omega))
    fresh = ~np.isin(ids, explored)                      # exclude already-counted configs
    return (np.concatenate([explored, ids]),
            np.concatenate([p_explored, w * fresh / n_samp]))
