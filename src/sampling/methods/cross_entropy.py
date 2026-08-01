"""Cross-entropy importance sampling for the tail of S(x) = max_b |mu_b|.

The cross-entropy method (Rubinstein) fits a tractable proposal to the rare set by a
sequence of elite-refit rounds: draw from the current Bernoulli proposal, keep the
top-rho by S, and reset each generator's on-probability to its elite frequency, so the
proposal marches toward the high-S region.  The proposal is a *product* Bernoulli, so
this sampler applies only when the nominal law is itself Bernoulli (J = 0); it is the
one method that exploits that product structure directly.

The final tail estimate reuses the same defensive mixture as ascent_importance_sampling
-- beta * nominal + (1 - beta) * Bern(r_CE) -- so the importance weights p_nom / q are
bounded by 1/beta and unbiased, and one reweighted sample gives the whole exceedance
curve.  Budget: a fraction ce_frac is spent on the fitting rounds, the rest on the final
weighted draw.

Usage (vs the other samplers): the runner passes it in for Bernoulli cases only.
"""

import numpy as np

import sampling.methods.ascent_importance_sampling as ais


def ce_round(problem, ev, rng, r, n, rho, floor):
    """One elite-refit round: draw n from Bern(r), refit r to the top-rho by S."""
    bits = (rng.random((n, problem.ngen)) < r).astype(np.int8)
    S = ev(problem.bits_to_id(bits))
    elite = np.argsort(-S)[: max(1, int(rho * n))]
    return np.clip(bits[elite].mean(axis=0), floor, 1.0 - floor)


def sample(problem, ev, rng, budget, n_rounds=5, rho=0.1, ce_frac=0.5, beta=0.2, floor=0.02):
    """Cross-entropy weighted sample (ids, weights) at the call budget (Bernoulli law only).

    Weights satisfy sum_i w_i g(x_i) ~ E_p[g] for any g (so sum_i w_i ~ 1)."""
    assert not np.any(problem.dist.J), "cross-entropy requires a Bernoulli (J=0) nominal law"
    r = 1.0 / (1.0 + np.exp(-problem.dist.h))            # nominal per-generator marginals
    fit_budget = int(ce_frac * budget)
    n_per = max(problem.ngen + 1, fit_budget // n_rounds)
    for _ in range(n_rounds):
        if ev.calls >= fit_budget:
            break
        r = ce_round(problem, ev, rng, r, n_per, rho, floor)

    R, omega = r[None, :], np.array([1.0])               # single CE mode in a defensive mixture
    n_samp = max(1, budget - ev.calls)
    bits = ais.sample_mixture(problem, R, omega, n_samp, rng, beta)
    ids = problem.bits_to_id(bits)
    ev(ids)
    log_w = problem.log_nominal_pmf(bits) - ais.mixture_logpmf(problem, bits, R, omega, beta)
    return ids, np.exp(log_w) / n_samp
