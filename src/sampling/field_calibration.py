"""Calibrate an MRF's field h to target marginals p under a fixed coupling J, by MCMC.

For a dense coupling (MRF-A puts an independent coefficient on every pair) the marginals
have no scalar summary to collapse onto, so they must be estimated by Gibbs sampling and
the field fit by stochastic moment matching -- there is no exact route as there is for the
uniform-within-block structured laws.

The model is p(x) proportional to exp(h . x + 1/2 x^T J x) with J symmetric, zero diagonal.
Its log-likelihood in h is concave with gradient p - m(h), where m(h) = E_model[x], so the
field that reproduces the marginals p is the unique root of m(h) = p.  This module finds it
by persistent-chain stochastic gradient ascent on that true gradient:

  - start h at the mean-field value, so the coupling is compensated to first order and the
    chains begin near the target rather than at the uncoupled field,
  - keep one ensemble of Gibbs chains alive across iterations (persistent contrastive
    divergence), advancing a few sweeps each step and averaging them for a lower-variance m,
  - step h += step_t (p - m) on the probability-space gradient, not a logit-space proxy,
  - return the Polyak-Ruppert average of h over the tail iterations, which converges at the
    Monte Carlo rate and removes the single-iterate noise a last-iterate return would keep.
"""

import numpy as np

from src.sampling.mrf import gibbs_sweep


def mean_field_field(J, p):
    """Mean-field starting field h_i = logit(p_i) - sum_j J_ij p_j.

    Solves the mean-field self-consistency m_i = sigmoid(h_i + (J m)_i) at m = p.
    """
    return np.log(p / (1.0 - p)) - J @ p


def averaged_marginals(h, J, X, rng, sweeps):
    """Advance the persistent ensemble X by `sweeps` Gibbs sweeps; return the mean marginals."""
    acc = np.zeros(len(h))
    for _ in range(sweeps):
        gibbs_sweep(h, J, X, rng)
        acc += X.mean(axis=0)
    return acc / sweeps


def calibrate_field_mcmc(J, p, chains=8000, warmup=300, iters=300, sweeps=2,
                         step0=0.5, tau=100.0, avg_frac=0.5, seed=0):
    """Field h whose model marginals under coupling J match p, by stochastic moment matching.

    Args:
        J: symmetric coupling, zero diagonal (n, n).
        p: target marginals (n,), in (0, 1).
        chains: persistent Gibbs chains averaged for each marginal estimate.
        warmup: burn-in sweeps at the mean-field field before the first step.
        iters: gradient steps.
        sweeps: Gibbs sweeps averaged per step (variance reduction on the gradient).
        step0, tau: Robbins-Monro step size step0 / (1 + t / tau).
        avg_frac: fraction of the final iterations Polyak-averaged into the returned field.
        seed: RNG seed.

    Returns (h, m): the averaged field and a low-variance marginal estimate at it.
    """
    rng = np.random.default_rng(seed)
    h = mean_field_field(J, p)
    X = (rng.random((chains, len(p))) < p).astype(np.int8)
    for _ in range(warmup):
        gibbs_sweep(h, J, X, rng)

    start_avg = int((1.0 - avg_frac) * iters)
    h_sum, n_avg = np.zeros(len(p)), 0
    for t in range(iters):
        m = averaged_marginals(h, J, X, rng, sweeps)
        h = h + (step0 / (1.0 + t / tau)) * (p - m)
        if t >= start_avg:
            h_sum += h
            n_avg += 1

    h_bar = h_sum / max(n_avg, 1)
    return h_bar, averaged_marginals(h_bar, J, X, rng, sweeps=20)
