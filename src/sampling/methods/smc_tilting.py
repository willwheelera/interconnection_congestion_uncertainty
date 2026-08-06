"""Stochastic-kernel (SMC) importance sampling for the tail of S(x) = max_b |mu_b|.

This replaces the greedy ascent of ascent_importance_sampling with a *stochastic*
Markov kernel, so every visited configuration is a proper sample from a known law
and can be used directly --- no Bernoulli ansatz for the importance function is
ever fit.  We anneal a population of particles along the exponential-tilting path

    q_theta(x) proportional to p_nom(x) * exp(theta * S(x)),     theta: 0 -> large,

with a sequential Monte Carlo sampler: reweight by the incremental tilt, resample,
then move with a single-bit-flip Metropolis kernel that leaves q_theta invariant.
The mean incremental weight at each step estimates the normalizing-constant ratio
Z_theta / Z_0, so each temperature's equally-weighted population is a usable sample
from q_theta with a known weight.

Because every temperature gives an unbiased estimate of the *same* nominal
exceedance P[S >= b] (each q_theta covers the whole space through its nominal
factor), we pool all temperatures' particles with the balance heuristic (multiple
importance sampling).  The nominal factor cancels, leaving an estimate that depends
only on each particle's S, the temperature ladder, and the estimated Z-ratios, and
that covers the full curve from one population.

Usage (vs the other samplers):
    uv run python -m src.sampling.run_experiments
"""

import numpy as np


def _logsumexp(a, axis=None):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def next_dtheta(S, target_ess):
    """Bisect the tilt increment so the incremental-weight ESS hits target_ess * N."""
    s = S - S.max()

    def ess_frac(dt):
        w = np.exp(dt * s)
        return w.sum() ** 2 / (len(S) * (w * w).sum())

    hi = 1.0
    while ess_frac(hi) > target_ess and hi < 1e6:
        hi *= 2.0
    lo = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ess_frac(mid) > target_ess:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def metropolis_tilt(problem, bits, S, theta, ev, rng):
    """One single-bit-flip Metropolis sweep leaving q_theta ~ p_nom * exp(theta S)."""
    n = bits.shape[0]
    rows = np.arange(n)
    j = rng.integers(problem.ngen, size=n)
    cand = bits.copy()
    cand[rows, j] ^= 1
    S_cand = ev(problem.bits_to_id(cand))

    log_nom_ratio = problem.log_nominal_pmf(cand) - problem.log_nominal_pmf(bits)
    log_alpha = log_nom_ratio + theta * (S_cand - S)    # single-flip proposal is symmetric
    acc = np.log(rng.random(n)) < log_alpha
    bits[acc] = cand[acc]
    S = np.where(acc, S_cand, S)
    return bits, S


def smc_sample(problem, ev, rng, budget, n_particles=64, n_move=3, target_ess=0.5):
    """Anneal a particle population along q_theta; return per-temperature populations.

    Returns (S_pops, id_pops, thetas, log_rs): for each temperature level the
    population's S values, its config ids, its tilt theta, and the log
    normalizing-constant ratio log(Z_theta/Z_0).  Level 0 is the nominal population
    (theta = 0, log ratio 0).
    """
    bits = problem.draw_nominal(n_particles, rng)
    ids = problem.bits_to_id(bits)
    S = ev(ids)
    S_pops, id_pops, thetas, log_rs = [S.copy()], [ids.copy()], [0.0], [0.0]
    theta, log_r = 0.0, 0.0

    while ev.calls + n_move * n_particles <= budget:
        dtheta = next_dtheta(S, target_ess)
        log_r += _logsumexp(dtheta * S) - np.log(n_particles)
        w = np.exp(dtheta * (S - S.max()))
        idx = rng.choice(n_particles, n_particles, p=w / w.sum())
        bits, S = bits[idx], S[idx]
        theta += dtheta
        for _ in range(n_move):
            bits, S = metropolis_tilt(problem, bits, S, theta, ev, rng)
        S_pops.append(S.copy())
        id_pops.append(problem.bits_to_id(bits))
        thetas.append(theta)
        log_rs.append(log_r)
    return S_pops, id_pops, np.array(thetas), np.array(log_rs)


def smc_weights(S_pops, thetas, log_rs):
    """Balance-heuristic MIS weight of every pooled particle.

    Each particle's MIS denominator is sum_k N * q_k(x) / p_nom(x) = N * sum_k
    exp(theta_k S - log_r_k); the nominal factor cancels, so the weight 1/denominator
    makes the pooled population a nominal-weighted sample (sum over S >= b is P[S>=b]).
    """
    allS = np.concatenate(S_pops)
    n = len(S_pops[0])
    terms = thetas[None, :] * allS[:, None] - log_rs[None, :]
    log_denom = np.log(n) + _logsumexp(terms, axis=1)
    return np.exp(-log_denom)


def sample(problem, ev, rng, budget, n_particles=64, n_move=3, target_ess=0.5):
    """SMC-tilting weighted sample (ids, weights) at the given solver-call budget."""
    S_pops, id_pops, thetas, log_rs = smc_sample(problem, ev, rng, budget,
                                                 n_particles, n_move, target_ess)
    return np.concatenate(id_pops), smc_weights(S_pops, thetas, log_rs)
