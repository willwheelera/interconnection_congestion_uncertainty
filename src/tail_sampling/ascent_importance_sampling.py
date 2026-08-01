"""Ascent-seeded defensive importance sampling for the tail of S(x) = max_b |mu_b|.

The dominant-point idea (FORM / large deviations): the rare set {S >= b} is best
sampled from configurations that already sit in it but carry appreciable nominal
weight.  We discover such configurations by greedy single-flip ascent on S --- the
discrete analog of following dmu/dP uphill.  Started from typical (nominal) draws,
each ascent trajectory climbs through progressively rarer, higher-S configs, so the
set of *visited* configurations tiles the tail across a range of depths.

Those visited configs seed an importance distribution: a mixture with one Bernoulli
component tightly centered on each visited config plus a defensive component that is
the nominal law itself, of weight beta.  The Bernoulli mode components are only the
*proposal* ansatz; the estimator queries the nominal law generically through
problem.log_nominal_pmf, so it makes no assumption about the law's form.  The
defensive component bounds the likelihood ratio by 1/beta, so the importance-sampling
estimator of P[S >= b] is unbiased with finite variance, and one reweighted sample set
yields the whole exceedance curve at once.

Hyperparameters are keyword args on `sample`, set by the runner (run_experiments):
    eps         per-component flip rate of a mode component (mode tightness)
    beta        weight of the defensive nominal component (bounds weights by 1/beta)
    ascent_frac fraction of the budget spent discovering modes by ascent
    gamma       tempering of mode weights, omega ~ nominal(mode)**gamma
                (gamma<1 keeps the rarest, deepest modes sampled for tail coverage)

Usage (standalone, vs the other samplers):
    uv run python -m tail_sampling.run_experiments
"""

import numpy as np

from utils.availability import get_all_neighbors

ASCENT_FRAC = 0.4   # default fraction of budget spent discovering modes by ascent


def step_candidates(problem, cur, move, score_lookup):
    """Which neighbor ids to evaluate this step under the move strategy.

    all  -> every single-flip neighbor (no proxy); best/top5 -> the 1/5 neighbors whose
    flip has the highest barrier-sensitivity proxy s_k(cur), so only they are evaluated.
    """
    nb = get_all_neighbors(np.array([cur]), problem.ngen)[0]
    if move == "all":
        return nb
    order = np.argsort(-score_lookup(cur))               # rank flips by proxy
    return nb[order[: 1 if move == "best" else 5]]


def ascent_from(problem, start_id, ev, move="all", score_lookup=None):
    """Greedy single-flip hill-climb on S; return all distinct ids visited en route.

    move selects which neighbors are S-evaluated per step (see step_candidates); the
    proxy read is a byproduct of cur's solve, so only the S-evaluations spend budget.
    """
    cur = int(start_id)
    visited = [cur]
    cur_S = ev(np.array([cur]))[0]
    for _ in range(problem.ngen):
        cand = step_candidates(problem, cur, move, score_lookup)
        cand_S = ev(cand)
        j = int(np.argmax(cand_S))
        if cand_S[j] <= cur_S:
            break
        cur, cur_S = int(cand[j]), cand_S[j]
        visited.append(cur)
    return visited


def discover_modes(problem, ev, rng, ascent_budget, move="all", score_lookup=None):
    """Run ascent from nominal starts until ascent_budget calls are spent."""
    visited = set()
    while ev.calls < ascent_budget:
        start = problem.bits_to_id(problem.draw_nominal(1, rng))[0]
        visited.update(ascent_from(problem, start, ev, move, score_lookup))
    return np.array(sorted(visited))


def build_mixture(problem, mode_ids, eps=0.1, gamma=0.3):
    """Mode Bernoulli parameters R (n_mode, ngen) and mixture weights omega.

    Each mode is weighted by its nominal probability so the proposal concentrates on
    the most probable configurations within the tail (the dominant points).
    """
    bits = problem.to_binary(mode_ids)
    R = bits * (1 - eps) + (1 - bits) * eps
    log_w = gamma * problem.log_nominal_pmf(bits)
    omega = np.exp(log_w - log_w.max())
    return R, omega / omega.sum()


def mixture_logpmf_terms(bits, R):
    """log pmf of each mode component for each sample: returns (n_sample, n_mode)."""
    lr, l1r = np.log(R), np.log(1 - R)
    return bits @ lr.T + (1 - bits) @ l1r.T


def sample_mixture(problem, R, omega, n, rng, beta=0.2):
    """Draw n configs from beta*nominal + (1-beta)*sum_m omega_m Bern(R_m)."""
    comp = rng.random(n) >= beta                       # True -> a mode component
    out = problem.draw_nominal(n, rng)
    n_mode = (comp).sum()
    if n_mode:
        m = rng.choice(len(omega), size=n_mode, p=omega)
        out[comp] = (rng.random((n_mode, problem.ngen)) < R[m]).astype(np.int8)
    return out


def mixture_logpmf(problem, bits, R, omega, beta=0.2):
    """log q(x) for the full defensive mixture."""
    log_nom = problem.log_nominal_pmf(bits)
    mode_terms = mixture_logpmf_terms(bits, R) + np.log(omega)[None, :]
    log_mode = _logsumexp(mode_terms, axis=1)
    stacked = np.stack([np.log(beta) + log_nom, np.log(1 - beta) + log_mode])
    return _logsumexp(stacked, axis=0)


def _logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def sample(problem, ev, rng, budget, ascent_frac=ASCENT_FRAC, eps=0.1, beta=0.2, gamma=0.3,
           move="all", score_lookup=None):
    """Ascent-seeded defensive-IS weighted sample (ids, weights) at the call budget.

    move / score_lookup pick the ascent's per-step neighbor evaluations (all / best /
    top5); weights satisfy sum_i w_i g(x_i) ~ E_p[g] for any g (so sum_i w_i ~ 1)."""
    mode_ids = discover_modes(problem, ev, rng, int(ascent_frac * budget), move, score_lookup)
    R, omega = build_mixture(problem, mode_ids, eps, gamma)

    n_samp = max(1, budget - ev.calls)
    bits = sample_mixture(problem, R, omega, n_samp, rng, beta)
    ids = problem.bits_to_id(bits)
    ev(ids)
    log_w = problem.log_nominal_pmf(bits) - mixture_logpmf(problem, bits, R, omega, beta)
    return ids, np.exp(log_w) / n_samp
