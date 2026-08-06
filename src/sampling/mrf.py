"""A pairwise Markov random field over binary configs --- the default nominal law.

    p(x) proportional to exp( sum_i h_i x_i + sum_{i<j} J_ij x_i x_j ),   x in {0,1}^n

stands in for the parameterized, correlated laws the project targets.  Single
coefficients h bias each generator toward on/off; pair coefficients J couple
generators, so the law is genuinely correlated and is *not* a product of marginals.
With J = 0 it collapses to an iid Bernoulli law, recovering the old testbed exactly.

`calibrate_field` solves for the single coefficients that reproduce a target set of
marginals under a given coupling, so two laws can be built to share marginals exactly
and differ only in J.

The samplers never import this class: they reach the nominal law only through
Problem's generic pmf lookup and categorical sampler, which consume the normalized
weights this class produces.  So nothing downstream assumes the pairwise (or any
other) form --- swapping in a richer MRF or any other law touches only this file.
"""

import numpy as np


class MarkovRandomField:
    """Pairwise binary MRF defined by single coefficients h and pair coefficients J."""

    def __init__(self, h, J):
        self.h = np.asarray(h, dtype=float)            # (n,) single coefficients
        self.J = np.asarray(J, dtype=float)            # (n, n) symmetric, zero diagonal
        self.ngen = len(self.h)

    def log_weight(self, bits):
        """Unnormalized log p(x) for each row of bits (m, ngen)."""
        single = bits @ self.h
        pair = 0.5 * ((bits @ self.J) * bits).sum(axis=1)   # = sum_{i<j} J_ij x_i x_j
        return single + pair


def all_configs(ngen):
    """Every config as a row of availability bits, id-ordered (generator 0 is the MSB)."""
    powers = (1 << np.arange(ngen - 1, -1, -1)).astype(np.int64)
    return ((np.arange(1 << ngen)[:, None] & powers[None, :]) != 0).astype(float)


def modal_config(mrf):
    """The single most probable configuration under the law, as an availability vector.

    Found by scanning all 2^ngen configurations: a coupled law's mode is not the
    per-generator threshold, so it cannot be read off the marginals.
    """
    bits = all_configs(mrf.ngen)
    return bits[int(mrf.log_weight(bits).argmax())].astype(np.int8)


def calibrate_field(J, p, tol=1e-9, iters=100):
    """The single field h whose exact marginals under coupling J are p.

    Damped Newton on the convex dual log Z(h) - p.h over the enumerated configs.  The
    mean-field compensation used elsewhere in this file is only a first-order correction
    and drifts once the coupling is strong, so a law meant to differ from Bernoulli in
    *correlation alone* has to have its field solved for rather than approximated.
    """
    bits = all_configs(len(p))
    pair = 0.5 * ((bits @ J) * bits).sum(axis=1)

    def weights(h):
        lw = bits @ h + pair
        w = np.exp(lw - lw.max())
        return w / w.sum()

    def dual(h):
        lw = bits @ h + pair
        return float(np.log(np.exp(lw - lw.max()).sum()) + lw.max() - p @ h)

    h = np.log(p / (1.0 - p))
    for _ in range(iters):
        w = weights(h)
        m = w @ bits
        grad = m - p
        if np.abs(grad).max() < tol:
            break
        cov = (bits * w[:, None]).T @ bits - np.outer(m, m)
        step = -np.linalg.solve(cov + 1e-9 * np.eye(len(p)), grad)
        f0, t = dual(h), 1.0
        for _ in range(40):                      # backtrack: the dual is convex in h
            if dual(h + t * step) <= f0 + 1e-4 * t * (grad @ step):
                break
            t *= 0.5
        h = h + t * step
    return h


def default_mrf(ngen, p_on=0.75, coupling=0.08, coupling_std=0.1, seed=0):
    """A fixed pairwise MRF tuned to spread the congestion target S over a graded tail.

    The single field is set so each generator's marginal stays near `p_on` after
    compensating for the mean coupling (the effective field per node is
    h + (ngen-1)*coupling*p_on).  A small ferromagnetic mean `coupling` adds variance
    to the number of on-generators, which fattens the lower-on configurations where S
    is large --- turning what would be a hard shelf into a graded rare-event tail while
    keeping the bulk (median S) low.  J is a fixed Gaussian coupling (seed-determined)
    with zero diagonal; coupling = coupling_std = 0 recovers iid Bernoulli(p_on).
    """
    rng = np.random.default_rng(seed)
    field = np.log(p_on / (1.0 - p_on)) - (ngen - 1) * coupling * p_on
    h = np.full(ngen, field)
    J = rng.normal(coupling, coupling_std, (ngen, ngen))
    J = np.triu(J, 1)                                  # keep upper triangle only
    return MarkovRandomField(h, J + J.T)               # symmetrize, zero diagonal
