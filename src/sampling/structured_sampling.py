"""Exact, mixing-free sampling of the block-uniform structured laws.

`cobuild` and `substitute` couple each fuel block ferromagnetically at the scale
`distributions.block_c` leaves --- strong enough that the block has a metastable all-on mode a
Gibbs sweep cannot cross, so Gibbs never reproduces the calibrated marginals (it stays in
whichever basin it starts in).  The same count collapse that makes the field solvable in
`structured_calibration.py` makes the block *exactly* sampleable with no chain at all: a
block's law factors as the count K of on-candidates times the choice of which K are on.

  - `cobuild`: draw K from its exact pmf (the Poisson-binomial pmf of the block's Bernoulli
    odds, tilted by the count energy), then draw the size-K on-set as a conditional Bernoulli
    --- a size-K subset with probability proportional to the product of the on odds.
  - `substitute`: draw the anchor bit from its exact marginal, then draw the rest as a cobuild
    block whose fields are shifted by -cp when the anchor is on (the anchor's -cp coupling).

Both are exact iid draws in O(m^2 + n*m) for n draws of a block of size m.  The block
structure, coupling sign, and anchor are read straight off the stored coupling J, so the
sampler draws exactly the stored law.
"""

import numpy as np

from src.sampling.structured_calibration import _sigmoid


def _suffix_pmf(q):
    """Suffix Poisson-binomial pmfs: A[i] is the pmf of sum of Bernoulli(q_i..q_{m-1}), A[m]=[1].

    Each A[i] is a proper pmf (sums to 1), so entries stay in [0, 1]; the recursion
    A[i] = conv([1-q_i, q_i], A[i+1]) is the same suffix convolution the calibration uses.
    """
    m = len(q)
    A = [None] * (m + 1)
    A[m] = np.array([1.0])
    for i in range(m - 1, -1, -1):
        A[i] = np.convolve([1.0 - q[i], q[i]], A[i + 1])
    return A


def _count_pmf(w, cp):
    """Normalized block count pmf: Poisson-binomial pmf w tilted by exp((cp/2)(k^2-k))."""
    k = np.arange(len(w))
    lg = 0.5 * cp * (k * k - k)
    pk = w * np.exp(lg - lg.max())
    return pk / pk.sum()


def _sample_sets(q, A, counts, rng):
    """A size-`count` on-set per draw, as a conditional Bernoulli, vectorized over the draws.

    Walk the m items once; include item i in a draw still needing j of its suffix with
    probability q_i * A[i+1][j-1] / A[i][j] --- the exact conditional-Bernoulli inclusion
    probability from the Poisson-binomial recursion A[i][j] = q_i*A[i+1][j-1] + (1-q_i)*A[i+1][j].
    """
    n, m = len(counts), len(q)
    X = np.zeros((n, m), dtype=np.int8)
    need = counts.astype(np.int64).copy()          # how many still to place per draw
    for i in range(m):
        ai, ai1 = A[i], A[i + 1]
        denom = ai[np.clip(need, 0, len(ai) - 1)]
        num = ai1[np.clip(need - 1, 0, len(ai1) - 1)]
        ratio = np.zeros(n)
        np.divide(num, denom, out=ratio, where=denom > 0)
        p_on = np.where(need > 0, q[i] * ratio, 0.0)
        take = rng.random(n) < p_on
        X[take, i] = 1
        need = need - take
    return X


def _sample_cobuild_block(h, cp, n, rng):
    """n exact iid draws (n, m) of a cobuild block: +cp on every within-block pair."""
    q = _sigmoid(h)
    A = _suffix_pmf(q)
    counts = rng.choice(len(A[0]), size=n, p=_count_pmf(A[0], cp))
    return _sample_sets(q, A, counts, rng)


def _anchor_prob(h, cp, anchor, rest):
    """Exact marginal P[anchor = 1] of a substitute block, from the two conditional partitions."""
    qr = _sigmoid(h[rest])
    w = _suffix_pmf(qr)[0]
    k = np.arange(len(w))
    lg0 = 0.5 * cp * (k * k - k)
    lg1 = lg0 - cp * k
    log_z0 = np.log(np.exp(lg0 - lg0.max()) @ w) + lg0.max()
    log_z1 = h[anchor] + np.log(np.exp(lg1 - lg1.max()) @ w) + lg1.max()
    return _sigmoid(log_z1 - log_z0)


def _sample_substitute_block(h, cp, anchor, n, rng):
    """n exact iid draws (n, m) of a substitute block: anchor -cp to the rest, rest +cp.

    Draw the anchor bit from its exact marginal; given it, the rest is a cobuild block whose
    fields drop by cp where the anchor is on (its -cp coupling suppresses every rest candidate).
    """
    m = len(h)
    rest = [i for i in range(m) if i != anchor]
    b = rng.random(n) < _anchor_prob(h, cp, anchor, rest)
    X = np.zeros((n, m), dtype=np.int8)
    X[:, anchor] = b.astype(np.int8)
    for bit in (0, 1):
        idx = np.where(b == bit)[0]
        if len(idx):
            X[np.ix_(idx, rest)] = _sample_cobuild_block(h[rest] - cp * bit, cp, len(idx), rng)
    return X


def _block_params(Jg):
    """(kind, cp, anchor) read off a block's coupling: cobuild if every pair is +cp, else substitute."""
    off = Jg - np.diag(np.diag(Jg))
    cp = float(off.max())
    if off.min() >= 0.0:
        return "cobuild", cp, None
    return "substitute", cp, int(off.sum(axis=1).argmin())   # the anchor row is the most negative


def sample_structured(h, J, groups, n, rng):
    """n exact iid draws (n, ngen) of a stored structured law, block by block off its (h, J).

    Each technology block is sampled from its own count collapse; a candidate in no group is an
    independent Bernoulli(sigmoid(h_i)).  The coupling J alone decides each block's kind, sign,
    and anchor, so the draws are of exactly the stored law.
    """
    ngen = len(h)
    X = np.zeros((n, ngen), dtype=np.int8)
    grouped = set()
    for g in groups:
        g = list(g)
        grouped.update(g)
        kind, cp, anchor = _block_params(J[np.ix_(g, g)])
        if kind == "cobuild":
            X[:, g] = _sample_cobuild_block(h[g], cp, n, rng)
        else:
            X[:, g] = _sample_substitute_block(h[g], cp, anchor, n, rng)
    for i in range(ngen):
        if i not in grouped:
            X[:, i] = (rng.random(n) < _sigmoid(h[i])).astype(np.int8)
    return X
