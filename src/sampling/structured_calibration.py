"""Exact, enumeration-free field calibration for the block-uniform structured laws.

`cobuild` and `substitute` couple each fuel block with a single within-block coefficient,
so a block's law depends on its bits only through the count K of on-candidates (`cobuild`)
or through the pair (anchor bit, rest count) (`substitute`).  The 2^m sum over a block of
size m therefore collapses to a sum over K = 0..m: the per-count weights are the
Poisson-binomial pmf of the block's Bernoulli odds, tilted by the count-only interaction
energy.  This gives the exact per-node marginals in O(m^2) without enumerating the block.

The field is then solved block by block by damped Newton on those exact marginals, reaching
the target build probabilities to machine precision.  This is the deterministic counterpart
of the dense-law MCMC calibrator in `field_calibration.py`; the two together replace the old
biased `mrf.calibrate_field_mcmc`.  It matches the enumerated `mrf.calibrate_field` on any
block small enough to enumerate, and unlike it scales to the full leveraged queue.

The blocks are the queue's technology groups.  A group's within-block coefficient is scaled
by `distributions.block_c` (the same Curie-Weiss normalization the law's coupling uses), and
`substitute`'s anchor is the group's highest-probability candidate, matching
`distributions.substitute_coupling` exactly so the calibrated field is the field of that J.
"""

import numpy as np

from src.sampling.distributions import block_c


def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700))),
                    np.exp(np.clip(z, -700, 700)) / (1.0 + np.exp(np.clip(z, -700, 700))))


def _leave_one_out(q):
    """Poisson-binomial pmf of sum_i Bernoulli(q_i), and each leave-one-out pmf.

    Returns (w, w1): w is the full pmf (len m+1); w1[i] is the pmf of the other m-1 nodes
    (len m).  Built from prefix and suffix convolutions so all m leave-one-outs cost O(m^2).
    """
    m = len(q)
    pre = [np.array([1.0])]
    for qi in q:
        pre.append(np.convolve(pre[-1], [1.0 - qi, qi]))
    suf = [None] * (m + 1)
    suf[m] = np.array([1.0])
    for t in range(m - 1, -1, -1):
        suf[t] = np.convolve([1.0 - q[t], q[t]], suf[t + 1])
    w1 = [np.convolve(pre[i], suf[i + 1]) for i in range(m)]
    return pre[m], w1


def _cobuild_marginals(h, cp):
    """Exact per-node marginals of a cobuild block: +cp on every within-block pair.

    Block energy is sum_i h_i x_i + (cp/2)(K^2 - K) with K the count on, so the count law
    is the Poisson-binomial pmf tilted by exp((cp/2)(K^2-K)); node i's marginal is its odds
    times the tilted leave-one-out mass where i is on.
    """
    m = len(h)
    q = _sigmoid(h)
    w, w1 = _leave_one_out(q)
    k = np.arange(m + 1)
    lg = 0.5 * cp * (k * k - k)
    b = np.exp(lg - lg.max())
    S = b @ w
    return np.array([q[i] * (b[1:] @ w1[i]) / S for i in range(m)])


def _substitute_marginals(h, cp, anchor):
    """Exact per-node marginals of a substitute block: anchor -cp to the rest, rest +cp.

    Conditioning on the anchor bit splits the block into two cobuild-style laws over the
    rest -- tilt (cp/2)(K^2-K) when the anchor is off, and that minus cp*K when it is on
    (the anchor's -cp coupling suppresses every co-built rest candidate).  The anchor's own
    marginal is the ratio of the two conditional partition functions.
    """
    m = len(h)
    rest = [i for i in range(m) if i != anchor]
    qr = _sigmoid(h[rest])
    mr = len(rest)
    w, w1 = _leave_one_out(qr)
    k = np.arange(mr + 1)
    lg0 = 0.5 * cp * (k * k - k)
    lg1 = lg0 - cp * k
    b0 = np.exp(lg0 - lg0.max())
    b1 = np.exp(lg1 - lg1.max())
    S0, S1 = b0 @ w, b1 @ w
    log_z0 = np.log(S0) + lg0.max()
    log_z1 = h[anchor] + np.log(S1) + lg1.max()
    pa = _sigmoid(log_z1 - log_z0)

    p = np.empty(m)
    p[anchor] = pa
    for idx, r in enumerate(rest):
        m0 = qr[idx] * (b0[1:] @ w1[idx]) / S0
        m1 = qr[idx] * (b1[1:] @ w1[idx]) / S1
        p[r] = (1.0 - pa) * m0 + pa * m1
    return p


def _newton(marg_fn, target, h, iters=80, tol=1e-13, eps=1e-6):
    """Levenberg step from a warm start h for the field whose marginals `marg_fn(h)` = target.

    The Jacobian is the block covariance, taken by central finite differences of the exact
    marginals.  Damping `lam` is raised until a step strictly reduces the max residual and
    lowered on success, so the step is bounded even where the Jacobian is near-singular --- as
    it is when many block candidates share a target marginal (a permutation null space).  A
    plain Newton solve there returns a runaway step; the adaptive damping rejects it and shrinks
    the step until it descends, keeping full Newton speed where the Jacobian is well-conditioned.
    """
    h = h.copy()
    m = len(h)
    eye = np.eye(m)
    r = target - marg_fn(h)
    merit = float(r @ r)
    lam = 1e-10
    for _ in range(iters):
        if np.abs(r).max() < tol:
            break
        jac = np.empty((m, m))
        for j in range(m):
            hp, hm = h.copy(), h.copy()
            hp[j] += eps
            hm[j] -= eps
            jac[:, j] = (marg_fn(hp) - marg_fn(hm)) / (2.0 * eps)
        for _ in range(60):                      # raise damping until the step descends
            step = np.linalg.solve(jac + lam * eye, r)
            rn = target - marg_fn(h + step)
            if float(rn @ rn) < merit:           # the damped step descends the 2-norm
                h, r, merit = h + step, rn, float(rn @ rn)
                lam = max(lam / 3.0, 1e-13)
                break
            lam *= 8.0
        else:
            break                                # damping could not find a descent step
    return h


def _calibrate_block(make_marg, target, cp):
    """Field for one block by continuation in the coupling from 0 (Bernoulli) to cp.

    At cp = 0 the field is exactly logit(target); at the target cp a large block sits in the
    near-bistable ferromagnetic regime `block_c` leaves, where Newton started at logit(target)
    does not converge.  Ramping cp in small per-node increments and warm-starting Newton at
    each keeps every solve near its start, so it tracks the solution through the stiff regime.
    `make_marg(c)` returns the exact-marginal function of the block at coupling c.
    """
    h = np.log(target / (1.0 - target))
    if cp == 0.0:
        return h
    n_steps = max(1, int(np.ceil(cp * (len(target) - 1) / 0.5)))
    for c in np.linspace(0.0, cp, n_steps + 1)[1:]:
        h = _newton(make_marg(c), target, h)
    return h


def calibrate_structured_field(name, groups, p, ngen, c, ref_block=None):
    """Field h whose exact marginals under the named structured coupling are p.

    Solves each technology block independently (the coupling is block-diagonal, so the law
    factors over blocks); ungrouped candidates keep the Bernoulli field logit(p).
    """
    p = np.asarray(p, dtype=float)
    h = np.log(p / (1.0 - p))
    for g in groups:
        g = list(g)
        cp = block_c(c, len(g), ref_block)
        pt = p[g]
        if name == "mrf_cobuild":
            make_marg = lambda c: (lambda hh: _cobuild_marginals(hh, c))
        elif name == "mrf_substitute":
            anchor = int(np.argmax(pt))
            make_marg = lambda c, a=anchor: (lambda hh: _substitute_marginals(hh, c, a))
        else:
            raise ValueError(f"{name} is not a block-uniform structured law")
        h[g] = _calibrate_block(make_marg, pt, cp)
    return h


def structured_marginals(name, groups, h, p, c, ref_block=None):
    """Exact per-node marginals of the named structured law at field h, block by block.

    The analytic companion to `calibrate_structured_field`: it evaluates the same count-collapse
    marginals a calibrated field is solved against, so `max |structured_marginals - p|` is the
    calibration's exact error (no Monte Carlo).  `p` selects each substitute block's anchor, as
    the coupling does; ungrouped candidates are independent Bernoulli(sigmoid(h)).
    """
    h, p = np.asarray(h, dtype=float), np.asarray(p, dtype=float)
    m = _sigmoid(h)
    for g in groups:
        g = list(g)
        cp = block_c(c, len(g), ref_block)
        if name == "mrf_cobuild":
            m[g] = _cobuild_marginals(h[g], cp)
        elif name == "mrf_substitute":
            m[g] = _substitute_marginals(h[g], cp, int(np.argmax(p[g])))
        else:
            raise ValueError(f"{name} is not a block-uniform structured law")
    return m
