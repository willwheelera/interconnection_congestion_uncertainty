"""Exact weighted per-line |mu_l| statistics and the single line selection every grid shares.

Every per-line tail figure shows the same lines in the same order: the top lines by descending
E[|mu_l|] under the baseline law, so a given line sits in the same panel position across the
figures, which are read together.  `select_lines` is that one selection -- figures call it and
never re-derive an order or apply their own keep mask, so the choice cannot drift between
figures.  It applies no congestion filter; the ranking alone decides which lines appear.  The
statistics -- exceedance probability, mean, quantile, congestion frequency -- and the
survivor/quantile curve math also live here.

`mu` is always the (nconfig, nline) table of |mu_l| and `w` the nominal config weights.

Statistics (one value per line):
    exceedance_prob    P[|mu_l| > c]
    mean_abs_mu        E[|mu_l|]
    quantile_abs_mu    the q-quantile of |mu_l|
    congestion_freq    P[|mu_l| > floor], how often the line congests at all
Curves (one line's whole distribution):
    weighted_survivor  (sorted values, P[value > each])
    weighted_quantiles value at each exceedance level
    decimate_monotone  thin a survivor polyline without visible change
Selection:
    order_lines        line indices and names by descending score (optional top-n)
    select_lines       the single filter-free selection: top lines by descending E[|mu_l|]
"""

import numpy as np


def exceedance_prob(mu, w, threshold):
    """Exact nominal-weighted P[|mu_l| > threshold] for every line (full enumeration)."""
    return (w[:, None] * (mu > threshold)).sum(0)


def mean_abs_mu(mu, w):
    """Exact nominal-weighted mean of |mu_l| for every line."""
    return w @ mu


def weighted_quantile(values, weights, q):
    """The q-quantile of one line's values under weights (exact nominal-weighted truth)."""
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return np.interp(q, cw / cw[-1], values[order])


def quantile_abs_mu(mu, w, q):
    """Exact nominal-weighted q-quantile of |mu_l| for every line."""
    return np.array([weighted_quantile(mu[:, l], w, q) for l in range(mu.shape[1])])


def congestion_freq(mu, w, floor=0.0):
    """Exact nominal-weighted P[|mu_l| > floor] -- how often each line congests."""
    return w @ (mu > floor)


def weighted_survivor(values, weights):
    """Return (sorted values, proportion of weight with value > each)."""
    order = np.argsort(values)
    v = values[order]
    cumw = np.zeros(weights.shape)
    cumw[1:] = np.cumsum(weights[order])[:-1]
    return v, 1.0 - cumw / cumw[-1]


def weighted_quantiles(values, weights, probs):
    """Value q with weighted P[|mu_l| > q] = p, for each exceedance level p in probs."""
    v, s = weighted_survivor(values, weights)
    return np.interp(probs, s[::-1], v[::-1])


def decimate_monotone(v, s, nbins=4000):
    """Thin a monotone survivor curve to <=~2*nbins points without visible change.

    s(v) is non-increasing, so per x-bin the first/last points are the y-extremes;
    keeping them reproduces the rendered polyline to within one bin in x (chosen
    finer than screen pixels), fabricating no information.  The dense head collapses
    while every sparse-tail point lands in its own bin and is kept verbatim."""
    if v.size <= 2 * nbins:
        return v, s
    span = v[-1] - v[0]
    if span <= 0:
        return v[[0, -1]], s[[0, -1]]
    bins = np.floor((v - v[0]) / span * nbins).astype(np.int64)
    change = bins[1:] != bins[:-1]
    keep = np.zeros(v.size, bool)
    keep[0] = keep[-1] = True
    keep[1:] |= change                                      # first point of each bin
    keep[:-1] |= change                                     # last point of each bin
    return v[keep], s[keep]


def order_lines(scores, line_names, n=None):
    """Line indices and names by descending score; optional top-n cap.

    The lower-level sort behind `select_lines`.  `scores` is one score vector, or a list
    [primary, secondary, ...] to break ties by the later scores; ties left after the last
    score are broken by ascending line index, so the order is stable.  `n` caps how many
    are returned.  There is deliberately no keep mask -- the selection applies no congestion
    filter, so `select_lines` stays the single, filter-free rule every grid shares.
    """
    keys = scores if isinstance(scores, (list, tuple)) else [scores]
    idx = np.lexsort([-np.asarray(k) for k in reversed(keys)])
    if n is not None:
        idx = idx[:n]
    return idx, [line_names[i] for i in idx]


def select_lines(mu, line_names, weights, n=None):
    """THE per-line selection every grid shares: the lines with the largest E[|mu_l|] under
    the given (baseline) law, in that order, optionally capped to the top n.

    The single definition of which lines a per-line figure shows and in what order.  It applies
    no congestion or threshold filter -- the ranking alone decides -- so the same lines land in
    the same panel positions across every figure read alongside another.  Figures call this and
    never re-derive an order or invent a keep mask.
    """
    return order_lines(mean_abs_mu(mu, weights), line_names, n=n)
