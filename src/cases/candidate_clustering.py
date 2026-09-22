"""Partition a queue's candidates into geographically compact groups by constrained k-means.

The coupled nominal laws couple within blocks; instead of one block per fuel (which on the
full leveraged queue makes a 95-candidate ferromagnet -- a strong fuel-wide correlation and a
metastable all-on mode), each technology is split into small groups of nearby candidates.  A
group's candidates share a location, so they share shift factors and load the same lines
together, which is what produces coordinated congestion.

The split is constrained k-means on the candidates' county coordinates with a hard size band
[lo, hi] (about `target`): standard k-means gives compact clusters, and the size band is
enforced by a transportation-LP assignment step each Lloyd iteration.  A fuel too small to
split (<= hi, or a count no [lo, hi] partition can cover, e.g. 7) stays a single group.  The
result is deterministic given the queue, so a law is built and sampled on identical blocks.
"""

import numpy as np
from scipy.optimize import linprog


def _project(lat, lng):
    """Equirectangular projection of county lat/lng to kilometers (a small region)."""
    r = 6371.0
    lat0 = float(lat.mean())
    x = np.radians(lng - lng.mean()) * np.cos(np.radians(lat0)) * r
    y = np.radians(lat - lat0) * r
    return np.column_stack([x, y])


def _kmeanspp(pts, k, rng):
    """k-means++ seeding: spread the initial centroids by squared-distance sampling."""
    n = len(pts)
    idx = [int(rng.integers(n))]
    d2 = ((pts - pts[idx[0]]) ** 2).sum(1)
    for _ in range(1, k):
        total = d2.sum()
        p = d2 / total if total > 0 else np.full(n, 1.0 / n)
        j = int(rng.choice(n, p=p))
        idx.append(j)
        d2 = np.minimum(d2, ((pts - pts[j]) ** 2).sum(1))
    return pts[idx].copy()


def _assign(pts, centroids, lo, hi):
    """Assign each point to a centroid minimizing total distance, cluster sizes in [lo, hi].

    A transportation LP: each point's row sums to 1, each cluster's column sums into [lo, hi].
    The constraint matrix is totally unimodular and the bounds are integer, so the HiGHS vertex
    is integral and the argmax read-off is exact.
    """
    n, k = len(pts), len(centroids)
    dist = np.sqrt(((pts[:, None, :] - centroids[None, :, :]) ** 2).sum(2))
    a_eq = np.zeros((n, n * k))
    for i in range(n):
        a_eq[i, i * k:(i + 1) * k] = 1.0
    a_ub = np.zeros((2 * k, n * k))
    for j in range(k):
        col = np.arange(n) * k + j
        a_ub[j, col], a_ub[k + j, col] = 1.0, -1.0
    b_ub = np.concatenate([np.full(k, hi), np.full(k, -lo)])
    res = linprog(dist.ravel(), A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=np.ones(n),
                  bounds=(0.0, 1.0), method="highs")
    return res.x.reshape(n, k).argmax(1)


def _constrained_kmeans(pts, k, lo, hi, seed, iters=100):
    """Lloyd iterations with the size-constrained assignment; returns integer labels."""
    rng = np.random.default_rng(seed)
    centroids = _kmeanspp(pts, k, rng)
    labels = None
    for _ in range(iters):
        new = _assign(pts, centroids, lo, hi)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            sel = labels == j
            if sel.any():
                centroids[j] = pts[sel].mean(0)
    return labels


def _choose_k(n, lo, hi, target):
    """Group count for n points near `target`, feasible for sizes in [lo, hi] (else 1)."""
    if n <= hi:
        return 1
    kmin, kmax = -(-n // hi), n // lo          # ceil(n/hi), floor(n/lo)
    if kmin > kmax:
        return 1
    return int(min(max(round(n / target), kmin), kmax))


def compact_groups(candidates, target=5, lo=4, hi=6, seed=0):
    """Candidate indices partitioned into geographically compact groups, one pass per fuel.

    Each technology's candidates are split by constrained k-means on their county coordinates
    into groups sized [lo, hi]; a fuel too small to split stays a single group.  Deterministic
    given the queue.
    """
    by_fuel: dict = {}
    for i, c in enumerate(candidates):
        by_fuel.setdefault(c["fuel"], []).append(i)
    groups = []
    for idx in by_fuel.values():
        n = len(idx)
        k = _choose_k(n, lo, hi, target)
        if k == 1:
            groups.append(list(idx))
            continue
        lat = np.array([candidates[i]["lat"] for i in idx])
        lng = np.array([candidates[i]["lng"] for i in idx])
        labels = _constrained_kmeans(_project(lat, lng), k, lo, hi, seed)
        for j in range(k):
            groups.append([idx[t] for t in range(n) if labels[t] == j])
    return groups
