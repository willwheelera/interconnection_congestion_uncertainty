""" Implement the simplest Metropolis sampler.
Symmetric moves, no bias, no importance weights.
Compare estimators: standard, rejection-averaged
"""

import numpy as np
import pandas as pd
from utils import avail_to_id, id_to_avail
import matplotlib.pyplot as plt
from scipy.stats import sem

from plot_dual_distributions import load_dual_cache
from pathlib import Path

NGEN = 14
BERNOULLI_PARAMETERS = np.full(NGEN, 0.1)
LOGP = np.log(BERNOULLI_PARAMETERS)
LOGQ = np.log(1 - BERNOULLI_PARAMETERS)

REPO_ROOT = Path(__file__).parent.parent


def run():
    lmp_df, mu_df = load_dual_cache(REPO_ROOT / "results" / "case14_gen_per_bus")
    #mu_df = pd.read_parquet(REPO_ROOT / "results" / "fake_data" / "dual_cache.parquet")
    S = mu_df.sum(axis=1)
    m, s = true_average(S)
    print(f"true avg {m:.3f} {s:.3f}")

    x0 = id_to_avail(16383, NGEN)
    trace, weights, acc = run_mc(x0, S, n_iter=2000)
    print(f"acceptance {acc:.2f}")
    plot_results(trace, weights)


def run_mc(x0, df, n_iter=100):
    x = x0
    trace = np.zeros((n_iter, 2))
    weights = np.ones(n_iter)
    w = 1.0
    acceptance = 0.0
    for i in range(n_iter):
        x, w, trace[i], a = move_w(x, w, df)
        weights[i] = w
        acceptance += a / n_iter
    return trace, weights, acceptance


def move_w(x, w, df):
    x_ = propose_move(x, 0.5)

    rp = p_ratio(x, x_, df.index)
    A = np.minimum(1.0, rp)

    v1 = df[avail_to_id(x)]
    if A == 0:
        return x, 1.0, [v1, v1], 0

    v2 = df[avail_to_id(x_)]
    y2 = (1-A)*v1 + A * v2
    f_ratio = v2 / v1
    Af = np.minimum(1.0, rp * f_ratio)

    accept = Af > np.random.random()
    if accept:
        x_new = x_
        y1 = v2
        w_new = w * A / Af
    else:
        x_new = x
        y1 = v1
        w_new = w * (1 - A) / (1 - Af)
    return x_new, w_new, [y1, y2], accept


def p_ratio(x, x_, allowed):
    n = avail_to_id(x_)
    if n not in allowed:
        return 0.0
    d = x_ - x
    return np.exp(d @ (LOGP - LOGQ))  # normalization cancels out here
    # prob    = p**x * (1-p)**(1-x)
    # newprob = p**x_ * (1-p)**(1-x_)
    # ratio = p**(x_ - x) * (1-p)**(-(x_-x))
    # logratio = d log p - d log (1-p)


def _new_prob(x, p_flip):
    return p_flip + x * (1 - 2 * p_flip)
    # = p_flip * (1 - 2*x) + x


def propose_move(x, p_flip=0.1):
    # This move is symmetric
    p = _new_prob(x, p_flip)
    xnew = np.random.random(len(x)) < p
    return xnew.astype(int)


def _reblock(array, nblocks, weights=None):
    if weights is None:
        weights = np.ones(len(array))
    vals = np.array_split(array, nblocks, axis=0)
    weights = np.array_split(weights, nblocks, axis=0)
    # The transposes are to use the broadcast rules of numpy in the case of vector data
    return [(v.T * w.T).T.mean(axis=0) / w.mean(axis=0) for v, w in zip(vals, weights)]


def bernoulli_prob(x):
    return np.exp(x @ LOGP + (1 - x) @ LOGQ)


def true_average(S):
    N = len(S)
    m = 0.0
    v = 0.0
    norm = 0.0
    for index, value in S.items():
        a = id_to_avail(index, NGEN)
        p = bernoulli_prob(a)
        m += p * value
        v += p * value**2
        norm += p
    var = v / norm - (m / norm) ** 2
    return m / norm, np.sqrt(var)


def plot_results(trace, weights):
    print("avg", np.average(trace[100:], axis=0, weights=weights[100:]))
    for nblocks in [20, 40, 60, 80, 100]:
        err = sem(_reblock(trace, nblocks, weights), axis=0)
        print(f"err {nblocks} blocks: {err[0]:.3f} {err[1]:.3f}")

    trace_mean = np.cumsum((trace.T*weights)[::-1], axis=1) / np.cumsum(weights[::-1])
    plt.plot(trace, lw=0.2)
    plt.plot(trace_mean[::-1].T)
    plt.plot(np.linspace(0, len(trace), 100), _reblock(trace, 100, weights))

    plt.figure()
    plt.plot(weights)
    plt.title("weights")
    plt.show()


if __name__ == "__main__":
    run()
