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
import time

NGEN = 14
BERNOULLI_PARAMETERS = np.full(NGEN, 0.1)
LOGP = np.log(BERNOULLI_PARAMETERS)
LOGQ = np.log(1 - BERNOULLI_PARAMETERS)

REPO_ROOT = Path(__file__).parent.parent


def run():
    t0 = time.perf_counter()
    path = REPO_ROOT / "results" / "case14_gen_per_bus" / "dual_cache.parquet"
    df = load_dual_cache(path)
    lmp_df, mu_df = df["lmp"], df["mu"]
    #mu_df = pd.read_parquet(REPO_ROOT / "results" / "fake_data" / "dual_cache.parquet")
    #S = mu_df.sum(axis=1)
    #m, s = true_average(S)
    #print(f"true avg {m:.3f} {s:.3f}")

    print("start", time.perf_counter() - t0)
    x0 = id_to_avail(16383, NGEN)
    trace, weights, acc = run_mc_w(x0, mu_df, n_iter=200)
    print("weights finished", time.perf_counter() - t0)
    x0 = id_to_avail(16383, NGEN)
    trace_p, acc = run_mc_p(x0, mu_df, n_iter=200)
    print("raw finished", time.perf_counter() - t0)
    print(f"acceptance {acc:.2f}")
    plot_threshold_results(mu_df, trace, weights, label="weights")
    plot_threshold_results(mu_df, trace_p, np.ones(len(trace)), label="raw")
    print("plots finished", time.perf_counter() - t0)
    plt.show()


def run_mc_w(x0, df, n_iter=100):
    x = x0
    trace = np.zeros((n_iter, 2, df.shape[1]))
    weights = np.ones(n_iter)
    w = 1.0
    acceptance = 0.0
    for i in range(n_iter):
        x, w, trace[i], a = move_w(x, w, df)
        weights[i] = w
        acceptance += a / n_iter
    return trace, weights, acceptance


def target_threshold(df, x, mu=100):
    row = df.loc[avail_to_id(x)].values
    return (row > mu).astype(int), row.sum()


def move_w(x, w, df):
    v1, v1s = target_threshold(df, x)
    x_ = propose_move(x, 0.1)
    rp = p_ratio(x, x_, df.index)
    if rp == 0:
        return x, w, [v1, v1], 0

    A = rp / (1 + rp)
    v2, v2s = target_threshold(df, x_)
    y2 = (1-A)*v1 + A * v2
    f_ratio = ((v2s + 1e-1) / (v1s + 1e-1)) **0.5
    #Af = np.minimum(1.0, rp * f_ratio)
    Af = rp * f_ratio / (1 + rp * f_ratio)

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


def run_mc_p(x0, df, n_iter=100):
    x = x0
    trace = np.zeros((n_iter, 2, df.shape[1]))
    acceptance = 0.0
    for i in range(n_iter):
        x, val, a = move_p(x, df)

        acceptance += a / n_iter
        trace[i] = val
    return trace, acceptance


def move_p(x, df):
    x_ = propose_move(x, 0.5)

    rp = p_ratio(x, x_, df.index)
    A = np.minimum(1.0, rp)

    v1, _ = target_threshold(df, x)
    y1 = y2 = v1
    x_new = x
    acc = 0
    if A > 0:
        v2, _ = target_threshold(df, x_)
        y2 = (1-A)*v1 + A * v2
        check = np.random.random()
        if A > check:
            x_new = x_
            y1 = v2
            acc = 1
    return x_new, [y1, y2], A


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


def propose_move(x, p_flip=0.1):
    # This move is symmetric
    new_p = p_flip + x * (1 - 2 * p_flip)
    xnew = np.random.random(len(x)) < new_p
    return xnew.astype(int)


def _reblock(array, nblocks, weights=None):
    if weights is None:
        weights = np.ones(len(array))
    vals = np.array_split(array, nblocks, axis=0)
    weights = np.array_split(weights, nblocks, axis=0)
    # The transposes are to use the broadcast rules of numpy in the case of vector data
    return np.array([(v.T * w.T).T.mean(axis=0) / w.mean(axis=0) for v, w in zip(vals, weights)])


def bernoulli_prob(x):
    return np.exp(x @ LOGP + (1 - x) @ LOGQ)


def true_average(S):
    N = len(S)
    S.index = S.index.astype(int)
    m = pd.Series(index=S.columns, data=0.0) # if S has columns
    v = pd.Series(index=S.columns, data=0.0)
    norm = 0.0
    for index, value in S.iterrows():
        a = id_to_avail(index, NGEN)
        p = bernoulli_prob(a)
        f = (value > 100).astype(int)
        m += p * f
        v += p * f**2
        norm += p
    var = v / norm - (m / norm) ** 2
    return m / norm, np.sqrt(var)


def plot_threshold_results(df, trace, weights, label=""):
    avg = np.average(trace[100:], axis=0, weights=weights[100:])
    print("avg", np.around(avg, 3))
    t0 = time.perf_counter()
    mean, sdev = true_average(df)
    print("  true avg", time.perf_counter() - t0)

    fig, axs = plt.subplots(4, 5, figsize=(8, 6), sharex=True, sharey=True)
    rb = _reblock(trace, 100, weights)
    print("  reblock", time.perf_counter() - t0)
    trace_mean = np.cumsum((trace.T*weights)[..., ::1], axis=-1) / np.cumsum(weights[::1])
    print("  trace mean", time.perf_counter() - t0)
    trace_mean = np.cumsum((trace.T*weights)[..., ::1], axis=-1) / np.cumsum(weights[::1])
    for i, ax in enumerate(axs.ravel()):
        ax.plot(trace[:, :, i], lw=0.2, label="trace")
        #x = np.linspace(0, trace.shape[0], 100)
        #ax.plot(x, rb[:, :, i], label="reblocked")
        ax.plot(trace_mean[i].T[::1], label="trace_mean", lw=0.2)
        ax.axhline(mean[str(i+1)], lw=0.2, c="k", ls=":")
        ax.set_title(f"{i+1}", y=.9)
    print("  subplots", time.perf_counter() - t0)
    fig.suptitle(label)
    plt.legend()

    plt.figure()
    plt.plot(weights)
    plt.title("weights")

    

def plot_results(trace, weights):
    print("avg", np.average(trace[100:], axis=0, weights=weights[100:]))
    for nblocks in [20, 40, 60, 80, 100]:
        err = sem(_reblock(trace, nblocks, weights), axis=0)
        print(f"err {nblocks} blocks: {err[0]:.3f} {err[1]:.3f}")

    trace_mean = np.cumsum((trace.T*weights)[:, ::-1], axis=1) / np.cumsum(weights[::-1])

    plt.figure()
    plt.plot(trace, lw=0.2, label="trace")
    plt.plot(trace_mean[:, ::-1].T, label="trace_mean")
    plt.plot(np.linspace(0, len(trace), 100), _reblock(trace, 100, weights), label="reblocked")
    plt.legend()

    plt.figure()
    plt.plot(weights)
    plt.title("weights")


if __name__ == "__main__":
    run()
