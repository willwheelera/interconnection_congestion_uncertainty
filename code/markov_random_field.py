
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


def mrf(x, a, b):
    r""" Markov random field
        x: binary availability (nsamples, ngen)
        a, b: parameters
        returns: log probability
       
        log P =  \sum_i a_i x_i + \sum_{ij} b_{ij} x_i x_j
              = x_{ni} a_i + x_{ni} b_{ij} x_{nj}
    """
    ng = x.shape[1]
    B = np.zeros((ng, ng))
    B[np.triu_indices(ng, 1)] = b
    return x @ a + np.einsum("ni,ij,nj->n", x, B, x)


def run():
    lmp_df, mu_df = load_dual_cache(REPO_ROOT / "results" / "case14_gen_per_bus")
    S = mu_df.sum(axis=1)
    # Learn best-fit distribution to full data
    a = np.random.random(NGEN)
    b = np.random.random(NGEN*(NGEN-1)//2)
    a, b = cross_entropy_minimization(S, a, b, n_iter=200, tstep=0.01)
    #a, b = cross_entropy_minimization(S, a, b, n_iter=10, tstep=0.01)

    #trace, acc = run_mc(x0, S, n_iter=6000)
    #print(f"acceptance {acc:.2f}")
    #print("avg", trace[100:].mean(axis=0))
    #for nblocks in [20, 40, 60, 80, 100]:
    #    err = sem(_reblock(trace, nblocks), axis=0)
    #    print(f"err {nblocks} blocks: {err[0]:.3f} {err[1]:.3f}")

    #trace_mean = np.cumsum(trace[::-1], axis=0) / np.arange(1, len(trace)+1)[:, np.newaxis]
    #plt.plot(trace)
    #plt.plot(trace_mean[::-1])
    #plt.plot(np.linspace(0, len(trace), 100), _reblock(trace, 100))
    #plt.show()

def cross_entropy_minimization(S, a0, b0, n_iter=10, tstep=0.01):
    X = all_vectors()
    S = S.sort_index()
    allowed = np.isin(np.arange(int(2**NGEN)), S.index)
    X = X[allowed]
    p = full_p(S, X) # doesn't change with parameters
    M = p @ S.values
    Ei = np.einsum("n,n,ni->i", p, S.values, X)
    Eij_ = np.einsum("n,n,ni,nj->ij", p, S.values, X, X)
    triu_i = np.triu_indices(NGEN, 1)
    Eij = Eij_[triu_i]

    a = a0.copy()
    b = b0.copy()
    for i in range(n_iter):
        q = full_q(X, a, b)
        Za = np.einsum("n,ni->i", q, X)
        Zb = np.einsum("n,ni,nj->ij", q, X, X)[triu_i]
        grad_a = (Ei - M * Za)
        grad_b = (Eij - M * Zb)
        if (i + 1) % (n_iter // 10) == 0:
            print(f"Step {i+1}:", np.einsum("n,n,n->", p, S.values, np.log(q)))
            print(f"    grad_a: {np.linalg.norm(grad_a)}")
            print(f"    grad_b: {np.linalg.norm(grad_b)}")

        a += tstep * grad_a
        b += tstep * grad_b
    return a, b

def step_direction(


def full_q(x, a, b):
    logq = mrf(x, a, b)
    q = np.exp(logq)
    return q / q.sum()
    

def full_p(df, x):
    p_ = np.exp(x @ LOGP + (1 - x) @ LOGQ)
    allowed = np.isin(np.arange(len(x)), df.index)
    p = p_ * allowed.astype(float) # not normalized
    return p / p.sum() # normalized


def all_vectors():
    powers = 2 ** np.arange(NGEN-1,-1,-1)
    index = np.arange(int(2**NGEN)).reshape(-1, 1)
    return ((index & powers) != 0).astype(int) # all possible availability vectors
    

    
if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    run()
    print("time", time.perf_counter() - t0)
