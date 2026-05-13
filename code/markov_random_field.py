import numpy as np
import pandas as pd
import time
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.optimize

from plot_dual_distributions import load_dual_cache


NGEN = 14
BERNOULLI_PARAMETERS = np.full(NGEN, 0.1)
LOGP = np.log(BERNOULLI_PARAMETERS)
LOGQ = np.log(1 - BERNOULLI_PARAMETERS)

REPO_ROOT = Path(__file__).parent.parent

def run():
    lmp_df, mu_df = load_dual_cache(REPO_ROOT / "results" / "case14_gen_per_bus")
    S = mu_df.sum(axis=1)

    fname = "mrf_var_parameters.npz"
    if not (Path(__file__).parent / fname).exists():
        a = np.random.random(NGEN)
        b = np.random.random(NGEN * (NGEN-1) // 2)
        h = direct_optimization(S, a, b)
        a, b = h[:NGEN], h[NGEN:]
        np.savez(fname, a=a, b=b)

    data = np.load(fname)
    a, b = data["a"], data["b"]
    compare_q(S, a, b)

def compare_q(S, a, b):
    X_all = all_vectors(NGEN)
    allowed = np.isin(np.arange(len(X_all)), S.index)
    X = X_all[allowed]
    p = full_p(X)
    fp = p * S.values
    fp /= fp.sum()
    q = full_q(X_all, a, b)[allowed]

    sigma_p = calc_stddev(p, S.values)
    sigma_q = calc_stddev(q, p/q*S.values)
    print(f"sigma p: {sigma_p}")
    print(f"sigma q: {sigma_q}")
    plt.plot(fp, label="fp")
    plt.plot(p, label="p")
    plt.plot(q, label="q")
    plt.legend()

    plt.figure()
    plt.semilogy(fp/p+1e-5, label="fp/p")
    plt.semilogy(fp/q+1e-5, label="fp/q")
    plt.title("ratio")
    plt.show()

def calc_stddev(p, est):
    m = p @ est
    v = p @ (est - m)**2
    return np.sqrt(v)
    

def direct_optimization(S, a0, b0):
    print("Initializing optimization", flush=True)
    t0 = time.perf_counter()
    #func = CrossEntropy(S)
    func = EstVariance(S)
    cb_func = CallbackFn()
    h0 = np.concatenate([a0, b0])
    res = scipy.optimize.minimize(func, h0, callback=cb_func, jac=True)
    print(res.success)
    print(res.message)
    print(f"optimization time {time.perf_counter()-t0:.3f}")
    return res.x
    
    
class CallbackFn:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.iter = 0

    def __call__(self, intermediate_result):
        self.iter += 1
        if self.iter % 1 == 0:
            s_iter = f"Iteration {self.iter}"
            s_val = f"val {intermediate_result.fun:.3f}"
            s_time = f"time {time.perf_counter() - self.t0:.2f}"
            print(f"{s_iter} {s_val} {s_time}")


class EstVariance:
    def __init__(self, S):
        self.f = S.values
        self.allowed = np.isin(np.arange(int(2**NGEN)), S.index)
        self.X_all = all_vectors(NGEN)
        self.X = self.X_all[self.allowed]
        self.p = full_p(self.X)
        self.M = self.p @ self.f
        print(f"Mean M: {self.M}")
        
    def __call__(self, h):
        a, b = h[:NGEN], h[NGEN:]
        q = full_q(self.X_all, a, b)
        y = (self.p * self.f)**2 / q[self.allowed]
        V = np.sum(y) - self.M**2

        Eqi = np.einsum("n,ni->i", q, self.X_all)
        Eqij = np.einsum("n,ni,nj->ij", q, self.X_all, self.X_all)
        grad_a = -y @ (self.X - Eqi)
        grad_b = -np.einsum("n,ni,nj->ij", y, self.X, self.X) + y.sum() * Eqij

        triu_i = np.triu_indices(NGEN, 1)
        grad_h = np.concatenate([grad_a, grad_b[triu_i]])
        return V, grad_h

class CrossEntropy:
    def __init__(self, S):
        self.f = S.values
        self.X_all = all_vectors(NGEN)
        self.allowed = np.isin(np.arange(int(2**NGEN)), S.index)
        self.X = self.X_all[self.allowed]
        self.p = full_p(self.X)
        self.Epi = np.einsum("n,n,ni->i", self.p, self.f, self.X)
        self.Epij = np.einsum("n,n,ni,nj->ij", self.p, self.f, self.X, self.X)
        self.M = self.p @ self.f
        

    def __call__(self, h):
        a, b = h[:NGEN], h[NGEN:]
        q = full_q(self.X_all, a, b)
        val = -np.einsum("n,n,n->", self.p, self.f, np.log(q[self.allowed]))

        Eqi = np.einsum("n,ni->i", q, self.X_all)
        Eqij = np.einsum("n,ni,nj->ij", q, self.X_all, self.X_all)
        grad_a = self.Epi - self.M * Eqi
        grad_b = (self.Epij - self.M * Eqij)[np.triu_indices(NGEN, 1)]
        grad = np.concatenate([grad_a, grad_b])

        return val, -grad
    

def mrf(X, a, b):
    B = np.zeros((len(a), len(a)))
    B[np.triu_indices(len(a), 1)] = b
    return X @ a + np.einsum("ni,ij,nj->n", X, B, X)


def full_q(X, a, b):
    logq = mrf(X, a, b)
    logq -= np.amax(logq)
    q = np.exp(logq)
    return q / q.sum()

def full_p(X):
    p = np.exp(X @ LOGP + (1 - X) @ LOGQ)
    return p / p.sum()


def all_vectors(n):
    powers = (2 ** np.arange(n)).astype(int)
    index = np.arange(int(2 ** n), dtype=int)
    return ((index[:, np.newaxis] & powers[::-1]) != 0).astype(int) # all possible availability vectors

if __name__ == "__main__":
    run()
    
