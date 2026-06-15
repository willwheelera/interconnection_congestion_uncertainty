"""
Bernoulli importance sampling with cross-entropy training.

Scenarios in results/case14_gen_per_bus/ are indexed by 14-bit availability
codes — bit i = 1 means generator i+1 is available.  We model scenario draws
as independent Bernoullis Bern(p_i), conditioned on at least MIN_ON
generators being on (smaller fleets are load-infeasible).  The cross-entropy
method (Kroese 2011) tunes a Bernoulli proposal q so it puts mass on
scenarios with large target H(x) = sum_b mu(x)_b**2.

Usage:
    uv run python src/bernoulli_sampler.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

from utils.availability import avail_to_id, id_to_avail

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "case14_gen_per_bus"

NGEN = 14
MIN_ON = 8
EPS = 1e-12
BERNOULLI_PARAMETERS = np.full(NGEN, 0.8)


def log_pmf(x, p):
    """log of the unconditional Bernoulli pmf prod_i p_i^x_i (1-p_i)^(1-x_i)."""
    return x @ np.log(p + EPS) + (1 - x) @ np.log(1 - p + EPS)


def load_duals() -> pd.DataFrame:
    """Cached dual variables (columns lmp/<bus> and mu/<branch>) per scenario."""
    df = pd.read_parquet(RESULTS_DIR / "dual_cache.parquet")
    df.index = df.index.astype(int)
    return df.sort_index()


def calc_weights(samples, p, q, allowed):
    """Per-sample importance sampling ratio p(x)/q(x) 
    Note that p(x) is defined to be zero on samples not in the data (for this example)
    q(x) is not ever restricted to be zero
    """
    log_ratio = np.array([log_pmf(x, p) - log_pmf(x, q) for x in samples])
    in_data = np.array([avail_to_id(x) in allowed for x in samples])
    return np.exp(log_ratio) * in_data
    

def calc_H(samples, df, target_fn):
    """Per-sample target H(x).  
    Scenarios absent from df (load-infeasible) get H = 0 via reindex+fillna.  """
    ids = [avail_to_id(x) for x in samples]
    H_vals = target_fn(df.reindex(ids)).fillna(0.0).values
    return H_vals


def cross_entropy_step(p, q, df, target_fn, samples, smoothing=0.3):
    """One CE update of q toward the optimal proposal proportional to H(x)p(x).

    smoothing ∈ [0, 1] is the weight kept on the previous q (Kroese eq. for
    smooth updating); smoothing=0 takes the pure CE estimate.
    """
    w = calc_weights(samples, p, q, df.index)
    H = calc_H(samples, df, target_fn)
    wH = w * H
    q_est = (wH[:, None] * samples).sum(axis=0) / wH.sum()
    q_new = smoothing * q + (1 - smoothing) * q_est
    q_new = np.clip(q_new, 1e-6, 1 - 1e-6)
    est = wH.mean()
    return q_new, est


def sample(p, rng):
    return (rng.random(NGEN) < p).astype(np.int8)


def train(
    p, df, target_fn, rng, Z_p=1., n_iter=20, n_samples=200, smoothing=0.3, q_init=None,
) -> tuple[np.ndarray, list[float]]:
    """Iteratively refine q via CE updates, starting from q_init (default p)."""
    q = q_init.copy() if q_init is not None else p.copy()
    history = []
    for t in range(n_iter):
        samples = np.array([sample(q, rng) for _ in range(n_samples)])
        q, est = cross_entropy_step(p, q, df, target_fn, samples, smoothing)
        est /= Z_p
        history.append(est)
        print(f"iter {t:3d}  est={est:.4g}  q={np.round(q, 2)}")
    return q, history


def estimator_stats(
    p, q, df, target_fn, rng, Z_p=1., n_samples=500, n_replicates=20,
) -> tuple[float, float]:
    """Mean and std of the importance sampling estimator over replicates."""
    estimates = np.empty(n_replicates)
    for r in range(n_replicates):
        samples = np.array([sample(q, rng) for _ in range(n_samples)])
        w = calc_weights(samples, p, q, df.index)
        H = calc_H(samples, df, target_fn)
        estimates[r] = (H * w).mean()
    estimates /= Z_p
    return float(estimates.mean()), float(estimates.std(ddof=1))


def target_mu_sq(df: pd.DataFrame) -> pd.Series:
    return (df["mu"] ** 2).sum(axis=1)

def mu_sum(df: pd.DataFrame) -> pd.Series:
    return (df["mu"]).abs().sum(axis=1)

def lambda_sum(df: pd.DataFrame) -> pd.Series:
    return (df["lmp"]).abs().sum(axis=1)


def main():
    rng = np.random.default_rng(0)
    df = load_duals()
    target_fn = mu_sum
    print(f"Loaded duals for {len(df)} scenarios")

    p = BERNOULLI_PARAMETERS
    print(f"Initial generator availability p = {p}")
    allprobs = np.exp(np.array([log_pmf(id_to_avail(_id, NGEN), p) for _id in df.index]))
    Z_p = allprobs.sum()

    q = cross_entropy_deterministic(p, target_fn(df))
    compare_true_result(p, q, df, target_fn)
    #q_trained, _ = train(p, df, target_fn, rng, n_iter=20, n_samples=200, smoothing=0.2, Z_p=Z_p)
    #print(f"\nTrained q = {np.round(q_trained, 3)}")

    #kws = dict(n_samples=50, n_replicates=20)
    #compare_samplers(df, p, q_trained, target_mu_sq, allprobs, label="mu^2", **kws)
    #compare_samplers(df, p, q_trained, mu_sum, allprobs, label="mu1", **kws)
    #compare_samplers(df, p, q_trained, lambda_sum, allprobs, label="lambda1", **kws)


def cross_entropy_deterministic(p, S):
    """One CE update of q toward the optimal proposal proportional to H(x)p(x).
    """
    X_all = all_vectors(NGEN)
    allowed = np.isin(np.arange(len(X_all)), S.index)
    X = X_all[allowed]
    P = full_p(X, p)
    M = P @ S.values
    q = np.einsum("n,n,ni->i", P, S.values, X) / M
    return q


def full_p(X, param):
    p = np.exp(log_pmf(X, param))
    return p / p.sum()

def all_vectors(n):
    powers = (2 ** np.arange(n)).astype(int)
    index = np.arange(int(2 ** n), dtype=int)
    return ((index[:, np.newaxis] & powers[::-1]) != 0).astype(int) # all possible availability vectors

def compare_true_result(P, Q, df, target):
    S = target(df)
    X_all = all_vectors(NGEN)
    allowed = np.isin(np.arange(len(X_all)), S.index)
    X = X_all[allowed]
    p = full_p(X, P)
    q = full_p(X_all, Q)[allowed]
    M = p @ S.values
    var_p = p @ S.values**2 - M**2
    var_q = q @ (p/q * S.values)**2 - M**2

    q_star = p * S.values / M
    ce_star = q_star @ np.log(q_star) * M
    ce_p = p @ (S.values * np.log(p))
    ce_q = p @ (S.values * np.log(q))
    print(f"mean {M:.2f}")
    print(f"sigma p {np.sqrt(var_p):.2f}")
    print(f"sigma q {np.sqrt(var_q):.2f}")
    print(f"ce q* (ref) {ce_star:.2f}")
    print(f"ce p  {ce_p:.2f}")
    print(f"ce q  {ce_q:.2f}")

def compare_samplers(df, p, q, target_fn, allprobs, label="", n_samples=500, n_replicates=20):
    rng = np.random.default_rng(1)
    Z_p = allprobs.sum()
    print(f"\n{label:<10} estimator comparison (n_samples={n_samples}, n_replicates={n_replicates})")
    m_p, s_p = estimator_stats(p, p, df, target_fn, rng, Z_p=Z_p, n_samples=n_samples, n_replicates=n_replicates)
    m_q, s_q = estimator_stats(p, q, df, target_fn, rng, Z_p=Z_p, n_samples=n_samples, n_replicates=n_replicates)
    print(f"  q = p:       mean={m_p:.4g}  std={s_p:.4g}")
    print(f"  q = trained: mean={m_q:.4g}  std={s_q:.4g}")

    true_mean = (target_fn(df) * allprobs).sum() / allprobs.sum()
    centered = target_fn(df) - true_mean
    true_var = (centered**2 * allprobs).sum() / allprobs.sum() 
    true_std = np.sqrt(true_var)
    print(f"  True value:  mean={true_mean:.4g}  std={true_std:.4g}")


if __name__ == "__main__":
    main()
