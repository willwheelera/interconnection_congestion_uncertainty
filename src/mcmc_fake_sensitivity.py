"""Two single-flip samplers for the per-line congestion-dual (mu) marginals.

Estimate, per line, the distribution of mu_l over availability configs drawn
from a target MRF p(x).  Both walk the single-bit-flip neighbour graph:

* p-only (run_chain_p): target p, no mu information, unit weights.
* dmu-biased (run_chain_dmu): target pi(x) propto p(x)*exp(alpha*M(x)),
  M(x) = sum_l mu_l(x); each visit reweighted by exp(-alpha*M(x)) to recover p.

Writes a compact Parquet of weighted mu samples for plot_cdf_compare.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from informed_mcmc import MRF
from utils.availability import get_all_neighbors, id_to_avail

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "case14_gen_per_bus"

NGEN = 14
NCONF = 1 << NGEN
BITMASK = (1 << (NGEN - 1 - np.arange(NGEN))).astype(np.int64)


def get_neighbors(x):
    return x ^ BITMASK


def load_mu_table(case_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Dense mu lookup MU[id], shape (NCONF, n_lines), columns sorted by line.

    id 0 (invalid, absent from the cache) stays an all-zero row.
    """
    mu = pd.read_parquet(case_dir / "dual_cache.parquet")["mu"]
    mu.columns = mu.columns.astype(int)
    mu = mu.sort_index(axis=1)
    table = np.zeros((NCONF, mu.shape[1]))
    table[mu.index.to_numpy()] = mu.to_numpy()
    return table, [str(c) for c in mu.columns]


def all_unnormalized_p(target: MRF) -> np.ndarray:
    """Unnormalized p(x) for every id 0..NCONF-1; invalid id 0 set to 0.

    Shifted by the max log-prob over valid configs for numerical stability.
    """
    avail = np.array([id_to_avail(i, NGEN) for i in range(NCONF)])
    log_p = target.log_p(avail)
    p = np.exp(log_p - log_p[1:].max())
    p[0] = 0.0
    return p


def calc_dmu(table: np.ndarray, center, neigh) -> np.ndarray:
    """Signed total-mu change on a move, M(neigh) - M(center), summed over lines.

    center: one id (scalar) or batch (N,); neigh: its neighbours (NGEN,) or
    (N, NGEN).  Result drops the line axis to (NGEN,) or (N, NGEN); sign kept.
    """
    return (table[neigh] - table[center][..., None, :]).sum(axis=-1)


def scale_alpha(table: np.ndarray) -> float:
    """Rms single-flip |dmu| over all edges; used to set alpha = coef / scale."""
    ids = np.arange(1, NCONF)
    neigh = get_all_neighbors(ids, NGEN)
    d = calc_dmu(table, ids, neigh)
    return float(np.sqrt((d[neigh > 0] ** 2).mean()))


def proposal_weight(score: np.ndarray, alpha: float) -> np.ndarray:
    """Non-negative proposal weight from a signed score: exp(alpha*score).

    alpha is the temperature (0 ignores the score).  Max-shifted for stability.
    """
    return np.exp(alpha * (score - score.max()))


def propose_local(xi: int, f: np.ndarray, 
                  rng: np.random.Generator) -> tuple[int, float]:
    """Draw a neighbour a propto f(a); return (a, Z(x) = sum of f over neighbours).

    xi: config id
    f: the sampled target (p or pi) indexed by config id.
    """
    neigh = get_neighbors(xi)
    fa = f[neigh]
    a = int(rng.choice(neigh, p=fa / fa.sum()))
    return a, float(fa.sum())


def run_chain_p(p: np.ndarray, x0: int, n_iter: int,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Locally-balanced MH with target p; returns (visited ids, unit weights)."""
    xi = x0
    ids = np.empty(n_iter, dtype=np.int64)
    for t in range(n_iter):
        a, Z_x = propose_local(xi, p, rng)
        Z_a = float(p[get_neighbors(a)].sum())
        if rng.random() < min(1.0, Z_x / Z_a):
            xi = a
        ids[t] = xi
    return ids, np.ones(n_iter)


def run_chain_dmu(p: np.ndarray, tilt: np.ndarray,
                  x0: int, n_iter: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Locally-balanced MH on tilted target pi; returns (visited ids, weights).

    p: probability of each config
    tilt: exp(alpha*M) by id, the importance factor
    """
    pi = p * tilt # precomputing the importance distribution
    inv_tilt = 1.0 / tilt # precomputing the weight factors
    xi = x0
    ids = np.empty(n_iter, dtype=np.int64)
    weights = np.empty(n_iter)
    for t in range(n_iter):
        a, Z_x = propose_local(xi, pi, rng)
        Z_a = float(pi[get_neighbors(a)].sum()) # normalization for return move
        if rng.random() < min(1.0, Z_x / Z_a):
            xi = a
        ids[t] = xi
        weights[t] = inv_tilt[xi]
    return ids, weights


def run_sampler(chain, n_chains: int, rng: np.random.Generator
                ) -> tuple[np.ndarray, np.ndarray]:
    """Run n_chains chains from random valid starts; concatenate (ids, weights).

    chain(x0, rng) runs one chain and returns (ids, weights).
    """
    out = [chain(int(rng.integers(1, NCONF)), rng) for _ in range(n_chains)]
    ids = np.concatenate([o[0] for o in out])
    weights = np.concatenate([o[1] for o in out])
    return ids, weights


def aggregate_weights(ids: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum sample weights per distinct config id; returns (ids, total weight)."""
    total = np.bincount(ids, weights=weights, minlength=NCONF)
    present = np.nonzero(total)[0]
    return present, total[present]


def samples_frame(present: np.ndarray, weight: np.ndarray, table: np.ndarray,
                  line_names: list[str], method: str) -> pd.DataFrame:
    """One row per distinct config: columns method, avail_id, weight, then mu by line."""
    frame = pd.DataFrame(table[present], columns=line_names)
    frame.insert(0, "weight", weight)
    frame.insert(0, "avail_id", present)
    frame.insert(0, "method", method)
    return frame


def main():
    target = MRF.random(NGEN, seed=42, a_scale=0.8, b_scale=1.0)
    rng = np.random.default_rng(0)
    n_chains, n_iter, alpha_coef = 12, 300, 0.5

    table, line_names = load_mu_table(RESULTS_DIR)
    p = all_unnormalized_p(target)

    alpha = alpha_coef / scale_alpha(table) 
    M = table.sum(axis=1)
    tilt = proposal_weight(M, alpha)
    print(f"alpha = {alpha:.4g}  (coef {alpha_coef})")

    chains = {
        "p-only": lambda x0, rng: run_chain_p(p, x0, n_iter, rng),
        "dmu-biased": lambda x0, rng: run_chain_dmu(p, tilt, x0, n_iter, rng),
    }
    frames = []
    for method, chain in chains.items():
        print(f"{method}: {n_chains} chains x {n_iter} steps ...")
        ids, weights = run_sampler(chain, n_chains, rng)
        present, weight = aggregate_weights(ids, weights)
        ess = weights.sum() ** 2 / (weights ** 2).sum() # effective sample size
        frames.append(samples_frame(present, weight, table, line_names, method))
        print(f"  distinct configs: {len(present)}   weight ESS: {ess:.0f} / {len(weights)}")

    out = pd.concat(frames, ignore_index=True)
    out_path = RESULTS_DIR / "fake_sensitivity_samples.parquet"
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
