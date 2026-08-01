"""MCMC samplers for the MRF availability target, compared by OPF cost.

In production the bottleneck is per-config OPF solves: each newly visited x
costs one optimization, repeats are free (cached).  We compare samplers on
the same OPF budget.

Target: MRF on the 14-bit availability vector,
    log p(x) = a^T x + sum_{i<j} b_ij x_i x_j - log Z.

Samplers
--------
1. ``baseline_mh`` -- vanilla single-bit-flip Metropolis-Hastings.  1 OPF
   lookup per step.

2. ``informed_only`` -- Zanella's locally balanced proposal,
       q(y_k | x) proportional to sqrt(p(y_k) / p(x)),
   with the standard MH accept/reject and chain estimator.  Proposal
   weights cost only MRF evaluations (free); still 1 OPF lookup per step.

3. ``informed_sensitivity`` -- locally balanced proposal multiplied by the
   barrier-regularised gradient magnitude in each flip direction,
       q(y_k | x) proportional to sqrt(p(y_k) / p(x)) * s_k(x)^alpha,
   where s_k(x) = |d(sum_l mu_l) / dP_{bus(k)}| is evaluated at the OPF
   solution of x.  The score is large precisely where flipping bit k is
   close to triggering an active-set change -- the regime where vanilla and
   sqrt-Zanella proposals are blind.  1 OPF lookup per step; the gradient
   is a byproduct of the OPF solve.

Reference: exact enumeration over the cached 16383 states.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.availability import avail_to_id, id_to_avail


REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "case14_gen_per_bus"

NGEN = 14
INVALID_ID = 0


@dataclass
class MRF:
    """log p(x) = a^T x + x^T B x (B strictly upper triangular, unnormalized)."""

    a: np.ndarray
    B: np.ndarray

    @classmethod
    def random(cls, ngen: int, seed: int = 0, a_scale: float = 0.8, b_scale: float = 0.5,
        a_shift: float=-0.5, b_shift: float=-0.5):
        rng = np.random.default_rng(seed)
        a = rng.normal(scale=a_scale, size=ngen) + a_shift
        B = np.triu(rng.normal(scale=b_scale, size=(ngen, ngen)) + b_shift, 1)
        return cls(a=a, B=B)

    def log_p(self, X: np.ndarray) -> np.ndarray:
        return X @ self.a + np.einsum("...i,ij,...j->...", X, self.B, X)

    def log_p_flip(self, x: np.ndarray) -> np.ndarray:
        """log p(y_k) - log p(x) for each single-bit-flip neighbour y_k of x."""
        return (1 - 2 * x) * (self.a + (self.B + self.B.T) @ x)


def feasible_log_p_flip(target: MRF, x: np.ndarray) -> np.ndarray:
    """log_p_flip with forbidden moves (would land at the all-zero state) set to -inf."""
    out = target.log_p_flip(x).astype(np.float64)
    if x.sum() == 1:
        out[x.astype(bool)] = -np.inf
    return out


def informed_proposal(log_diff: np.ndarray) -> tuple[np.ndarray, float]:
    """Zanella sqrt proposal q[k] proportional to exp(0.5 log_diff[k]).

    Returns (q, log Z).  When all neighbours are forbidden the chain is stuck.
    """
    half = 0.5 * log_diff
    m = half.max()
    if not np.isfinite(m):
        raise RuntimeError("All single-bit-flip proposals forbidden.")
    w = np.exp(half - m)
    Z = w.sum()
    return w / Z, m + np.log(Z)


@dataclass
class Particle:
    """A weighted set of configurations representing one MCMC step.

    Standard samplers return particles of size 1 (the visited state with
    weight 1); importance-weighted samplers carry their IS weight here.
    """

    ids: np.ndarray
    weights: np.ndarray


def baseline_mh(target: MRF, x0: np.ndarray, n_iter: int, rng: np.random.Generator) -> list[Particle]:
    """Single-bit-flip random-walk Metropolis."""
    x = x0.copy()
    out: list[Particle] = []
    for _ in range(n_iter):
        k = rng.integers(NGEN)
        log_ratio = feasible_log_p_flip(target, x)[k]
        if np.log(rng.random()) < log_ratio:
            x = x.copy()
            x[k] = 1 - x[k]
        out.append(Particle(np.array([avail_to_id(x)]), np.array([1.0])))
    return out


def informed_only(target: MRF, x0: np.ndarray, n_iter: int,
                  rng: np.random.Generator) -> list[Particle]:
    """Locally balanced (Zanella) proposal, standard chain estimator (no RB).

    Per step we evaluate p at the 14 neighbours of x (cheap, just the MRF) to
    build the proposal and at the 14 neighbours of the candidate y for the
    Hastings ratio.  Only one OPF lookup is needed per step (the visited
    state).  This is the "fair-cost" variant: same OPF budget per step as
    vanilla MH but much faster mixing.
    """
    x = x0.copy()
    log_diff_x = feasible_log_p_flip(target, x)
    q_x, logZ_x = informed_proposal(log_diff_x)
    out: list[Particle] = []

    for _ in range(n_iter):
        k_sel = rng.choice(NGEN, p=q_x)
        y = x.copy()
        y[k_sel] = 1 - y[k_sel]
        log_diff_y = feasible_log_p_flip(target, y)
        _, logZ_y = informed_proposal(log_diff_y)
        if np.log(rng.random()) < min(0.0, logZ_x - logZ_y):
            x = y
            log_diff_x = log_diff_y
            q_x, logZ_x = informed_proposal(log_diff_x)
        out.append(Particle(np.array([avail_to_id(x)]), np.array([1.0])))
    return out


def sensitivity_proposal(log_diff: np.ndarray, log_score: np.ndarray
                         ) -> tuple[np.ndarray, float]:
    """Locally balanced sqrt(p_ratio) proposal times a per-direction score.

    log_diff[k] = log p(y_k) - log p(x).
    log_score[k] = alpha * log s_k(x), the (scaled-log) sensitivity weight.

    Forbidden flips (log_diff = -inf) propagate to a zero proposal weight.
    Returns (q, logZ) with logZ = log sum_k exp(0.5 log_diff[k] + log_score[k]).
    """
    combined = 0.5 * log_diff + log_score
    m = combined.max()
    if not np.isfinite(m):
        raise RuntimeError("All single-bit-flip proposals forbidden.")
    w = np.exp(combined - m)
    Z = w.sum()
    return w / Z, m + np.log(Z)


def informed_sensitivity(target: MRF, x0: np.ndarray, n_iter: int,
                         rng: np.random.Generator,
                         score_by_id: dict[int, np.ndarray],
                         alpha: float = 1.0,
                         floor: float = 1e-6) -> list[Particle]:
    """Locally balanced Zanella proposal weighted by per-direction sensitivity.

    Proposal at x:
        q(y_k | x) propto sqrt(p(y_k)/p(x)) * s_k(x)^alpha
    where s_k(x) = |d(sum_l mu_l) / dP_{bus(k)}| from the cached barrier
    sensitivity at x.  Same OPF cost as ``informed_only`` (1 lookup per
    step; the gradient comes with the OPF solve).  Acceptance is the
    standard MH ratio against the MRF target -- the target ratio cancels
    with the sqrt in q, leaving

        log A = log s_{k*}(y_{k*})^alpha + logZ(x)
                - log s_{k*}(x)^alpha    - logZ(y_{k*}).
    """
    x = x0.copy()
    log_diff_x = feasible_log_p_flip(target, x)
    log_s_x = alpha * np.log(np.maximum(score_by_id[avail_to_id(x)], floor))
    q_x, logZ_x = sensitivity_proposal(log_diff_x, log_s_x)
    out: list[Particle] = []

    for _ in range(n_iter):
        k_sel = rng.choice(NGEN, p=q_x)
        y = x.copy()
        y[k_sel] = 1 - y[k_sel]
        log_diff_y = feasible_log_p_flip(target, y)
        log_s_y = alpha * np.log(np.maximum(score_by_id[avail_to_id(y)], floor))
        _, logZ_y = sensitivity_proposal(log_diff_y, log_s_y)
        log_ratio = log_s_y[k_sel] - log_s_x[k_sel] + logZ_x - logZ_y
        if np.log(rng.random()) < min(0.0, log_ratio):
            x = y
            log_diff_x = log_diff_y
            log_s_x = log_s_y
            q_x, logZ_x = sensitivity_proposal(log_diff_x, log_s_x)
        out.append(Particle(np.array([avail_to_id(x)]), np.array([1.0])))
    return out


def load_score_cache(case_dir: Path = RESULTS_DIR) -> dict[int, np.ndarray]:
    """Load sensitivity_cache.parquet into a dict avail_id -> 14-vector."""
    df = pd.read_parquet(case_dir / "sensitivity_cache.parquet")
    return {int(i): row for i, row in zip(df.index, df.values)}


def load_score_scalar(case_dir: Path = RESULTS_DIR, floor: float = 1.0
                      ) -> dict[int, float]:
    """Per-config L1 norm of the sensitivity vector, floored away from zero.

    s(x) = max(floor, sum_k |d(sum_l mu_l)/dP_{bus(k)}| (x)).  The floor
    keeps the IS weight 1/s bounded above on configs deep inside a single
    active region (where every per-direction sensitivity is small).
    """
    df = pd.read_parquet(case_dir / "sensitivity_cache.parquet")
    s = df.values.sum(axis=1)
    s = np.maximum(s, floor)
    return dict(zip(df.index.astype(int).tolist(), s.tolist()))


def informed_iw(target: MRF, x0: np.ndarray, n_iter: int,
                rng: np.random.Generator,
                score_scalar_by_id: dict[int, float],
                alpha: float = 1.0) -> list[Particle]:
    """Importance-weighted MH targeting q(x) ∝ p(x) s(x)^alpha.

    Proposal: locally balanced Zanella against the *original* target p,
        q_prop(y_k|x) propto sqrt(p(y_k)/p(x)).
    Acceptance for q-target (the sqrt-p factor cancels):
        log A = alpha*(log s(y) - log s(x)) + logZ_p(x) - logZ_p(y).

    Per step: 1 OPF query at the proposed y (read of s(y) from cache).
    Per-step particle weight is the IS weight 1/s(x_n)^alpha so the
    self-normalized chain estimator targets E_p[f].  s(x) is large near
    active-set boundaries (the barrier eps makes ∂μ^eps/∂P blow up there),
    so the chain concentrates samples where mu is most variable.
    """
    x = x0.copy()
    s_x = float(score_scalar_by_id[avail_to_id(x)])
    log_s_x = alpha * np.log(s_x)
    log_diff_x = feasible_log_p_flip(target, x)
    q_x, logZ_x = informed_proposal(log_diff_x)
    out: list[Particle] = []

    for _ in range(n_iter):
        k_sel = rng.choice(NGEN, p=q_x)
        y = x.copy()
        y[k_sel] = 1 - y[k_sel]
        s_y = float(score_scalar_by_id[avail_to_id(y)])
        log_s_y = alpha * np.log(s_y)
        log_diff_y = feasible_log_p_flip(target, y)
        _, logZ_y = informed_proposal(log_diff_y)
        log_A = (log_s_y - log_s_x) + (logZ_x - logZ_y)
        if np.log(rng.random()) < min(0.0, log_A):
            x = y
            s_x, log_s_x = s_y, log_s_y
            log_diff_x = log_diff_y
            q_x, logZ_x = informed_proposal(log_diff_x)
        w = np.exp(-log_s_x)
        out.append(Particle(np.array([avail_to_id(x)]), np.array([w])))
    return out


def true_distribution(target: MRF, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Exact enumeration over the cached configurations: returns (p, F)."""
    ids = np.array(df.index)
    X = np.stack([id_to_avail(int(i), NGEN) for i in ids])
    log_p = target.log_p(X)
    log_p -= log_p.max()
    p = np.exp(log_p)
    return p / p.sum(), df.values.astype(np.float64)


def run_chains(sampler: Callable, target: MRF, n_iter: int, n_chains: int,
               rng: np.random.Generator, **sampler_kwargs) -> list[list[Particle]]:
    """Run independent chains and return a list of length n_chains."""
    chains = []
    for _ in range(n_chains):
        x0 = (rng.random(NGEN) < 0.5).astype(np.int8)
        if x0.sum() == 0:
            x0[0] = 1
        chains.append(sampler(target, x0, n_iter, rng, **sampler_kwargs))
    return chains


def chain_running_mean(chain: list[Particle], scalar_by_id: dict[int, float]) -> np.ndarray:
    """Self-normalized running mean: sum_n w_n f(x_n) / sum_n w_n.

    Equivalent to the cumulative arithmetic mean when particle weights sum to
    one per step (baseline_mh, informed_only).  For sensitivity-IS samplers
    whose particle weight is the importance weight
    1/s(x)^alpha, this is the standard self-normalized estimator.
    """
    wf = np.array([
        sum(w * scalar_by_id.get(int(i), 0.0) for i, w in zip(p.ids, p.weights))
        for p in chain
    ])
    w_total = np.array([float(p.weights.sum()) for p in chain])
    return np.cumsum(wf) / np.cumsum(w_total)


def chain_distinct_count(chain: list[Particle]) -> np.ndarray:
    """Per-step running count of distinct cached config-ids seen so far.

    Models the OPF query budget in production: each newly visited config
    requires one OPF solve, repeats are free.  The all-zero state is
    load-infeasible and not counted.
    """
    seen: set[int] = set()
    counts = np.empty(len(chain), dtype=np.int64)
    for t, p in enumerate(chain):
        for i, w in zip(p.ids, p.weights):
            if w > 0 and int(i) != INVALID_ID:
                seen.add(int(i))
        counts[t] = len(seen)
    return counts


def aggregate_running_mean(chains: list[list[Particle]], scalar_by_id: dict[int, float]) -> np.ndarray:
    return np.stack([chain_running_mean(ch, scalar_by_id) for ch in chains])


def variance_vs_budget(runs: np.ndarray, distinct: np.ndarray, true_mean: float,
                       budgets: np.ndarray) -> np.ndarray:
    """For each OPF budget, compute the chain-to-chain std of the running mean.

    runs: (n_chains, n_iter) running-mean trace.
    distinct: (n_chains, n_iter) running distinct-config count.
    Returns sd[i] = std over chains of running estimate at the *first* step
    where the chain has visited >= budgets[i] distinct configs.  Chains that
    never reach a given budget are excluded from that point's std.
    """
    sd = np.empty(len(budgets))
    for j, b in enumerate(budgets):
        vals = []
        for c in range(runs.shape[0]):
            t = np.searchsorted(distinct[c], b, side="left")
            if t < runs.shape[1]:
                vals.append(runs[c, t])
        sd[j] = np.std(vals, ddof=1) if len(vals) >= 2 else np.nan
    return sd


SAMPLER_COLORS = {
    "vanilla MH": "tab:red",
    "informed-only": "tab:green",
    "informed-IW": "tab:purple",
}


def setup_target_truth(df: pd.DataFrame, target: MRF):
    """Exact enumeration: returns p_true, F_true, mu/lmp column indices, the
    sum_b mu_b lookup, and (mean, sd) of f(x) = sum_b mu_b under p_true."""
    p_true, F_true = true_distribution(target, df)
    mu_cols = [i for i, c in enumerate(df.columns) if c[0] == "mu"]
    lmp_cols = [i for i, c in enumerate(df.columns) if c[0] == "lmp"]
    f_mu_sum = F_true[:, mu_cols].sum(axis=1)
    mu_sum_by_id = dict(zip(df.index.astype(int).tolist(), f_mu_sum.tolist()))
    true_mean = float((p_true * f_mu_sum).sum())
    true_std = float(np.sqrt((p_true * f_mu_sum ** 2).sum() - true_mean ** 2))
    return p_true, F_true, mu_cols, lmp_cols, mu_sum_by_id, true_mean, true_std


def run_all_samplers(target: MRF, n_iter: int, n_chains: int,
                     rng: np.random.Generator,
                     score_scalar: dict[int, float]) -> dict[str, list[list[Particle]]]:
    """Run the three samplers under comparison and return {name: chains}."""
    print(f"running vanilla MH for {n_iter} steps (1 OPF / step)...")
    base = run_chains(baseline_mh, target, n_iter, n_chains, rng)
    print(f"running informed-only for {n_iter} steps (1 OPF / step)...")
    only = run_chains(informed_only, target, n_iter, n_chains, rng)
    print(f"running informed-IW alpha=0.25 for {n_iter} steps (1 OPF / step)...")
    iw = run_chains(informed_iw, target, n_iter, n_chains, rng,
                    score_scalar_by_id=score_scalar, alpha=0.25)
    return {"vanilla MH": base, "informed-only": only, "informed-IW": iw}


def summarize_samplers(samplers, runs, distinct):
    """Print per-sampler distinct-config count and final running-mean stats."""
    for name, ch in samplers.items():
        final_dist = distinct[name][:, -1].mean()
        est = runs[name][:, -1]
        print(f"  {name:14s}  steps={len(ch[0]):5d}  distinct/chain={final_dist:7.0f}  "
              f"est={est.mean():.3f}  sd={est.std(ddof=1):.4f}")


def sd_vs_budget_curves(samplers, runs, distinct, true_mean):
    """Common-budget grid and chain-to-chain sd of the running estimate."""
    min_distinct = min(distinct[k][:, -1].min() for k in samplers)
    budgets = np.unique(np.linspace(20, min_distinct, 40).astype(int))
    curves = {k: variance_vs_budget(runs[k], distinct[k], true_mean, budgets)
              for k in samplers}
    return budgets, curves


def plot_sd_curve(ax, budgets, sd_curves):
    for name, sd in sd_curves.items():
        ax.plot(budgets, sd, color=SAMPLER_COLORS[name], lw=1.6, label=name)
    ax.set_xlabel("distinct configs evaluated (OPF budget)")
    ax.set_ylabel("chain-to-chain sd of running estimate")
    ax.set_title("Estimator sd vs. OPF cost")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")


def plot_running_mean(ax, runs, true_mean):
    for name, R in runs.items():
        iters = np.arange(1, R.shape[1] + 1)
        for c in range(R.shape[0]):
            ax.plot(iters, R[c], color=SAMPLER_COLORS[name], alpha=0.2, lw=0.6)
        ax.plot(iters, R.mean(0), color=SAMPLER_COLORS[name], lw=2, label=name)
    ax.axhline(true_mean, color="k", ls="--", lw=1.2, label=f"truth = {true_mean:.2f}")
    ax.set_xlabel("MCMC step")
    ax.set_ylabel(r"running estimate of $E\,[\sum_b \mu_b]$")
    ax.set_xscale("log")
    ax.set_title("Running-mean trace")
    ax.legend(fontsize=9)


def save_trace_figure(runs, budgets, sd_curves, true_mean, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_sd_curve(axes[0], budgets, sd_curves)
    plot_running_mean(axes[1], runs, true_mean)
    fig.tight_layout()
    fig.savefig(out_path)


def plot_exact_marginal(ax, vals, p_true, lo, xlabel, title):
    """Plot the exact marginal of `vals` (one column of F_true) under p_true."""
    low = vals.min() if lo is None else lo
    bins = np.linspace(low, max(vals.max(), low + 1e-6), 40)
    density, _ = np.histogram(vals, bins=bins, weights=p_true, density=True)
    centers = 0.5 * (bins[1:] + bins[:-1])
    ax.step(centers, density, where="mid", color="k", lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)


def save_marginal_figure(F_true, p_true, mu_cols, lmp_cols, line, bus, out_path):
    """Exact per-line mu and per-bus lambda marginals under p_true."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_exact_marginal(
        axes[0], F_true[:, mu_cols[line]], p_true, lo=0.0,
        xlabel=fr"$\mu$ on line {line + 1}",
        title=f"Exact mu marginal (line {line + 1})")
    plot_exact_marginal(
        axes[1], F_true[:, lmp_cols[bus]], p_true, lo=None,
        xlabel=fr"$\lambda$ on bus {bus + 1}",
        title=f"Exact lambda marginal (bus {bus + 1})")
    fig.tight_layout()
    fig.savefig(out_path)


def main():
    df = pd.read_parquet(RESULTS_DIR / "dual_cache.parquet").sort_index()
    target = MRF.random(NGEN, seed=42, a_scale=0.8, b_scale=1.0)
    rng = np.random.default_rng(0)
    n_chains, n_iter = 24, 2000
    line, bus = 12, 13

    p_true, F_true, mu_cols, lmp_cols, mu_sum_by_id, true_mean, true_std = \
        setup_target_truth(df, target)
    print(f"target MRF: support={len(p_true)}, eff. support (p>1e-5)={(p_true > 1e-5).sum()}")
    print(f"truth: E[sum_b mu_b] = {true_mean:.4f},  sd = {true_std:.4f}\n")

    print("loading sensitivity cache...")
    score_scalar = load_score_scalar(RESULTS_DIR, floor=1.0)
    samplers = run_all_samplers(target, n_iter, n_chains, rng, score_scalar)

    runs = {k: aggregate_running_mean(v, mu_sum_by_id) for k, v in samplers.items()}
    distinct = {k: np.stack([chain_distinct_count(ch) for ch in v]) for k, v in samplers.items()}
    print()
    summarize_samplers(samplers, runs, distinct)

    budgets, sd_curves = sd_vs_budget_curves(samplers, runs, distinct, true_mean)
    save_trace_figure(runs, budgets, sd_curves, true_mean,
                      RESULTS_DIR / "informed_mcmc_trace.pdf")

    save_marginal_figure(F_true, p_true, mu_cols, lmp_cols, line, bus,
                         RESULTS_DIR / "informed_mcmc_distribution.pdf")
    print(f"figures saved under {RESULTS_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
