"""Subset simulation for the high tail of the congestion target S(x) = max_b |mu_b|.

Standard direct sampling from the nominal law p spends almost its whole budget on
typical (low-S) configurations and almost never sees S >= b for a high threshold b,
so its tail estimate P[S >= b] is zero or wildly noisy.  Subset simulation (Au & Beck)
factors that rare probability into a product of ~p0-sized conditional probabilities at
adaptively rising thresholds b_1 < b_2 < ... and grows samples conditioned on each
level with a modified Metropolis kernel.  It marches into the tail under the same
solver budget and returns the whole exceedance curve, unbiased.

This file holds two of the compared samplers: `direct` (the baseline) and
`subset_sim`.  The shared problem (S, nominal law, Evaluator, ground truth) lives in
problem_setup.

Usage:
    uv run python -m src.sampling.methods.subset_simulation
"""

import numpy as np

from src.cases.registry import get as get_case
from src.sampling.distributions import case_dist
from src.sampling.problem import Problem


def n_per_level_for_budget(budget, p0=0.1, n_levels=6):
    """Per-level sample size whose total S-evaluations match the target budget.

    Cost is n_per_level at level 0 plus n_levels * n_per_level * (1 - p0) conditioned
    evals across the levels, so invert that sum for n_per_level.
    """
    return max(1, round(budget / (1 + n_levels * (1 - p0))))


def mmh_step(problem, bits, S_cur, b, evaluator, rng, n_flip):
    """One Metropolis sweep of parallel chains held in the region {S >= b}.

    bits: (n_chains, ngen) int8, one row per chain; S_cur: (n_chains,) their S.
    Exactly n_flip distinct components per chain are flipped (a fixed-size random scan,
    symmetric, so every step proposes a real move); the whole candidate is accepted by
    the nominal pmf ratio p(cand)/p(bits) and rejected if it leaves the region.  Using
    the joint pmf ratio (not per-component marginals) keeps the kernel correct for a
    general correlated nominal law.  Returns the advanced (bits, S) at the same shapes.
    """
    idx = rng.random(bits.shape).argsort(axis=1)[:, :n_flip]  # n_flip distinct cols/chain
    flip = np.zeros(bits.shape, dtype=bool)
    np.put_along_axis(flip, idx, True, axis=1)
    cand = np.where(flip, 1 - bits, bits)

    S_cand = evaluator(problem.bits_to_id(cand))
    log_ratio = problem.log_nominal_pmf(cand) - problem.log_nominal_pmf(bits)
    in_region = S_cand >= b                                  # region selector
    keep = in_region & (np.log(rng.random(len(bits))) < log_ratio)
    out = np.where(keep[:, None], cand, bits)
    out_S = np.where(keep, S_cand, S_cur)
    return out, out_S


def populate_level(problem, seed_bits, seed_S, b, chain_len, evaluator, rng, n_flip):
    """Grow chain_len conditioned samples from each seed (seed included).

    seed_bits: (n_seed, ngen) int8 batch of chain starts; seed_S: (n_seed,).
    Runs the n_seed seeds as parallel chains for chain_len steps and flattens the
    snapshots: returns (chain_len*n_seed, ngen) bits and (chain_len*n_seed,) S.
    """
    n_seed, ngen = seed_bits.shape
    bits_trace = np.empty((chain_len, n_seed, ngen), dtype=seed_bits.dtype)
    S_trace = np.empty((chain_len, n_seed), dtype=seed_S.dtype)
    bits_trace[0], S_trace[0] = seed_bits, seed_S
    for i in range(1, chain_len):
        bits_trace[i], S_trace[i] = mmh_step(
            problem, bits_trace[i - 1], S_trace[i - 1], b, evaluator, rng, n_flip)
    return bits_trace.reshape(chain_len * n_seed, ngen), S_trace.reshape(-1)


def subset_simulation(problem, evaluator, rng, p0=0.1, n_per_level=200, n_levels=6, flip_schedule=(3, 2, 1)):
    """Returns (levels, S_pools, id_pools, pool_probs): per-level thresholds b_k (each the
    lowest selected seed's S, so {S >= b_k} is exactly the seeded set), the S-pools, the
    matching config-id pools (pool 0 is the nominal sample), and pool_probs[k] = estimated
    nominal P[S >= conditioning threshold of pool k] (pool_probs[0] = 1).

    P[S >= b] for b >= b_{k-1} is estimated as pool_probs[k] * mean(pool_k >= b).  Each
    level's conditional factor is the realized seeded fraction n_seed / pool_size, which
    equals the nominal p0 only when it divides evenly; using the realized fraction keeps
    the weights consistent under the budget-driven rounding and per-level pool shrinkage.
    """
    n_seed = max(1, round(n_per_level * p0))
    chain_len = n_per_level // n_seed
    pool_size = chain_len * n_seed                    # uniform pool size across all levels
    bits = problem.draw_nominal(pool_size, rng)
    S = evaluator(problem.bits_to_id(bits))

    levels, S_pools, id_pools = [], [S.copy()], [problem.bits_to_id(bits)]
    pool_probs = [1.0]
    for k in range(n_levels):
        n_flip = flip_schedule[min(k, len(flip_schedule) - 1)]   # ramp down as region tightens
        order = np.argsort(-S)[:n_seed]
        b_k = S[order[-1]]                              # lowest selected S = conditioning cutoff
        pool_probs.append(pool_probs[-1] * n_seed / len(S))
        bits, S = populate_level(problem, bits[order], S[order], b_k, chain_len, evaluator, rng, n_flip)
        levels.append(b_k)
        S_pools.append(S.copy())
        id_pools.append(problem.bits_to_id(bits))
    return np.array(levels), S_pools, id_pools, np.array(pool_probs)


def ss_exceedance(problem, evaluator, rng, thresholds, budget, p0=0.1, n_levels=6, flip_schedule=(3, 2, 1)):
    """Subset-simulation CCDF at the target solver-call budget: size the per-level
    sample from the budget, then use the highest level whose b_k <= b."""
    n_per_level = n_per_level_for_budget(budget, p0, n_levels)
    levels, pools, _, pool_probs = subset_simulation(problem, evaluator, rng, p0, n_per_level, n_levels, flip_schedule)
    out = np.empty(len(thresholds))
    for j, b in enumerate(thresholds):
        k = int(np.searchsorted(levels, b, side="right"))   # pools index (0 = nominal)
        out[j] = pool_probs[k] * np.mean(pools[k] >= b)
    return out


def sample(problem, ev, rng, budget, p0=0.1, n_levels=6, flip_schedule=(3, 2, 1)):
    """Subset-simulation weighted sample (ids, weights) at the target solver budget.

    Each pooled config is assigned to its S-band [b_k, b_{k+1}) and weighted by that
    band's nominal mass over the band count.  Within a band the importance weight p/q
    is constant (the kernel's stationary law is the nominal restricted to the band),
    so sum_i w_i g(x_i) ~ E_p[g] for any g.
    """
    n_per_level = n_per_level_for_budget(budget, p0, n_levels)
    levels, S_pools, id_pools, pool_probs = subset_simulation(problem, ev, rng, p0, n_per_level, n_levels, flip_schedule)
    edges = np.concatenate([[-np.inf], levels, [np.inf]])
    ids, weights = [], []
    for k in range(n_levels + 1):
        in_band = (S_pools[k] >= edges[k]) & (S_pools[k] < edges[k + 1])
        cnt = int(in_band.sum())
        if cnt == 0:
            continue
        mass = pool_probs[k] - pool_probs[k + 1] if k < n_levels else pool_probs[n_levels]
        ids.append(id_pools[k][in_band])
        weights.append(np.full(cnt, mass / cnt))
    return np.concatenate(ids), np.concatenate(weights)


def run_replicates(problem, thresholds, budget, n_rep, p0=0.1, n_levels=6, flip_schedule=(3, 2, 1), seed0=0):
    """Run both methods n_rep times at the target solver-call budget; collect curves."""
    ss, direct, budgets = [], [], []
    for r in range(n_rep):
        evaluator = problem.evaluator()
        rng = np.random.default_rng(seed0 + r)
        ss.append(ss_exceedance(problem, evaluator, rng, thresholds, budget, p0, n_levels, flip_schedule))
        budgets.append(evaluator.calls)

        direct.append(problem.direct_exceedance(rng, evaluator.calls, thresholds))
    return np.array(ss), np.array(direct), int(np.mean(budgets))


def report(name, est, truth, idx):
    """Print mean estimate, relative bias, and relative RMSE at the rows in idx."""
    mean = est.mean(axis=0)
    rmse = np.sqrt(((est - truth[None, :]) ** 2).mean(axis=0))
    zero = (est == 0).mean(axis=0)
    print(f"\n{name}")
    print(f"  {'P_true':>10} {'mean_est':>10} {'rel_rmse':>9} {'frac_zero':>9}")
    for j in idx:
        print(f"  {truth[j]:10.2e} {mean[j]:10.2e} {rmse[j] / truth[j]:9.2f} {zero[j]:9.2f}")


def main():
    case = get_case("case14_gen_per_bus")
    problem = Problem(case.results_dir, dist=case_dist(case, "bernoulli"))
    grid_probs = np.logspace(-1, -6, 40)                 # dense tail grid for the curve
    thresholds = problem.thresholds_for_probs(grid_probs)
    truth = problem.true_exceedance(thresholds)
    target_idx = [np.argmin(np.abs(grid_probs - q)) for q in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)]

    budget, n_rep = 1280, 60
    p0, n_levels, flip_schedule = 0.1, 6, (3, 2, 1)     # subset-simulation hyperparameters
    ss, direct, achieved = run_replicates(problem, thresholds, budget, n_rep, p0, n_levels, flip_schedule)
    print(f"nominal law={type(problem.dist).__name__}  target budget={budget}  "
          f"achieved={achieved} solves ({achieved / problem.nconf:.1%} of {problem.nconf} configs)"
          f"  replicates={n_rep}")
    report("direct nominal sampling (baseline)", direct, truth, target_idx)
    report("subset simulation", ss, truth, target_idx)

    out = problem.results_dir / "subset_simulation_compare.npz"
    np.savez(out, thresholds=thresholds, truth=truth, ss=ss, direct=direct,
             budget=achieved, nconf=problem.nconf)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
