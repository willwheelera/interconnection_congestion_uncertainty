"""Subset-simulation sampler for the congestion-target tail over the 2^20 queue space.

The target S(x) = max over lines of load-averaged mu is an expensive N-1 SCDCOPF
solve (via sample_harness.ParallelTarget.evaluate), so the 2^20 = 1.05M queue
configurations cannot be enumerated the way tail_sampling's small cached cases are.
This estimates the high tail P[S >= b] under the nominal law -- each queue generator
comes online independently with its calibrated completion probability p -- using
subset simulation (Au & Beck): factor the rare tail probability into a product of
~p0-sized conditional levels at adaptively rising thresholds, marching chains into
the tail with a modified Metropolis kernel.

Two adaptations to the expensive black-box setting:
  - the objective is driven in batches through the live snapshot-pinned workers,
    which stay alive across every level (one ParallelTarget started once), and
  - a config-id cache means a config revisited across chains/levels is solved once;
    the solver-call budget counts unique solves only.

Usage:
    uv run python src/texas/sample_tail.py --budget 4000
    uv run python src/texas/sample_tail.py --budget 4000 --audit   # certify the top configs
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "texas"))

from sample_harness import ParallelTarget, scalar_target

OUT = REPO_ROOT / "results" / "activsg2000" / "tail_sample.npz"


class BernoulliLaw:
    """Nominal law: each candidate online independently with probability p."""

    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)
        self.logp = np.log(self.p)
        self.log1p = np.log1p(-self.p)

    def draw(self, n, rng):
        return (rng.random((n, len(self.p))) < self.p).astype(np.int8)

    def log_pmf(self, bits):
        return bits @ self.logp + (1 - bits) @ self.log1p


class CachedTarget:
    """Batch S-evaluator over the live workers; caches per config id, counts unique solves."""

    def __init__(self, target, ngen):
        self.target = target
        self.powers = (1 << np.arange(ngen - 1, -1, -1)).astype(np.int64)  # MSB-first
        self.S = {}      # id -> scalar target
        self.mu = {}     # id -> load-averaged mu dict (for tail analysis)
        self.calls = 0

    def __call__(self, bits):
        ids = (bits @ self.powers).astype(np.int64)
        unseen = {}
        for row, i in zip(bits, ids):
            i = int(i)
            if i not in self.S and i not in unseen:
                unseen[i] = row.tolist()
        if unseen:
            avgs = self.target.evaluate(list(unseen.values()))
            for i, avg in zip(unseen, avgs):
                self.mu[i] = avg
                self.S[i] = scalar_target(avg)
            self.calls += len(unseen)
        return np.array([self.S[int(i)] for i in ids])


def n_per_level_for_budget(budget, p0, n_levels):
    """Per-level sample size whose total S-evaluations match the target budget."""
    return max(1, round(budget / (1 + n_levels * (1 - p0))))


def mmh_step(bits, S_cur, b, ev, law, rng, n_flip):
    """One modified-Metropolis sweep of parallel chains held in {S >= b}.

    Flips exactly n_flip distinct components per chain (symmetric proposal), accepts
    the whole candidate by the joint nominal pmf ratio, rejects any leaving the region.
    """
    idx = rng.random(bits.shape).argsort(axis=1)[:, :n_flip]
    flip = np.zeros(bits.shape, dtype=bool)
    np.put_along_axis(flip, idx, True, axis=1)
    cand = np.where(flip, 1 - bits, bits)

    S_cand = ev(cand)
    log_ratio = law.log_pmf(cand) - law.log_pmf(bits)
    keep = (S_cand >= b) & (np.log(rng.random(len(bits))) < log_ratio)
    out = np.where(keep[:, None], cand, bits)
    out_S = np.where(keep, S_cand, S_cur)
    return out, out_S


def populate_level(seed_bits, seed_S, b, chain_len, ev, law, rng, n_flip):
    """Grow chain_len conditioned samples from each seed (seed included), then flatten."""
    n_seed, ngen = seed_bits.shape
    bits_trace = np.empty((chain_len, n_seed, ngen), dtype=seed_bits.dtype)
    S_trace = np.empty((chain_len, n_seed), dtype=seed_S.dtype)
    bits_trace[0], S_trace[0] = seed_bits, seed_S
    for i in range(1, chain_len):
        bits_trace[i], S_trace[i] = mmh_step(
            bits_trace[i - 1], S_trace[i - 1], b, ev, law, rng, n_flip)
    return bits_trace.reshape(chain_len * n_seed, ngen), S_trace.reshape(-1)


def subset_simulation(law, ev, rng, p0, n_per_level, n_levels, flip_schedule):
    """Return (levels, bits_pools, S_pools, pool_probs).

    levels[k] = the k-th conditioning threshold (lowest seeded S at level k), so
    {S >= levels[k]} is exactly the seeded set; pool_probs[k] = estimated nominal
    P[S >= that threshold] (pool_probs[0] = 1). P[S >= b] for b >= levels[k-1] is
    pool_probs[k] * mean(S_pools[k] >= b).
    """
    n_seed = max(1, round(n_per_level * p0))
    chain_len = n_per_level // n_seed
    pool_size = chain_len * n_seed
    bits = law.draw(pool_size, rng)
    S = ev(bits)

    levels, bits_pools, S_pools, pool_probs = [], [bits.copy()], [S.copy()], [1.0]
    for k in range(n_levels):
        n_flip = flip_schedule[min(k, len(flip_schedule) - 1)]
        order = np.argsort(-S)[:n_seed]
        b_k = S[order[-1]]
        pool_probs.append(pool_probs[-1] * n_seed / len(S))
        bits, S = populate_level(bits[order], S[order], b_k, chain_len, ev, law, rng, n_flip)
        levels.append(b_k)
        bits_pools.append(bits.copy())
        S_pools.append(S.copy())
    return np.array(levels), bits_pools, S_pools, np.array(pool_probs)


def weighted_sample(levels, bits_pools, S_pools, pool_probs):
    """Assign each pooled config to its S-band and weight by that band's nominal mass.

    Returns (bits, weights, S): sum_i w_i g(x_i) approximates E_nominal[g] for any g.
    """
    n_levels = len(levels)
    edges = np.concatenate([[-np.inf], levels, [np.inf]])
    out_bits, out_w, out_S = [], [], []
    for k in range(n_levels + 1):
        in_band = (S_pools[k] >= edges[k]) & (S_pools[k] < edges[k + 1])
        cnt = int(in_band.sum())
        if cnt == 0:
            continue
        mass = pool_probs[k] - pool_probs[k + 1] if k < n_levels else pool_probs[n_levels]
        out_bits.append(bits_pools[k][in_band])
        out_w.append(np.full(cnt, mass / cnt))
        out_S.append(S_pools[k][in_band])
    return np.concatenate(out_bits), np.concatenate(out_w), np.concatenate(out_S)


def exceedance(S, weights, thresholds):
    """Subset-simulation estimate of P[S >= b] for each b, from the weighted sample."""
    return np.array([weights[S >= b].sum() for b in thresholds])


def report_top(bits, S, ev, k):
    """Print the k highest-S configs found and each one's driving constraint."""
    order = np.argsort(-S)
    seen, shown = set(), 0
    print(f"\ntop {k} congestion configs discovered:")
    for i in order:
        cid = int(bits[i] @ ev.powers)
        if cid in seen:
            continue
        seen.add(cid)
        n_on = int(bits[i].sum())
        top_con, top_mu = max(ev.mu[cid].items(), key=lambda kv: kv[1], default=("none", 0.0))
        print(f"  S {S[i]:8.2f} $/MWh  {n_on:2d}/{len(ev.powers)} on  "
              f"driver {top_con} ({top_mu:.1f})")
        shown += 1
        if shown >= k:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=4000, help="target unique solver calls")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p0", type=float, default=0.1)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--audit", action="store_true",
                        help="certify the top discovered configs against dropped contingencies")
    args = parser.parse_args()
    flip_schedule = (3, 2, 1)

    target = ParallelTarget()
    ngen = len(target.candidates)
    law = BernoulliLaw([c["p"] for c in target.candidates])
    target.start()
    try:
        ev = CachedTarget(target, ngen)
        rng = np.random.default_rng(args.seed)
        n_per_level = n_per_level_for_budget(args.budget, args.p0, args.levels)

        t0 = time.time()
        levels, bits_pools, S_pools, pool_probs = subset_simulation(
            law, ev, rng, args.p0, n_per_level, args.levels, flip_schedule)
        dt = time.time() - t0

        bits, weights, S = weighted_sample(levels, bits_pools, S_pools, pool_probs)
        thresholds = np.linspace(0.0, S.max(), 60)
        ccdf = exceedance(S, weights, thresholds)

        print(f"nominal law=Bernoulli  ngen={ngen}  target budget={args.budget}  "
              f"unique solves={ev.calls} ({ev.calls / (1 << ngen):.2%} of 2^{ngen})  "
              f"wall {dt:.1f}s ({dt / max(ev.calls, 1):.3f}s/solve)")
        print(f"conditioning thresholds: {np.array2string(levels, precision=1)}")
        for q in (1e-1, 1e-2, 1e-3, 1e-4):
            j = int(np.searchsorted(-ccdf, -q))
            j = min(j, len(thresholds) - 1)
            print(f"  P[S >= {thresholds[j]:7.1f}] ~ {ccdf[j]:.2e}")
        report_top(bits, S, ev, 10)

        np.savez(OUT, thresholds=thresholds, ccdf=ccdf, levels=levels,
                 pool_probs=pool_probs, sample_bits=bits, sample_w=weights, sample_S=S,
                 calls=ev.calls, ngen=ngen)
        print(f"\nsaved -> {OUT}")

        if args.audit:
            order = np.argsort(-S)[:20]
            top_bits = bits[order]
            t0 = time.time()
            audits = target.audit(top_bits.tolist())
            viol = sorted({cn for a in audits for cn in a})
            print(f"\naudit ({time.time() - t0:.1f}s): {len(viol)} dropped contingency(ies) "
                  f"violated by the top configs")
            if viol:
                print(f"  promote and re-run: {viol}")
            else:
                print("  the discovered tail is provably exact on the screened case")
    finally:
        target.close()


if __name__ == "__main__":
    main()
