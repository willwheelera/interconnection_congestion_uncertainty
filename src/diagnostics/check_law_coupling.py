"""How much of the mu distribution the nominal law's *correlations* control.

Every law in `sampling.distributions` is built at the same per-candidate marginals, so
anything it changes about mu is attributable to coupling alone.  This prints, per law:

  marginal error   max |P[x_i = 1] - p_i|, the check that the laws really do share marginals
  corr             mean and max pairwise correlation of the availability bits
  TV               total variation distance to the Bernoulli law over configurations
  P[S > t]         exact nominal exceedance of the target at each threshold
  per line         exact P[|mu_l| > c] at each element's own Bernoulli reference level

Everything is exact: the case is fully enumerated, so each law is a weight vector over
all 2^ngen configurations and no sampling enters.

Usage:
    uv run python -m src.diagnostics.check_law_coupling --case activsg2000_original_v12
    uv run python -m src.diagnostics.check_law_coupling --case activsg2000_original_v12 \
        --dists bernoulli,mrf_cobuild,mrf_substitute --at 1e-4
"""

import argparse

import numpy as np

from src.cases.registry import CASES, add_case_arg
from src.sampling.problem import TargetProblem
from src.sampling.distributions import case_dist, DIST_NAMES, COUPLED_NAMES


def law_weights(case, dists):
    """{law name: exact weight vector} plus the problem the weights are read against."""
    problems = {d: TargetProblem(case.target_cache, dist=case_dist(case, d)) for d in dists}
    return {d: pr.w for d, pr in problems.items()}, next(iter(problems.values()))


def bit_moments(w, bits):
    """Marginals and the pairwise correlation matrix of the availability bits under w."""
    m = w @ bits
    cov = (bits * w[:, None]).T @ bits - np.outer(m, m)
    sd = np.sqrt(np.diag(cov))
    return m, cov / np.outer(sd, sd)


def law_summary(w, w_bern, bits, p, S, thresholds):
    """One law's row: marginal error, correlation, distance to Bernoulli, S exceedance."""
    m, corr = bit_moments(w, bits)
    off = corr[~np.eye(len(p), dtype=bool)]
    return dict(marg_err=float(np.abs(m - p).max()),
                corr_mean=float(np.abs(off).mean()), corr_max=float(np.abs(off).max()),
                tv=0.5 * float(np.abs(w - w_bern).sum()),
                mean_S=float(w @ S),
                exceed=[float(w[S > t].sum()) for t in thresholds])


def print_target_table(rows, thresholds, target):
    """The per-law target table, tail probabilities as a ratio to the Bernoulli law."""
    head = "  ".join(f"P[{target}>{t:g}]" for t in thresholds)
    print(f"\n{'law':16s} {'marg err':>9s} {'|corr|':>7s} {'max':>6s} {'TV':>6s} "
          f"{'E[S]':>8s}  {head}")
    base = rows["bernoulli"]["exceed"]
    for name, r in rows.items():
        cells = "  ".join(f"{e:.2e}({e / b:5.2f}x)" if b else f"{e:.2e}( --- )"
                          for e, b in zip(r["exceed"], base))
        print(f"{name:16s} {r['marg_err']:9.1e} {r['corr_mean']:7.3f} {r['corr_max']:6.3f} "
              f"{r['tv']:6.3f} {r['mean_S']:8.1f}  {cells}")


def level_at(v, w, q):
    """The |mu_l| level exceeded with probability q under weights w (exact enumeration)."""
    order = np.argsort(-v)
    surv = np.cumsum(w[order])
    return float(v[order][min(np.searchsorted(surv, q), len(order) - 1)])


def print_line_table(weights, mu, names, at, n_lines):
    """Per-line exceedance at each line's own Bernoulli-`at` level, and mean |mu_l|.

    Every line gets its own threshold -- the level it exceeds with probability `at` under
    the Bernoulli law -- so the columns read as the factor by which each law moves that
    line's tail, whatever scale the line's mu sits on."""
    w_bern = weights["bernoulli"]
    order = np.argsort(-np.array([float(w_bern @ mu[:, l]) for l in range(mu.shape[1])]))
    laws = list(weights)
    print(f"\nper line at its own Bernoulli {at:g} level: exact P[|mu_l| > c]  "
          f"(mean |mu_l| in parentheses)")
    print(f"{'element':16s} {'c':>8s} " + " ".join(f"{d:>21s}" for d in laws))
    for l in order[:n_lines]:
        v = mu[:, l]
        c = level_at(v, w_bern, at)
        cells = [f"{float(w[v > c].sum()):8.2e} ({float(w @ v):8.1f})" for w in weights.values()]
        print(f"{names[l]:16s} {c:8.1f} " + " ".join(cells))


def main():
    parser = argparse.ArgumentParser()
    add_case_arg(parser, default="activsg2000_original_v12")
    parser.add_argument("--dists", default=",".join(DIST_NAMES + COUPLED_NAMES))
    parser.add_argument("--thresholds", default="1200,1400,1600,1800,2000",
                        help="target levels the exact exceedance is reported at")
    parser.add_argument("--at", type=float, default=1e-3,
                        help="per-line reference exceedance probability under Bernoulli")
    parser.add_argument("--n-lines", type=int, default=10)
    args = parser.parse_args()

    case = CASES[args.case]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    weights, problem = law_weights(case, args.dists.split(","))
    bits = problem.to_binary(np.arange(problem.nconf)).astype(float)
    p = np.asarray(case.nominal_p(), float)
    mu, names = problem.mu_table()

    print(f"{case.name}: {problem.nconf} configurations, "
          f"groups {case.candidate_groups()}")
    rows = {d: law_summary(w, weights["bernoulli"], bits, p, problem.S, thresholds)
            for d, w in weights.items()}
    print_target_table(rows, thresholds, problem.target)
    print_line_table(weights, mu, names, args.at, args.n_lines)


if __name__ == "__main__":
    main()
