"""Is the target an additive function of the queue bits?

A cheaper way to replace sampling than the per-constraint descriptors: if the target were
a sum of per-candidate contributions, S(x) = c + sum_g a_g x_g, then one linear model
fitted to a handful of configurations would give every configuration's target, and the
tail would be the tail of a weighted sum of Bernoulli bits rather than something the
network has to be re-solved to learn.

Two measurements of that, over a case's full enumeration:

  - the ordinary least squares fit of the target on the ngen bits, reported as R^2 and as
    how far up the target range the fitted values reach;
  - the exact variance share of each interaction order, from the Walsh-Hadamard transform
    of the target over the hypercube.  Order 1 is the additive (main-effect) share, so it
    is an upper bound on any linear model's R^2 under the uniform measure; orders >= 2 are
    what no linear model in the bits can represent.

**Result.** The target is additive exactly where a single candidate carries it, and the
additive part is never the tail.

Where the fit looks good, one coefficient is the fit.  case118_queue reaches an order-1
share of 1.00 on a max_mu that takes *two* levels, 115.3 and 153.4 $/MWh, switched by one
candidate: a one-variable step function, not an additive target.  activsg2000_2025 is 0.99
additive with 0.995 of the fit's variance in one candidate over a 58 $/MWh range, and
activsg2000_original_v12 is 0.91 with 0.94 in one candidate.  The cases with genuine
combinatorial spread are not additive at all: the order-1 share is 0.28 on case30_queue --
0.26 of its variance is in pair interactions alone and 0.18 sits above order 4 -- and 0.52
on both case14 testbeds.

And in every case with a spread the fitted values cover far less range than the target: the
largest fitted value on case30_queue is 199 against a true maximum of 451, on
case30_gen_per_bus 277 against 982.  The linear model therefore places *zero* configurations
above the top-1% level of the target on every case except the two-level one -- the tail
capture rate is 0.00 across the matrix.  This is what a max-of-duals target does to a linear
fit: extreme mu comes from combinations that push one element onto a binding active set, and
regressing that over all configurations spreads it across every coefficient.

So a high R^2 here does not license replacing the sweep, and it does not make the choice of
nominal law a second-order question either: an additive S makes the target a weighted sum of
the bits, whose tail still depends on how those bits co-vary (see
`diagnostics/check_law_coupling.py`).

`figures.additivity` draws this.

Usage:
    uv run python -m src.descriptors.additivity
    uv run python -m src.descriptors.additivity --target one_mu
"""

import argparse

import numpy as np
import pandas as pd

from src.cases.registry import CASES
from src.sampling.distributions import nominal_law
from src.sampling.problem import read_parquet_meta

# case name, short label for the figures
ADDITIVITY_CASES = [("case14_queue", "IEEE 14 queue"),
                    ("case30_queue", "IEEE 30 queue"),
                    ("case118_queue", "IEEE 118 queue"),
                    ("activsg2000_original_v12", "Texas, 2016 fleet"),
                    ("activsg2000_2025", "Texas, 2025 fleet"),
                    ("case14_gen_per_bus", "case14 gen-per-bus"),
                    ("case30_gen_per_bus", "case30 gen-per-bus")]
MAX_ORDER = 4          # orders resolved individually; the rest are pooled into "5+"
TAIL_Q = 0.01          # tail fraction the fit is asked to reproduce


def config_bits(ids, ngen):
    """The ngen availability bits of each configuration id, MSB-first."""
    powers = (1 << np.arange(ngen - 1, -1, -1)).astype(np.int64)
    return ((ids[:, None] & powers[None, :]) != 0).astype(float)


def load_target(case, target):
    """A case's target over its full enumeration: (bits, y, ids, ngen).

    Ids absent from the cache (an infeasible configuration) are dropped.
    """
    df, meta = read_parquet_meta(case.target_cache)
    ngen = int(meta["ngen"])
    ids = df.index.to_numpy().astype(np.int64)
    return config_bits(ids, ngen), df[target].to_numpy(), ids, ngen


def linear_fit(bits, y, w=None):
    """Weighted OLS of y on the bits with an intercept; returns (coefficients, fitted)."""
    X = np.column_stack([np.ones(len(y)), bits])
    if w is None:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    else:
        r = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(X * r[:, None], y * r, rcond=None)
    return coef, X @ coef


def r2(y, pred, w=None):
    """Fraction of (optionally weighted) variance the fit explains."""
    w = np.ones(len(y)) if w is None else w
    w = w / w.sum()
    mean = (w * y).sum()
    return 1 - (w * (y - pred) ** 2).sum() / (w * (y - mean) ** 2).sum()


def nominal_weights(case, ids, ngen):
    """Nominal probability of each enumerated id under the case's independent law."""
    logw = nominal_law(case).log_weight(config_bits(ids, ngen))
    w = np.exp(logw - logw.max())
    return w / w.sum()


def walsh_hadamard(y):
    """Walsh-Hadamard coefficients of a length-2^n array (the +-1 Fourier basis)."""
    a = y.astype(float).copy()
    n = a.size
    h = 1
    for _ in range(n.bit_length() - 1):
        a = a.reshape(-1, 2, h)
        top, bot = a[:, 0].copy(), a[:, 1].copy()
        a[:, 0], a[:, 1] = top + bot, top - bot
        a = a.reshape(n)
        h *= 2
    return a / n


def variance_by_order(y, ids, ngen):
    """Share of the target's variance carried by each interaction order, orders 1..MAX_ORDER+.

    Exact for a complete enumeration; a missing id is filled with the mean of the rest,
    which is one configuration in 2^20 on the gen-per-bus cases and none elsewhere.
    """
    full = np.full(1 << ngen, np.nan)
    full[ids] = y
    full[np.isnan(full)] = np.nanmean(full)
    power = walsh_hadamard(full) ** 2
    order = np.array([bin(i).count("1") for i in range(full.size)])
    total = power[1:].sum()
    shares = [power[order == k].sum() / total for k in range(1, MAX_ORDER + 1)]
    return np.array(shares + [1 - sum(shares)])


def tail_capture(y, pred, q=TAIL_Q):
    """Fraction of the top-q configurations of y whose fitted value also clears that level."""
    return float((pred[y >= np.quantile(y, 1 - q)] >= np.quantile(y, 1 - q)).mean())


def top_candidate_share(coef):
    """Share of the fit's variance carried by its single largest candidate coefficient.

    Bits are independent and equally weighted under the uniform measure, so each
    candidate contributes in proportion to its squared coefficient.
    """
    a = coef[1:] ** 2
    return float(a.max() / a.sum())


def compute_case(name, target):
    """Every additivity measurement for one case, plus the arrays the figure draws."""
    case = CASES[name]
    bits, y, ids, ngen = load_target(case, target)
    coef, pred = linear_fit(bits, y)
    w = nominal_weights(case, ids, ngen)
    _, pred_w = linear_fit(bits, y, w)
    shares = variance_by_order(y, ids, ngen)
    return {"case": name, "ngen": ngen, "n": len(y), "spread": float(y.max() - y.min()),
            # to the tenth of a $/MWh: below that a level is solver noise, not a level
            "n_distinct": int(np.unique(y.round(1)).size), "r2": r2(y, pred),
            "r2_nominal": r2(y, pred_w, w), "order1": float(shares[0]),
            "order2": float(shares[1]), "shares": shares,
            "top_share": top_candidate_share(coef),
            "pred_max_frac": float((pred.max() - y.min()) / (y.max() - y.min())),
            "tail_capture": tail_capture(y, pred), "y": y, "pred": pred}


def compute_all(target="max_mu"):
    """compute_case for every case in the matrix, keyed by case name."""
    return {name: compute_case(name, target) for name, _ in ADDITIVITY_CASES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="max_mu", help="target_cache column to fit")
    args = parser.parse_args()

    rows = compute_all(args.target)
    cols = ["case", "ngen", "n_distinct", "spread", "r2", "r2_nominal", "order1", "order2",
            "top_share", "pred_max_frac", "tail_capture"]
    df = pd.DataFrame([{c: r[c] for c in cols} for r in rows.values()])
    print(f"target: {args.target}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
