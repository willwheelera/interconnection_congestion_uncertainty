"""Do cheap static descriptors reproduce the high-mu/low-mu ranking that sampling gives?

This is Claim 2 of notes/paper_plan.md, and the answer is no.  If the per-line ranking were
obtainable from a single OPF solve plus network intuition, sampling would be doing nothing an
easy guess could not.  So this scores the two most intuitive single-configuration descriptors
(`descriptors.compute`) against an exact nominal-weighted per-line target statistic -- the
quantity sampling recovers -- across the case matrix:

  - proximity to the limit: slack_l = rating_l - |flow_l|;
  - generator->line PTDF sensitivity: swing sigma_l, the std of line flow under
    generator-outage swings, and z_l = slack_l / sigma_l.

The target family and its parameter come from the command line: `mean` (E[mu], the default),
`quantile <q>` (the weighted q-quantile of mu), or `exceed <thr>` (P[mu > thr]).

**Result.** Raw proximity partly orders case30 (Spearman -0.78) but the PTDF-sensitivity
descriptors add nothing (sigma Spearman ~0), and the combined linear fit's respectable-looking
R^2 of 0.53 collapses to 0.06 once the single highest-mu line is removed -- the whole fit is one
leverage point.  On the redispatch-driven case14 every descriptor and their combination fail
(slack +0.36, wrong sign; combined R^2 0.13).  The two Texas fleets then show that *which*
descriptor works is not a property of the descriptor at all: on the 2025 fleet it is proximity,
as on case30 (slack -0.71, sigma -0.25, combined rank-R^2 0.52 of which slack alone is 0.51),
while on the 2016 fleet the ordering inverts (sigma -0.75 carries it, slack falls to -0.41, z
comes out wrong-signed at +0.17, combined rank-R^2 0.64 against slack alone at 0.17).  Same
network, same queue, same recipe -- only the incumbent fleet differs, and the descriptor that
ranks congestion swaps.

Neither Texas fleet gives magnitude: raw R^2 0.15 (2025) and 0.31 (2016), and unlike case30
neither is one leverage point (0.17 and 0.35 with the top constraint dropped) -- they are simply
low.  So the descriptors give a coarse who-is-near-the-limit ordering, by a rule that cannot be
chosen in advance, and nothing about how large mu gets: sampling captures combinatorial
active-set structure a single configuration cannot see.

**Base configuration.** All-generators-on for the gen-per-bus cases, which have no incumbent
fleet (with every generator off there is only load shed, so no flow to measure).  The
incumbent-fleet cases use the queue-empty grid instead: E[candidates online] is 5.4 of 20 on
Texas, putting all-on far outside the nominal mass.  This matters for slack -- 2025 fleet
Spearman -0.71 off the queue-empty grid against -0.63 off all-on, 2016 fleet -0.41 against
-0.13 -- and not at all for sigma, which is -0.75 on the 2016 fleet at every base configuration
because it is a property of the network and the candidate capacities rather than of the dispatch.

This file computes; `figures.descriptors` draws.

Usage:
    uv run python -m figures.descriptors                # mean, the default
    uv run python -m figures.descriptors quantile 0.95
    uv run python -m figures.descriptors exceed 200
"""

from collections import namedtuple

import numpy as np

from cases import registry
from descriptors.compute import descriptors
from sampling.distributions import make_dist
from sampling.problem import Problem

# The cases the study reports on, with the congestion regime each one turned out to show.
# The regime label is a finding about the case, not a property of it, so it lives here
# rather than in the registry.
STUDY_CASES = [("case30_gen_per_bus", "proximity-driven"),
               ("case14_gen_per_bus", "redispatch-driven"),
               ("activsg2000_original_v12", "Texas N-1, 2016 fleet"),
               ("activsg2000_2025", "Texas N-1, 2025 fleet")]

# descriptor name, raw-axis label, whether the raw axis is symlog
COLS = [("slack", "slack = rating - |flow|  (MW)", False),
        ("sigma", "swing sigma  (MW)", False),
        ("z", "z = slack / sigma", True)]

# tag: filename fragment; label: axis text; reduce: (per-config |mu| table, weights) -> per-line stat
Target = namedtuple("Target", "tag label reduce")


def weighted_quantile(values, w, q):
    """Per-column weighted q-quantile of `values` (nconf x ncol) under normalized weights w."""
    out = np.empty(values.shape[1])
    for j in range(values.shape[1]):
        order = np.argsort(values[:, j])
        cw = np.cumsum(w[order])
        out[j] = values[order[min(np.searchsorted(cw, q), len(cw) - 1)], j]
    return out


def target_mean(args):
    return Target("mean", "E[mu]", lambda mu, w: w @ mu)


def target_quantile(args):
    q = float(args[0])
    return Target(f"q{q:g}", f"p{q * 100:g} mu", lambda mu, w: weighted_quantile(mu, w, q))


def target_exceed(args):
    thr = float(args[0])
    return Target(f"gt{thr:g}", f"P[mu>{thr:g}]", lambda mu, w: w @ (mu > thr))


def parse_target(argv):
    """Target for `family [param ...]`; family in TARGET_FAMILIES, default mean."""
    TARGET_FAMILIES = {"mean": target_mean, "quantile": target_quantile, "exceed": target_exceed}
    family = argv[0] if argv else "mean"
    return TARGET_FAMILIES[family](argv[1:])


def nominal_law(case):
    """The case's nominal law: its calibrated queue marginals, or the samplers' default.

    A case with no queue file gets None, which leaves Problem on the shared default law the
    samplers are scored under.
    """
    if case.candidates_file is None:
        return None
    p = case.build_p()
    return make_dist("bernoulli", len(p), p)


def load_target(case, keys, target):
    """The selected nominal-weighted per-element target statistic (what sampling recovers)."""
    prob = Problem(case.results_dir, dist=nominal_law(case))
    mu_table, cols = prob.mu_table()
    idx = [cols.index(k) for k in keys]
    return target.reduce(mu_table, prob.w)[idx]


def ols_r2(X, y):
    """OLS R^2 (and fitted values) with intercept; predictors standardized so R^2 is scale-free."""
    Xs = (X - X.mean(0)) / X.std(0)
    A = np.column_stack([np.ones(len(y)), Xs])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - pred) ** 2).sum() / ss, pred


def r2_no_top(X, y, top):
    """R^2 of the same OLS after removing the single highest-target line (leverage test)."""
    keep = np.arange(len(y)) != top
    return ols_r2(X[keep], y[keep])[0]


def compute_case(name, target):
    """Descriptors, the sampled target, binding mask, leverage index, and predictors."""
    case = registry.get(name)
    d = descriptors(case)
    y = load_target(case, list(d.index), target)
    binding = d.mu_base.values > 1e-6
    top = int(y.argmax())
    X = d[["slack", "sigma", "z"]].values
    return d, y, binding, top, X


def compute_all(target):
    """Every study case's computed row, keyed by case name."""
    return {name: compute_case(name, target) for name, _ in STUDY_CASES}
