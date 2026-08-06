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

**Result** (at the modal base, on E[mu]).  Raw proximity partly orders case30 (Spearman -0.78)
but the PTDF-sensitivity descriptors add nothing (sigma Spearman -0.06), and the combined linear
fit's respectable-looking R^2 of 0.53 collapses to 0.06 once the single highest-mu line is
removed -- the whole fit is one leverage point.  On the redispatch-driven case14 every descriptor
and their combination fail (slack +0.36, wrong sign; combined R^2 0.13).  The two Texas fleets
then show that *which* descriptor works is not a property of the descriptor at all: on the 2016
fleet sigma carries the ordering (-0.75) while slack falls to -0.37 and z comes out wrong-signed
at +0.20, so the combined rank-R^2 of 0.62 stands against slack alone at 0.14; on the 2025 fleet
the two are comparable and both weak (slack -0.38, sigma -0.43, combined rank-R^2 0.33).  Same
network, same queue, same recipe -- only the incumbent fleet differs, and the weight the ranking
puts on each descriptor moves with it.

Neither Texas fleet gives magnitude: raw R^2 0.09 (2025) and 0.29 (2016), and unlike case30
neither is one leverage point (0.13 and 0.33 with the top constraint dropped) -- they are simply
low.  So the descriptors give a coarse who-is-near-the-limit ordering, by a rule that cannot be
chosen in advance, and nothing about how large mu gets: sampling captures combinatorial
active-set structure a single configuration cannot see.

**Base configuration.** The descriptors are read off one configuration, and the default is the
most probable one under the case's nominal law (`descriptors.compute`), which is the only choice
that does not put the base outside the mass the target averages over.  It costs the gen-per-bus
cases nothing -- their law has no queue and favors availability, so its mode is all-on, the
configuration they were always read at -- and it replaces the queue-empty grid on the
incumbent-fleet cases, where 5 of 20 candidates clear.  On the 2016 fleet that choice moves slack
(-0.37 modal, -0.41 queue-empty, -0.13 all-on) and leaves sigma at -0.75 everywhere, since sigma
is a property of the network and the candidate capacities rather than of the dispatch; on the
2025 fleet, whose queue barely moves the network state, modal and queue-empty agree to two
decimals on every descriptor while all-on still costs slack (-0.24).  So the modal base never
loses to the queue-empty one by more than rounding, and both beat the corner the earlier
gen-per-bus reading would have suggested.

This file computes; `figures.descriptors` draws.

Usage:
    uv run python -m src.figures.descriptors                # mean, the default
    uv run python -m src.figures.descriptors quantile 0.95
    uv run python -m src.figures.descriptors exceed 200
    uv run python -m src.figures.descriptors --base on
"""

from collections import namedtuple

import numpy as np

from src.cases import registry
from src.descriptors.compute import DEFAULT_BASE, descriptors
from src.sampling.distributions import nominal_law
from src.sampling.problem import Problem

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


def compute_case(name, target, base=DEFAULT_BASE):
    """Descriptors, the sampled target, binding mask, leverage index, and predictors."""
    case = registry.get(name)
    d = descriptors(case, base)
    y = load_target(case, list(d.index), target)
    binding = d.mu_base.values > 1e-6
    top = int(y.argmax())
    X = d[["slack", "sigma", "z"]].values
    return d, y, binding, top, X


def compute_all(target, base=DEFAULT_BASE):
    """Every study case's computed row at one base configuration, keyed by case name."""
    return {name: compute_case(name, target, base) for name, _ in STUDY_CASES}
