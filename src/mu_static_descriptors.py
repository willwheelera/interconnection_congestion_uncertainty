"""Do cheap static descriptors reproduce the coarse high-mu/low-mu ranking that sampling gives?

Motivation: the tail-sampling harness does not return precise per-line mu, but it does rank lines
into high-mu vs low-mu fairly reliably.  If that ranking were obtainable from a single OPF solve
plus network intuition, sampling would be doing nothing an easy guess could not.  So this file tests
the two most intuitive static descriptors, from one base-configuration solve, against a selectable exact
nominal-weighted per-line target statistic -- the quantity the coarse sampling recovers -- on both
congestion regimes.  The target family and its numeric parameter come from the command line: `mean`
(E[mu], the default), `quantile <q>` (the weighted q-quantile of mu, e.g. q=0.95), or `exceed <thr>`
(the exceedance probability P[mu>thr], for any thr); see TARGET_FAMILIES:

  - proximity to the limit: slack_l = rating_l - |flow_l|;
  - generator->line PTDF sensitivity: swing sigma_l, the std of line flow under generator-outage
    swings (df_l^g = pg_g (sum_h gamma_h PTDF[l,h] - PTDF[l,g])), and z_l = slack_l / sigma_l.

Two figures, each with one row per case (proximity-driven case30, redispatch-driven case14, and the
2000-bus Texas N-1 case under its 2016 and its 2025 fleet): a rank figure (rank-rank scatter per
descriptor annotated with Spearman and the linear R^2, plus a combined rank fit) and a values figure
(raw descriptor magnitude vs E[mu], no fitted line -- the cloud itself is the evidence -- plus the
combined raw linear fit E[mu] ~ slack+sigma+z as predicted-vs-actual).  Both figures come from the
same code path: case_row(..., rank=True/False).  The Texas rows are per N-1 constraint rather than
per line and their descriptors come from src/texas/static_descriptors.py, which must be run first.

The base configuration is the all-generators-on one for case14/case30, which have no incumbent fleet
(with every generator off there is only load shed, so no flow to measure).  The Texas cases do have
one, so their base configuration is the queue-empty grid: E[candidates online] is 5.4 of 20, putting
all-on far outside the nominal mass.  This matters for slack -- 2025 fleet Spearman -0.71 off the
queue-empty grid against -0.63 off all-on, 2016 fleet -0.41 against -0.13 -- and not at all for
sigma, which is -0.75 on the 2016 fleet at every base configuration because it is a property of the
network and the candidate capacities rather than of the dispatch.

The point is to SHOW where these fail.  Result: raw proximity partly orders the ranking on case30
(Spearman -0.78) but the PTDF-sensitivity descriptors add nothing (sigma Spearman ~0), and the
combined linear fit's respectable-looking R^2 of 0.53 collapses to 0.06 once the single highest-mu
line is removed -- the whole fit is one leverage point (the same "a few points drive the fit" failure
that motivated the check; sigma's raw R^2 of 0.30 is the same artifact, 0.01 without that line).  On
the redispatch-driven case14 every descriptor and their combination fail (slack +0.36, wrong sign;
combined R^2 0.13).  The two Texas fleets then show that WHICH descriptor works is not a property of
the descriptor at all.  On the 2025 fleet it is proximity, as on case30: slack -0.71, sigma -0.25,
combined rank-R^2 0.52 of which slack alone is 0.51.  On the 2016 fleet the ordering inverts: sigma
-0.75 carries it, slack falls to -0.41, z comes out wrong-signed at +0.17, and combined rank-R^2 is
0.64 against slack alone at 0.17.  Same network, same queue, same recipe -- only the incumbent fleet
differs, and the descriptor that ranks congestion swaps.

Neither Texas fleet gives magnitude: raw R^2 0.15 (2025) and 0.31 (2016), and unlike case30 neither
is one leverage point (0.17 and 0.35 with the top constraint dropped), they are simply low.  So on
the realistic target the descriptors give a coarse who-is-near-the-limit ordering, by a rule that
cannot be chosen in advance, and nothing about how large mu gets: sampling captures combinatorial
active-set structure the descriptors cannot see.

Run from the repo root, e.g. `uv run python src/mu_static_descriptors.py quantile 0.95` or
`uv run python src/mu_static_descriptors.py exceed 200`; with no argument it uses `mean` (E[mu]).
"""
import sys
from collections import namedtuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

sys.path.insert(0, str(Path(__file__).parent))
from egret.data.model_data import ModelData
from egret.model_library.transmission.tx_calc import calculate_ptdf
from egret.model_library.defn import BasePointType
from tail_sampling.problem_setup import Problem, REPO_ROOT
from tail_sampling.distributions import make_dist, queue_p

P_ON = 0.65   # nominal on-marginal; a common scalar on sigma, so it does not affect any rank

# case, row label, descriptor source: "wide" = per-line, from this file's own all-on solve
# read out of the dual_cache; "texas" = per-N-1-constraint, from src/texas/static_descriptors.py
CASES = [("case30_gen_per_bus", "proximity-driven", "wide"),
         ("case14_gen_per_bus", "redispatch-driven", "wide"),
         ("activsg2000_original_v12", "Texas N-1, 2016 fleet", "texas"),
         ("activsg2000_2025", "Texas N-1, 2025 fleet", "texas")]
# descriptor name, raw-axis label, whether the raw axis is symlog
COLS = [("slack", "slack = rating - |flow|  (MW)", False),
        ("sigma", "swing sigma  (MW)", False),
        ("z", "z = slack / sigma", True)]

# tag: filename fragment; label: axis text; reduce: (per-config |mu| table, weights) -> per-line stat
Target = namedtuple("Target", "tag label reduce")

FIG_WIDTH, ROW_HEIGHT = 6.0, 1.7   # inches; 6 in wide fits a document text block
# type sizes to match: a 4-panel row is only ~1.5 in wide, so everything shrinks with it
STYLE = {"font.size": 4.5, "axes.labelsize": 5, "axes.titlesize": 5.5, "figure.titlesize": 7,
         "legend.fontsize": 4.5, "xtick.labelsize": 4, "ytick.labelsize": 4,
         "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
         "xtick.major.size": 2, "ytick.major.size": 2, "savefig.dpi": 300}


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


def base_setup(case):
    """PTDF, ratings, gen->bus map, and the dual cache for one enumerated case."""
    md = ModelData.read(str(REPO_ROOT / "data" / f"{case}.json"))
    els = md.data["elements"]
    buses, branches, gens = els["bus"], els["branch"], els["generator"]
    ref_bus = md.data["system"]["reference_bus"]
    bus_keys, branch_keys = list(buses), list(branches)
    bi = {b: i for i, b in enumerate(bus_keys)}
    ptdf = calculate_ptdf(branches, buses, branch_keys, bus_keys, ref_bus,
                          base_point=BasePointType.FLATSTART)
    ratings = np.array([float(branches[b]["rating_long_term"]) for b in branch_keys])
    g_bus = np.array([bi[g["bus"]] for g in gens.values()])
    dc = pd.read_parquet(REPO_ROOT / "results" / case / "dual_cache.parquet")
    return bus_keys, branch_keys, ptdf, ratings, g_bus, dc


def base_descriptors(bus_keys, branch_keys, ptdf, ratings, g_bus, dc):
    """Per-line slack, swing sigma, reliability z, and base mu at the all-on solve."""
    row = dc.loc[dc.index.max()]
    flow = np.array([row[("flow", b)] for b in branch_keys])
    mu_base = np.abs([row[("mu", b)] for b in branch_keys])
    pg = np.array([row[("p_gen", bus_keys[b])] for b in g_bus])
    Hg = ptdf[:, g_bus]
    tot = pg.sum()
    swing = ((Hg @ pg)[:, None] - Hg * pg[None, :]) / np.maximum(tot - pg, 1e-9)[None, :]
    df = pg[None, :] * (swing - Hg)                            # (nline x ngen) MW flow swing
    sigma = np.sqrt(P_ON * (1 - P_ON) * (df ** 2).sum(axis=1))
    slack = ratings - np.abs(flow)
    z = slack / np.maximum(sigma, 1e-9)
    return pd.DataFrame({"slack": slack, "sigma": sigma, "z": z, "mu_base": mu_base},
                        index=branch_keys)


def descriptors(case, kind):
    """Per-line (wide) or per-N-1-constraint (texas) slack, sigma, z, and base mu."""
    if kind == "texas":
        return pd.read_parquet(REPO_ROOT / "results" / case / "static_descriptors_off.parquet")
    return base_descriptors(*base_setup(case))


def nominal_law(kind):
    """The case's nominal law: the Texas queue's calibrated per-candidate build p, else
    the shared default the samplers are scored under."""
    if kind != "texas":
        return None
    p = queue_p()
    return make_dist("bernoulli", len(p), p)


def load_target(case, keys, target, kind):
    """The selected nominal-weighted per-element target statistic (what sampling recovers)."""
    prob = Problem(REPO_ROOT / "results" / case, dist=nominal_law(kind))
    mu_table, cols = prob.mu_table()
    idx = [cols.index(k) for k in keys]
    return target.reduce(mu_table, prob.w)[idx]


def compute_case(case, kind, target):
    """Descriptors, the selected sampled target, binding mask, leverage index, and predictors."""
    d = descriptors(case, kind)
    y = load_target(case, list(d.index), target, kind)
    binding = d.mu_base.values > 1e-6
    top = int(y.argmax())
    X = d[["slack", "sigma", "z"]].values
    return d, y, binding, top, X


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


def panel(ax, x, y, xlabel, ylabel, text, ha="left"):
    """Scatter with axis labels and a corner annotation box."""
    ax.scatter(x, y, s=4, c="#3a6", alpha=0.85, zorder=2)
    lo, hi = float(np.min(y)), float(np.max(y))
    ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.34 * (hi - lo))   # headroom under the text box
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.03 if ha == "left" else 0.96, 0.97, text, transform=ax.transAxes, va="top", ha=ha,
            bbox=dict(boxstyle="round", pad=0.25, fc="w", ec="0.6", lw=0.4))


def mark(ax, x, y, binding, top, rank):
    """Rank basis marks the lines binding at the base solve; raw basis circles the leverage line."""
    if rank:
        ax.scatter(x[binding], y[binding], s=5, c="#d33", edgecolor="k", lw=0.3, zorder=3)
    else:
        ax.scatter(x[top], y[top], s=34, facecolors="none", edgecolors="k", lw=0.7, zorder=4)


def target_axis(lab, rank):
    return f"rank of {lab}" if rank else lab


def case_row(axes, d, y, binding, top, X, lab, rank=False):
    """One case: a panel per descriptor then the combined fit, in ranks (rank=True) or raw values."""
    tx = rankdata if rank else (lambda v: v)
    ty, ylabel = tx(y), target_axis(lab, rank)
    for ax, (name, xlabel, symlogx) in zip(axes, COLS):
        feat = d[name].values
        panel(ax, tx(feat), ty, f"rank of {name}" if rank else xlabel, ylabel,
              descriptor_text(feat, y, top, rank), ha="left" if rank else "right")
        mark(ax, tx(feat), ty, binding, top, rank)
        if symlogx and not rank:
            ax.set_xscale("symlog", linthresh=1.0)
            ax.xaxis.get_major_locator().set_params(numticks=4)   # decades do not fit at 1.5 in
    combined_fit(axes[3], X, y, top, lab, rank, tx)


def descriptor_text(feat, y, top, rank):
    """Annotation for one descriptor panel; both R^2 are on the raw values, not the ranks."""
    r2 = ols_r2(feat[:, None], y)[0]
    if rank:
        return f"Spearman = {spearmanr(feat, y).correlation:+.2f}\nlinear R2 = {r2:.2f}"
    return f"R2 = {r2:.2f}\nno circled line: {r2_no_top(feat[:, None], y, top):.2f}"


def combined_fit(ax, X, y, top, lab, rank, tx):
    """Fourth panel: slack+sigma+z fit in the row's basis, plotted as predicted vs actual."""
    Xt = np.column_stack([tx(col) for col in X.T])
    ty = tx(y)
    r2, pred = ols_r2(Xt, ty)
    text = (f"combined rank-R2 = {r2:.2f}\nslack alone = {ols_r2(Xt[:, :1], ty)[0]:.2f}" if rank
            else f"R2 = {r2:.2f}\nno circled line: {r2_no_top(Xt, ty, top):.2f}")
    xlabel = (f"rank of predicted {lab}" if rank else f"predicted {lab} (slack+sigma+z)")
    panel(ax, pred, ty, xlabel, target_axis(lab, rank), text)
    ax.set_title(f"combined fit ({'rank' if rank else 'raw'})")
    if not rank:
        mark(ax, pred, ty, None, top, rank)
        lims = [min(pred.min(), ty.min()), max(pred.max(), ty.max())]
        ax.plot(lims, lims, ls="--", c="#333", lw=0.5)


def legend_handle(lab, rank):
    """Marker key: binding lines (rank figure) or the circled leverage line (values figure)."""
    kwargs = dict(marker="o", ls="", mec="k",)
    if rank:
        kwargs["label"] = "binding at base solve"
        kwargs["c"] = "#d33"
    else:
        kwargs["label"] = f"highest-{lab} line (circled point, dominates the fit)"
        kwargs["mfc"] = "none"
    return plt.Line2D([], [], **kwargs)


def make_figure(data, lab, rank=False):
    """Grid of one row per case, in ranks (rank=True) or raw values; FIG_WIDTH inches wide."""
    subtitle = "rank ordering" if rank else "raw magnitude and combined linear fit"
    height = ROW_HEIGHT * len(CASES) + 0.5
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(CASES), 4, figsize=(FIG_WIDTH, height))
        for c, (case, regime, _) in enumerate(CASES):
            case_row(axes[c], *data[case], lab, rank)
            axes[c, 0].set_ylabel(f"{case}\n({regime})\n{target_axis(lab, rank)}")
        fig.legend(handles=[legend_handle(lab, rank)], loc="lower center", frameon=False)
        fig.suptitle(f"Cheap static descriptors vs the sampled {lab}: {subtitle}")
        # strips reserved in inches, not fractions: they must not grow with the case count
        fig.tight_layout(rect=(0, 0.16 / height, 1, 1 - 0.24 / height), pad=0.6, w_pad=0.8)
    return fig


def main():
    target = parse_target(sys.argv[1:])
    data = {case: compute_case(case, kind, target) for case, _, kind in CASES}
    for tag, rank in [("rank", True), ("values", False)]:
        fig = make_figure(data, target.label, rank)
        out = REPO_ROOT / "results" / f"mu_static_descriptors_{target.tag}_{tag}.png"
        fig.savefig(out, dpi=300)
        print("wrote", out)
    plt.show()


if __name__ == "__main__":
    main()
