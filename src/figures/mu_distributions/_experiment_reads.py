"""Figure-ready slices of one case's stored sampling experiment.

The comparison figures each read a narrow slice of a sweep and shape it for plotting: the
direct-method sample rows, or one (dist, target, snapshot, budget) cell of the exceedance
curves.  Those reads live here so a figure never reaches into a sibling figure for them; the
raw parquet loaders they build on are in `sampling.problem`.

    direct_samples   a case, its direct-method sample rows, and the run metadata
    direct_slice     the direct rows of one law/target/snapshot cell of a loaded frame
    load_slice       one (dist, target, snapshot, budget) cell of the exceedance curves
    select_series    keep the curve series whose method is in the requested selection
    read_dual_samples  the wide per-line |mu| MCMC draw table of a sample-only case
    read_modal_point   per-line |mu| of that case's most-probable config
"""

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from src.cases.registry import get_case
from src.sampling.problem import load_experiment, load_experiment_rows, experiment_meta
from src.figures.style import select_methods


def direct_samples(case, kind="samples"):
    """A case, only its direct-method experiment rows, and the run metadata.

    Every direct-* figure uses only the direct rows, so the method filter is pushed into the
    parquet scan and the other samplers -- the bulk of a sweep -- are never materialized as
    pandas.  Metadata comes from the shard schemas separately, since a filtered read carries
    none.
    """
    c = get_case(case)
    frame = load_experiment_rows(c.results_dir, kind, pads.field("method") == "direct")
    return c, frame, experiment_meta(c.results_dir, kind)


def direct_slice(samples, dist, budgets, target="max_mu", snapshot="multi"):
    """The direct-method rows of one nominal law, target and snapshot at the given budgets.

    Errors out naming what the experiment holds when the slice is empty -- the usual cause is
    a law that run_experiments never swept for this case.
    """
    sl = samples[(samples.method == "direct") & (samples.dist == dist)
                 & (samples.target == target) & (samples.snapshot == snapshot)
                 & samples.budget.isin(budgets)]
    if sl.empty:
        raise ValueError(
            f"no direct draws for dist={dist}, target={target}, snapshot={snapshot} at "
            f"budgets {list(budgets)}; this experiment holds dists "
            f"{sorted(samples.dist.unique())} at budgets "
            f"{sorted(int(b) for b in samples.budget.unique())} "
            f"-- sweep the law first with sampling.run_experiments")
    return sl


def load_slice(case_dir, sel):
    """Return (thresholds, truth, {(method, move): (n_rep, n_thr)}, meta) for one slice."""
    df, meta = load_experiment(case_dir, "curves")
    for key, val in sel.items():
        df = df[df[key] == val]
    thresholds = np.sort(df["threshold"].unique())
    truth = (df[df.method == "truth"].set_index("threshold")["estimate"]
             .reindex(thresholds).to_numpy())
    series = {}
    for (m, mv), g in df[df.method != "truth"].groupby(["method", "ascent_move"]):
        series[(m, mv)] = (g.pivot(index="rep", columns="threshold", values="estimate")
                           .reindex(columns=thresholds).to_numpy())
    return thresholds, truth, series, meta


def select_series(series, methods):
    """Keep only the curve series whose method is in the requested selection."""
    keep = set(select_methods({m for m, _ in series}, methods))
    return {key: est for key, est in series.items() if key[0] in keep}


def read_dual_samples(case, law):
    """The wide per-line |mu| MCMC draw table (nsamples x nline) and its line names.

    The sample-only (ngen ~ 195) analogue of `direct_samples`: that case has no exact table,
    only the draws in results/<case>/mcmc/<law>/dual_samples.parquet.  `case` is a resolved
    Case; returns |mu| as a dense (nsamples, nline) array and the line names from its columns.
    """
    df = pd.read_parquet(case.results_dir / "mcmc" / law / "dual_samples.parquet")
    return np.abs(df["mu"].to_numpy()), list(df["mu"].columns)


def read_modal_point(case, law):
    """Per-constraint |mu| of the most-probable config from modal_point.parquet, {} if absent."""
    path = case.results_dir / "mcmc" / law / "modal_point.parquet"
    if not path.exists():
        return {}
    return {c: float(v) for c, v in pd.read_parquet(path)["mu"].iloc[0].items()}
