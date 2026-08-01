"""The congestion-dual tail problem, as a class the runner constructs.

A `Problem` bundles everything that depends on the *problem* (as opposed to any
one estimation method) for a single enumerated case:

  - the target S(x) = max_b |mu_b| loaded from that case's dual cache,
  - the nominal law over generator-availability configs (a correlated Markov random
    field by default; see mrf.py).  The law is exposed *only* as a normalized weight
    vector over all configs, a pmf lookup, and an exact categorical sampler, so no
    method assumes the law has any particular form (in particular, not a product or
    Bernoulli form),
  - the exact nominal-weighted exceedance curve used as ground truth,
  - bit/id helpers and the nominal sampler shared across methods.

The case (results directory) is chosen by the executing script and passed in, along
with an optional nominal law; the number of generators is read off the cache (the
all-on config is the maximum id present), so nothing here is hardcoded to one case and
no problem state lives as a module global.  The model has cached duals for all 2^ngen
configurations, so the nominal weight of every config is known exactly by enumeration,
the exceedance curve is exact ground truth, and every sampler can be scored against it.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tail_sampling.mrf import default_mrf

REPO_ROOT = Path(__file__).parent.parent.parent

DATA = REPO_ROOT / "data"

# Per-case run configuration: the build-probability marginals and whether a generic per-bus
# sensitivity proxy exists for the ascent move strategies.  `p` is either a scalar mean,
# which make_dist spreads across the generators, or the case's queue file, whose
# per-candidate p is used as given (see distributions.case_p).
CASES = {
    "case14_gen_per_bus": dict(p=0.65, proxy=True),
    "case30_gen_per_bus": dict(p=0.65, proxy=True),
    "activsg2000_original": dict(p=DATA / "activsg2000_queue20.json", proxy=False),
    "activsg2000_2025": dict(p=DATA / "activsg2000_queue20.json", proxy=False),
    "activsg2000_original_v12": dict(p=DATA / "activsg2000_queue20.json", proxy=False),
    "case14_queue": dict(p=DATA / "case14_queue_candidates.json", proxy=False),
    "case30_queue": dict(p=DATA / "case30_queue_candidates.json", proxy=False),
    "case118_queue": dict(p=DATA / "case118_queue_candidates.json", proxy=False),
}


def case_tag(case):
    """Short unique filename tag for a case (drops the shared _gen_per_bus boilerplate)."""
    return case.replace("_gen_per_bus", "")


def write_parquet_meta(df, meta, path):
    """Write df to parquet with the run metadata embedded in the file's schema."""
    table = pa.Table.from_pandas(df)
    schema_meta = {**(table.schema.metadata or {}), b"run_meta": json.dumps(meta).encode()}
    pq.write_table(table.replace_schema_metadata(schema_meta), path)


def read_parquet_meta(path):
    """Read a parquet written by write_parquet_meta; return (df, meta)."""
    table = pq.read_table(path)
    meta = json.loads(table.schema.metadata[b"run_meta"])
    return table.to_pandas(), meta


def experiment_dir(case_dir, kind):
    """Directory of per-budget parquet shards for an experiment ("samples" or "curves")."""
    return Path(case_dir) / f"experiment_{kind}"


def write_experiment_shard(df, meta, case_dir, kind, budget):
    """Write one budget's slice of an experiment as its own parquet shard.

    Sharding by budget means a later run over a different budget set adds files rather than
    overwriting an earlier run; only a rerun of the same budget replaces its own shard."""
    d = experiment_dir(case_dir, kind)
    d.mkdir(exist_ok=True)
    write_parquet_meta(df, {**meta, "budgets": [budget]}, d / f"b{budget}.parquet")


def load_experiment(case_dir, kind):
    """Read and combine every budget shard of an experiment into one (df, meta).

    Shards carry the same case-level metadata tagged with their own budget; the combined
    budgets field is the sorted union across shards.  Falls back to a legacy monolithic
    experiment_{kind}.parquet when no shard directory is present."""
    d = experiment_dir(case_dir, kind)
    if not d.exists():
        return read_parquet_meta(Path(case_dir) / f"experiment_{kind}.parquet")
    frames, budgets, meta = [], [], None
    for shard in sorted(d.glob("b*.parquet")):
        df, meta = read_parquet_meta(shard)
        frames.append(df)
        budgets.extend(meta["budgets"])
    return pd.concat(frames, ignore_index=True), {**meta, "budgets": sorted(set(budgets))}


class Problem:
    """One enumerated congestion-dual tail case: target, nominal law, ground truth."""

    def __init__(self, results_dir, dist=None):
        self.results_dir = Path(results_dir)
        df = pd.read_parquet(self.results_dir / "dual_cache.parquet")
        df.index = df.index.astype(int)
        self.ngen = int(df.index.max()).bit_length()            # all-on config is the max id
        self.nconf = 1 << self.ngen
        self._powers = (1 << np.arange(self.ngen - 1, -1, -1)).astype(np.int64)  # MSB-first
        self.dist = dist if dist is not None else default_mrf(self.ngen)  # nominal law
        self.S = self._build_S(df)
        self.w = self._nominal_weights()

    def _build_S(self, df):
        """S[id] = max_b |mu_b| for every config id; load-infeasible ids get S = 0."""
        S = np.zeros(self.nconf)
        S[df.index.values] = df["mu"].abs().max(axis=1).values
        return S

    def _nominal_weights(self):
        """Exact normalized nominal probability of every config under the nominal law.

        Enumerates the law's unnormalized log weight over all 2^ngen configs and
        normalizes; the law's form lives in self.dist, not here."""
        logw = self.dist.log_weight(self.to_binary(np.arange(self.nconf)))
        w = np.exp(logw - logw.max())
        return w / w.sum()

    def to_binary(self, arr):
        return ((arr[:, None] & self._powers[None, :]) != 0).astype(np.int8)

    def log_nominal_pmf(self, bits):
        """log of the normalized nominal pmf for each row of bits, by exact lookup.

        Pure table lookup into the enumerated weights, so it makes no assumption about
        the law's form (Bernoulli, MRF, or otherwise)."""
        return np.log(self.w[self.bits_to_id(bits)])

    def true_exceedance(self, thresholds):
        """Exact P[S >= b] under the nominal law for each b in thresholds."""
        return np.array([self.w[self.S >= b].sum() for b in thresholds])

    def thresholds_for_probs(self, probs):
        """Find b such that the exact nominal P[S >= b] hits each target probability."""
        order = np.argsort(-self.S)
        surv = np.cumsum(self.w[order])
        s_sorted = self.S[order]
        return np.array([s_sorted[np.searchsorted(surv, q)] for q in probs])

    def bits_to_id(self, bits):
        return bits @ self._powers

    def draw_nominal(self, n, rng):
        """iid draws from the nominal law via its exact categorical over all configs."""
        return self.to_binary(rng.choice(self.nconf, size=n, p=self.w))

    def evaluator(self):
        """A fresh Evaluator over this problem's S (counts solver calls per run)."""
        return Evaluator(self.S)

    def direct_exceedance(self, rng, n_draws, thresholds):
        """Baseline: empirical CCDF from n_draws iid draws of the nominal law."""
        S = self.evaluator()(self.bits_to_id(self.draw_nominal(n_draws, rng)))
        return (S[:, None] >= thresholds[None, :]).mean(axis=0)

    def mu_table(self):
        """Per-line |mu| for every config id (nconf x nlines) and the line names;
        load-infeasible ids get 0."""
        mu = pd.read_parquet(self.results_dir / "dual_cache.parquet")["mu"].abs()
        table = np.zeros((self.nconf, mu.shape[1]))
        table[mu.index.astype(int).values] = mu.values
        return table, list(mu.columns)


class TargetProblem(Problem):
    """Problem backed by a common target_cache.parquet (build_target_cache.py).

    One cache holds every case in the same schema (max_mu, sum_mu, one_mu, one_lmp,
    n_on, max_mu_single), so the runner swaps case, guiding target, and nominal law
    without touching any sampler.  self.S is the chosen guiding target; self.targets
    keeps all of them (full-nconf, 0 on missing/infeasible ids) so run_experiments can
    score any target from the same weighted sample.  A missing-data target (all-NaN
    column, e.g. one_lmp with no cached LMP) is excluded from self.available.
    """

    def __init__(self, target_cache, target="max_mu", dist=None, sensitivity=None):
        self.results_dir = Path(target_cache).parent
        df, self.meta = read_parquet_meta(target_cache)
        df.index = df.index.astype(int)
        self.target = target
        self.ngen = int(self.meta["ngen"])
        self.nconf = 1 << self.ngen
        self._powers = (1 << np.arange(self.ngen - 1, -1, -1)).astype(np.int64)  # MSB-first
        self.dist = dist if dist is not None else default_mrf(self.ngen)
        self.targets = {c: self._to_full(df[c]) for c in df.columns if c != "n_on"}
        self.available = {c for c in self.targets if df[c].notna().any()}
        self.S = self.targets[target]
        self.proxy = sensitivity
        self.w = self._nominal_weights()
        self._cumw = np.cumsum(self.w)                   # cache for fast categorical draws

    def _to_full(self, series):
        """Scatter a per-id target column into a full-nconf array (0 on missing ids)."""
        arr = np.zeros(self.nconf)
        arr[series.index.values] = np.nan_to_num(series.to_numpy(), nan=0.0)
        return arr

    def draw_nominal(self, n, rng):
        """iid nominal draws via the cached cumulative (no O(nconf) work per call).

        Inverse-CDF sampling from the same weights as the base categorical, so the law is
        identical; only the per-call cost drops, which matters across the full sweep."""
        idx = np.minimum(np.searchsorted(self._cumw, rng.random(n)), self.nconf - 1)
        return self.to_binary(idx)


class Evaluator:
    """Wraps the S lookup and counts solver calls (the shared budget currency)."""

    def __init__(self, S):
        self.S = S
        self.calls = 0
        self.seen = set()

    def __call__(self, ids):
        self.calls += len(ids)
        self.seen.update(ids.tolist())
        return self.S[ids]
