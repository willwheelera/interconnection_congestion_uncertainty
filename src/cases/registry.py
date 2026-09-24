"""The single registry of testbed cases -- every stage reads its case facts from here.

One `Case` record holds everything that distinguishes one testbed from another: its
physical model, where its inputs and caches live, and the modeling recipe it is solved
under.  Every entry point declares its own `--case` flag and expands the value with
`get_cases` (a case name, a substring of one, or `all`) or `get` (one exact case); each
reads paths off the record, so no script hardcodes a filename and no case needs custom
code in a downstream stage.

Three kinds of testbed, differing only in how the case is constructed and evaluated;
each kind shares a name stem, so a `--case` substring selects the whole kind:

  *_gen_per_bus  the legacy DCOPF testbeds (a generator at every load bus, no incumbent
                 fleet).  Enumerated as one solution file per config, reduced to a wide
                 dual_cache.
  activsg2000_*  ACTIVSg2000 under an N-1 SCDCOPF, evaluated over a collection of ERCOT
                 load snapshots (so its sweeps parallelize over snapshots and average mu);
                 the `*_v12_lev*` entries are leveraged-queue variants of one of these.
  case*_queue    IEEE 14/30/118 with the original fleet as incumbents and a queue on top,
                 same SCDCOPF recipe as ACTIVSg2000 but a single load level (so its
                 sweeps parallelize over configurations).

Each case's own inputs -- queue, screened case, relief set, contingencies, and nominal
laws -- live together in `data/<case>/`; the shared physical model (the network, the
snapshot collection, and the queue shared by the incumbent-fleet cases) stays flat in
`data/`, and outputs go under `results/<case>/`.  `paths` overrides an entry for a shared
or legacy file (the network, or a pre-convention name).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.cases.candidate_clustering import compact_groups
from src.model.relief import ALL_LOAD_BUSES
from src.paths import REPO_ROOT

DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"

# Representative marginal cost ($/MWh) by Texas queue fuel; the queue file records a
# fuel rather than a cost.  Renewables and storage discharge are zero-cost, thermal
# costs sit above the existing fleet's ~$17-19 so new thermal runs only when cheaper
# units are exhausted.
FUEL_COST = {"Solar": 0.0, "Wind": 0.0, "Battery": 0.0, "Hydro": 0.0,
             "Nuclear": 8.0, "Gas": 25.0, "Oil": 40.0, "Other": 30.0}


@dataclass(frozen=True)
class Recipe:
    """The modeling constants that make the mu distribution realistic.

    Derived in notes/ercot_price_formation.md: tighten the post-contingency rating to
    create scarcity, price nodal shed on a rising ORDC-shaped curve so mu spreads over
    a range instead of pinning at one VOLL, and cap every dual at ERCOT's Max Shadow
    Price via soft thermal limits.  Price ordering is overload < shed < system slack.
    """

    emergency_factor: float = 1.2       # post-contingency limit = factor x long_term
    ordc_min: float = 500.0             # nodal relief price floor ($/MWh)
    ordc_max: float = 5000.0            # nodal relief price ceiling
    ordc_blocks: int = 8                # relief piecewise blocks
    shadow_price_cap: float = 3500.0    # soft-limit mu cap ($/MWh)
    slack_cost: float = 9000.0          # EGRET system-wide slack, above the shed curve


SCDCOPF_RECIPE = Recipe()


@dataclass(frozen=True)
class Case:
    """One testbed: its physical model, cache layout, and modeling recipe."""

    name: str
    model: str                  # "dcopf" | "scdcopf"
    ngen: int                   # queue candidates == configuration bits
    ingest: str                 # "wide" | "long" -- how the sweep output is reduced
    parallel: str               # "config" | "snapshot" -- the sweep's parallel axis
    one_mu: str | None = None   # single-element target; None picks the most congested
    recipe: Recipe | None = SCDCOPF_RECIPE   # None for the plain-DCOPF cases
    p_default: float | None = None           # build probability when there is no queue file
    proxy: bool = False                      # a sensitivity_cache exists for ascent moves
    sample_only: bool = False                # too large to enumerate: MCMC-sampled, no cache
    build: dict | None = None                # construction knobs, used only by cases.ieee
    paths: dict = field(default_factory=dict)   # overrides for pre-convention layouts

    # -- inputs ------------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        """The per-case directory holding this case's own inputs -- queue, screened case,
        relief set, contingencies, laws.  The shared physical model stays flat in data/."""
        return DATA / self.name

    @property
    def case_file(self) -> Path:
        """The full case: every N-1 contingency, no screening (shared, flat in data/)."""
        return self.paths.get("case", DATA / f"{self.name}.json")

    @property
    def screened_file(self) -> Path:
        """The case restricted to the contingencies that ever bind."""
        return self.paths.get("screened", self.data_dir / "screened.json")

    @property
    def contingency_file(self) -> Path:
        """The kept-contingency name list written alongside the screened case."""
        return self.paths.get("contingencies", self.data_dir / "contingencies.json")

    def candidates_path(self) -> Path:
        """The queue file's path whether or not it exists (candidates_file guards existence)."""
        return self.paths.get("candidates", self.data_dir / "queue.json")

    @property
    def candidates_file(self) -> Path | None:
        """The queue file defining the candidates and their build probabilities."""
        p = self.candidates_path()
        return p if Path(p).exists() else None

    def relief_path(self) -> Path:
        """The relief-set file's path whether or not it exists (relief_file guards existence)."""
        return self.paths.get("relief", self.data_dir / "relief_set.json")

    @property
    def relief_file(self) -> Path | None:
        """The slim ORDC relief bus set, when one has been selected and validated."""
        p = self.relief_path()
        return p if Path(p).exists() else None

    @property
    def snapshots_file(self) -> Path | None:
        """The load/renewable snapshot collection; None for single-load-level cases."""
        return self.paths.get("snapshots")

    def solve_file(self, screened=True) -> Path:
        """The case a solve should read: the screened one when asked for and present."""
        return self.screened_file if screened and self.screened_file.exists() else self.case_file

    # -- outputs -----------------------------------------------------------------

    @property
    def results_dir(self) -> Path:
        return self.paths.get("results", RESULTS / self.name)

    @property
    def mu_cache(self) -> Path:
        """Long-format sweep output: one part file per chunk of (id, constraint, mu)."""
        return self.paths.get("mu_cache", self.results_dir / "mu_cache")

    @property
    def snapmax_cache(self) -> Path:
        """Per (config, snapshot): peak mu, and the overload and shed relief had to buy.

        Written by every sweep, in one schema.  A single-load-level case has one row per
        configuration with `snapshot` null; a snapshot-parallel case has one row per
        (configuration, snapshot).
        """
        return self.results_dir / "snapmax_cache"

    @property
    def lmp_cache(self) -> Path | None:
        """Congestion-LMP spread per configuration; only where reconstruct_lmp has run."""
        p = self.results_dir / "lmp_cache.parquet"
        return p if p.exists() else None

    @property
    def dual_cache(self) -> Path:
        """Wide per-element duals, indexed by configuration id.

        Column groups are a 2-level MultiIndex.  The `mu` group is always present -- it is
        what `sampling.problem.Problem.mu_table` reads.  A sweep that recorded the full
        solution adds more (`flow`, `p_gen`, `lmp`, `va`, `nu`, `load`, `mu_pmax`,
        `mu_pmin`); `dual_cache_groups()` reports which a case actually has.
        """
        return self.results_dir / "dual_cache.parquet"

    def dual_cache_groups(self) -> set:
        """The column groups this case's dual_cache carries (empty if it has none).

        A pandas MultiIndex is flattened to "('group', 'name')" strings in the parquet
        schema, so the group is the first element of each parsed tuple.
        """
        import ast
        import pyarrow.parquet as pq
        if not self.dual_cache.exists():
            return set()
        groups = set()
        for name in pq.read_schema(self.dual_cache).names:
            if name.startswith("("):
                groups.add(ast.literal_eval(name)[0])
        return groups

    @property
    def target_cache(self) -> Path:
        """The common per-config frame every case is reduced to.

        Columns: max_mu, sum_mu, one_mu, one_lmp, n_on, max_mu_single, driver.  `driver`
        names the constraint attaining max_mu, which is why there is no separate S cache.
        """
        return self.results_dir / "target_cache.parquet"

    def descriptor_cache(self, base="off") -> Path:
        """Per-constraint static descriptors at a chosen base configuration."""
        return self.results_dir / f"static_descriptors_{base}.parquet"

    # -- queue -------------------------------------------------------------------

    def candidates(self) -> list[dict]:
        """The queue candidates in one schema, whatever the source file looks like.

        Normalizes the Texas queue's `inr` key and fuel-implied cost onto the same
        `id` / `cost` fields the IEEE queue files carry, so a solve never branches on
        the case family to name a generator or price it.
        """
        if self.candidates_file is None:
            raise ValueError(f"{self.name} has no queue file; it is not a queue case")
        raw = json.loads(self.candidates_file.read_text())["candidates"]
        out = []
        for c in raw:
            fuel = c.get("fuel")
            out.append({"id": c.get("id", c.get("inr")),
                        "bus": c["bus"],
                        "capacity_mw": float(c["capacity_mw"]),
                        "cost": float(c["cost"]) if "cost" in c else FUEL_COST[fuel],
                        "p": float(c["p"]),
                        "fuel": fuel,
                        "lat": c.get("county_lat"),
                        "lng": c.get("county_lng"),
                        "name": c.get("name")})
        return out

    def candidate_groups(self) -> list[list[int]] | None:
        """Candidate indices in the blocks the coupled laws couple within.

        Geographically compact groups of about five within each fuel
        (`candidate_clustering.compact_groups`) when the queue carries county coordinates, so
        a large fuel is not one strong block; one block per fuel otherwise.  None when the
        case has no queue, which `make_dist` reads as a single block.
        """
        if self.candidates_file is None:
            return None
        cands = self.candidates()
        if any(c["lat"] is None for c in cands):
            groups: dict = {}
            for i, c in enumerate(cands):
                groups.setdefault(c["fuel"], []).append(i)
            return list(groups.values())
        return compact_groups(cands)

    def build_p(self) -> np.ndarray:
        """Per-candidate build probability as a length-ngen vector, for drawing configs."""
        if self.candidates_file is None:
            return np.full(self.ngen, self.p_default)
        return np.array([c["p"] for c in self.candidates()])

    def nominal_p(self):
        """The marginals spec the nominal law is built from -- vector or scalar.

        A queue case supplies calibrated per-candidate probabilities and they are used as
        given.  A case without a queue supplies a scalar mean, which `make_dist` spreads
        into per-generator marginals with a Beta; returning a filled vector instead would
        silently drop that spread and change the law.
        """
        return self.build_p() if self.candidates_file else self.p_default

    def relief_buses(self) -> set | str:
        """The buses that get ORDC relief; ALL_LOAD_BUSES until a set has been selected."""
        if self.relief_file is None:
            return ALL_LOAD_BUSES
        return set(json.loads(self.relief_file.read_text()))

    def snapshots(self) -> list[dict]:
        """The load snapshots to average over; a single implicit snapshot if none."""
        if self.snapshots_file is None:
            return [None]
        return json.loads(Path(self.snapshots_file).read_text())["snapshots"]


# The Texas cases share one queue and one snapshot collection.
_TX_QUEUE = DATA / "activsg2000_queue20.json"
_TX_SNAPSHOTS = DATA / "activsg2000_load_collection.json"


def _texas(name, one_mu, case_file, **paths) -> Case:
    return Case(name=name, model="scdcopf", ngen=20,
                ingest="long", parallel="snapshot", one_mu=one_mu,
                paths={"case": case_file, "candidates": _TX_QUEUE,
                       "snapshots": _TX_SNAPSHOTS, **paths})


def _leveraged(tag, ngen=20, sample_only=False) -> Case:
    """One activsg2000_original_v12 variant differing only in which queue sits on top.

    Same network, fleet, and snapshots as v12, so the sweeps compare directly.  The
    relief set and the screened contingency set are *not* inherited from v12: both
    depend on which buses the queue sits at, so each variant selects and screens its
    own.  `one_mu` is unset because the most congested element is not known until the
    case is swept.  `ngen` defaults to the 20-candidate draw; a larger draw (the q28
    variant) passes it explicitly.  The queue is the case's own input at
    `data/<case>/queue.json`.  `sample_only` marks the full-queue variant, whose 2^ngen
    space is too large to enumerate and is estimated by MCMC instead.
    """
    name = f"activsg2000_v12_{tag}"
    return Case(name=name, model="scdcopf", ngen=ngen,
                ingest="long", parallel="snapshot", one_mu=None, sample_only=sample_only,
                paths={"case": DATA / "activsg2000.json", "snapshots": _TX_SNAPSHOTS})


def _ieee(name, network, ngen, one_mu) -> Case:
    return Case(name=name, model="scdcopf", ngen=ngen,
                ingest="long", parallel="config", one_mu=one_mu,
                build=dict(network=network, ncand=ngen, base_margin=1.05))


CASES: dict[str, Case] = {
    # Legacy DCOPF testbeds: a generator at every load bus, no incumbent fleet, so the
    # low-n_on part of the space is served by priced shed rather than by a grid.  Kept
    # for continuity with the earlier results; not re-run.  case14 line 6 is the
    # interesting single-element target (P[mu > 100] ~ 5%); its most-congested line is
    # far rarer.
    "case14_gen_per_bus": Case(
        name="case14_gen_per_bus", model="dcopf", ngen=14,
        ingest="wide", parallel="config", one_mu="6", recipe=None,
        p_default=0.65, proxy=True),
    "case30_gen_per_bus": Case(
        name="case30_gen_per_bus", model="dcopf", ngen=20,
        ingest="wide", parallel="config", one_mu="27", recipe=None,
        p_default=0.65, proxy=True),

    # ACTIVSg2000.  "original" is the historical run at the case JSON's native 1.5
    # emergency factor, kept only for its enumerated cache, which predates the
    # results/<case>/ convention.  The other two are the 2016 and 2025 incumbent fleets
    # under the current factor-1.2 soft-cap recipe.
    "activsg2000_original": _texas(
        "activsg2000_original", "base:9", DATA / "activsg2000.json",
        screened=DATA / "activsg2000_screened.json",
        contingencies=DATA / "activsg2000_contingencies_screened.json"),
    "activsg2000_2025": _texas(
        "activsg2000_2025", "ctg:464:461", DATA / "activsg2000_2025.json"),
    "activsg2000_original_v12": _texas(
        "activsg2000_original_v12", "ctg:68:18", DATA / "activsg2000.json"),

    # Leveraged-queue variants of v12, drawn by cases.texas.build_leveraged_queue: the
    # candidates sit only at buses with shift-factor leverage on a priced constraint,
    # five independent draws (lev0-4) plus the same draws with three nearby projects
    # aggregated into one candidate (lev0a3-4a3).  Everything except the queue is v12's.
    **{f"activsg2000_v12_{tag}": _leveraged(tag)
       for i in range(5) for tag in (f"lev{i}", f"lev{i}a3")},

    # A larger aggregated leveraged draw: seven candidates per fuel instead of five, so
    # 28 configuration bits (2^28 configurations) rather than 20.  Same v12 network,
    # fleet, snapshots, and construction as lev0a3 -- only the per-fuel quota changed.
    "activsg2000_v12_q28_lev0a3": _leveraged("q28_lev0a3", ngen=28),

    # The full leveraged pool: every candidate that clears the shift-factor screen, no
    # per-fuel quota and no aggregation (195 candidates, blocks Wind 25 / Solar 51 /
    # Battery 95 / Gas 24).  2^195 is unenumerable, so this case is sampled by MCMC under
    # a coupled MRF nominal law (sampling.mcmc_sweep), never swept -- sample_only guards
    # the 1<<ngen enumeration paths against it.
    "activsg2000_v12_levfull": _leveraged("levfull", ngen=195, sample_only=True),

    # Incumbent-fleet IEEE queue testbeds: the network's own generators stay online and
    # a queue is added on top, under the same recipe as Texas so the five cases are
    # comparable.  case118's one_mu is unset -- its scalar max_mu is nearly degenerate,
    # so its single element must be picked from the per-line tail, not the default
    # most-congested rule.
    "case14_queue": _ieee("case14_queue", "case14", 14, "ctg:10:14"),
    "case30_queue": _ieee("case30_queue", "case30", 20, "ctg:10:41"),
    "case118_queue": _ieee("case118_queue", "case118", 20, None),
}


def get(name: str) -> Case:
    """The one registered case with this exact name (single-case entry points)."""
    return CASES[name]


def get_cases(value: str) -> list[Case]:
    """The cases a `--case` value names: one case, a name substring, or every case.

    An exact registered case name selects just that case, so a name that happens to be a
    substring of a longer name still selects only itself.  "all" selects every case.  Any
    other string selects every case whose name contains it, which is how the naming
    convention doubles as the grouping: `lev` selects the leveraged variants, `a3` the
    aggregated ones, `queue` the IEEE cases, `activsg2000` every Texas case.  A value
    matching no name raises.  This is the only place a `--case` value is interpreted, so a
    reader consults just this function to know everything the flag accepts.
    """
    if value == "all":
        return list(CASES.values())
    if value in CASES:
        return [CASES[value]]
    matches = [c for c in CASES.values() if value in c.name]
    if matches:
        return matches
    raise ValueError(f"unknown --case {value!r}: expected 'all', a case name, or a "
                     f"substring of one; names are {sorted(CASES)}")


def get_case(value) -> Case:
    """The single registered case a `--case` value names (exact name or unique substring).

    The one-case partner of `get_cases`, for the entry points -- most solves, every per-case
    figure -- that act on exactly one case: it interprets the value the same way but requires
    the match to be unique, so an ambiguous substring is an error rather than a silent pick.
    An already-resolved Case passes through unchanged, so a caller may hold either.
    """
    if isinstance(value, Case):
        return value
    matches = get_cases(value)
    if len(matches) != 1:
        raise ValueError(f"--case {value!r} names {[c.name for c in matches]}; name one case")
    return matches[0]

if __name__ == "__main__":
    for k in CASES:
        print(k)
