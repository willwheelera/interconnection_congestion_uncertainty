"""The single registry of testbed cases -- every stage reads its case facts from here.

One `Case` record holds everything that distinguishes one testbed from another: which
family it belongs to, its physical model, where its inputs and caches live, and the
modeling recipe it is solved under.  Every entry point takes `--case <name>` (see
`add_case_arg`), looks the record up with `get`, and reads paths off it, so no script
hardcodes a filename and no case needs custom code in a downstream stage.

Three families, differing only in how the case is constructed and evaluated:

  gen_per_bus  the legacy DCOPF testbeds (a generator at every load bus, no incumbent
               fleet).  Enumerated as one solution file per config, reduced to a wide
               dual_cache.
  texas        ACTIVSg2000 under an N-1 SCDCOPF, evaluated over a collection of ERCOT
               load snapshots (so its sweeps parallelize over snapshots and average mu).
  ieee_queue   IEEE 14/30/118 with the original fleet as incumbents and a queue on top,
               same SCDCOPF recipe as texas but a single load level (so its sweeps
               parallelize over configurations).

Paths follow the `data/<case>*.json` and `results/<case>/` convention and are derived
from the case name; `paths` overrides an entry for the cases whose files predate it.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from paths import REPO_ROOT
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
    """One testbed: its family, physical model, cache layout, and modeling recipe."""

    name: str
    family: str                 # "gen_per_bus" | "texas" | "ieee_queue"
    model: str                  # "dcopf" | "scdcopf"
    ngen: int                   # queue candidates == configuration bits
    ingest: str                 # "wide" | "long" -- how the sweep output is reduced
    parallel: str               # "config" | "snapshot" -- the sweep's parallel axis
    one_mu: str | None = None   # single-element target; None picks the most congested
    recipe: Recipe | None = SCDCOPF_RECIPE   # None for the plain-DCOPF cases
    p_default: float | None = None           # build probability when there is no queue file
    proxy: bool = False                      # a sensitivity_cache exists for ascent moves
    build: dict | None = None                # construction knobs, used only by cases.ieee
    paths: dict = field(default_factory=dict)   # overrides for pre-convention layouts

    # -- inputs ------------------------------------------------------------------

    @property
    def case_file(self) -> Path:
        """The full case: every N-1 contingency, no screening."""
        return self.paths.get("case", DATA / f"{self.name}.json")

    @property
    def screened_file(self) -> Path:
        """The case restricted to the contingencies that ever bind."""
        return self.paths.get("screened", DATA / f"{self.name}_screened.json")

    @property
    def contingency_file(self) -> Path:
        """The kept-contingency name list written alongside the screened case."""
        return self.paths.get("contingencies",
                              DATA / f"{self.name}_contingencies_screened.json")

    @property
    def candidates_file(self) -> Path | None:
        """The queue file defining the candidates and their build probabilities."""
        p = self.paths.get("candidates", DATA / f"{self.name}_candidates.json")
        return p if p and Path(p).exists() else None

    @property
    def relief_file(self) -> Path | None:
        """The slim ORDC relief bus set, when one has been selected and validated."""
        p = self.paths.get("relief", DATA / f"{self.name}_relief_set.json")
        return p if p and Path(p).exists() else None

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
    def snapmax_cache(self) -> Path | None:
        """Per-snapshot peak mu, overload, and shed; snapshot-parallel cases only."""
        if self.parallel != "snapshot":
            return None
        return self.paths.get("snapmax", self.results_dir / "snapmax_cache")

    @property
    def stress_cache(self) -> Path | None:
        """Per-config overload and shed; the config-parallel cases' snapmax equivalent."""
        if self.parallel != "config":
            return None
        return self.results_dir / "stress_cache"

    @property
    def lmp_cache(self) -> Path | None:
        p = self.paths.get("lmp", self.results_dir / "lmp_cache.parquet")
        return p if p and Path(p).exists() else None

    @property
    def dual_cache(self) -> Path:
        """Wide per-config duals; the sampling Problem reads this."""
        return self.results_dir / "dual_cache.parquet"

    @property
    def target_cache(self) -> Path:
        """The common per-config target frame every case is reduced to."""
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
                        "name": c.get("name")})
        return out

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

    def relief_buses(self) -> set | None:
        """The buses that get ORDC relief; None means every load bus."""
        if self.relief_file is None:
            return None
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
    return Case(name=name, family="texas", model="scdcopf", ngen=20,
                ingest="long", parallel="snapshot", one_mu=one_mu,
                paths={"case": case_file, "candidates": _TX_QUEUE,
                       "snapshots": _TX_SNAPSHOTS, **paths})


def _ieee(name, network, ngen, one_mu) -> Case:
    return Case(name=name, family="ieee_queue", model="scdcopf", ngen=ngen,
                ingest="long", parallel="config", one_mu=one_mu,
                build=dict(network=network, ncand=ngen, base_margin=1.05))


CASES: dict[str, Case] = {
    # Legacy DCOPF testbeds: a generator at every load bus, no incumbent fleet, so the
    # low-n_on part of the space is served by priced shed rather than by a grid.  Kept
    # for continuity with the earlier results; not re-run.  case14 line 6 is the
    # interesting single-element target (P[mu > 100] ~ 5%); its most-congested line is
    # far rarer.
    "case14_gen_per_bus": Case(
        name="case14_gen_per_bus", family="gen_per_bus", model="dcopf", ngen=14,
        ingest="wide", parallel="config", one_mu="6", recipe=None,
        p_default=0.65, proxy=True),
    "case30_gen_per_bus": Case(
        name="case30_gen_per_bus", family="gen_per_bus", model="dcopf", ngen=20,
        ingest="wide", parallel="config", one_mu="27", recipe=None,
        p_default=0.65, proxy=True),

    # ACTIVSg2000.  "original" is the historical run at the case JSON's native 1.5
    # emergency factor, kept only for its enumerated cache, which predates the
    # results/<case>/ convention.  The other two are the 2016 and 2025 incumbent fleets
    # under the current factor-1.2 soft-cap recipe.
    "activsg2000_original": _texas(
        "activsg2000_original", "base:9", DATA / "activsg2000.json",
        screened=DATA / "activsg2000_screened.json",
        contingencies=DATA / "activsg2000_contingencies_screened.json",
        mu_cache=RESULTS / "activsg2000" / "original_case_source" / "mu_cache",
        lmp=RESULTS / "activsg2000" / "original_case_source" / "lmp_cache_old.parquet"),
    "activsg2000_2025": _texas(
        "activsg2000_2025", "ctg:464:461", DATA / "activsg2000_2025.json"),
    "activsg2000_original_v12": _texas(
        "activsg2000_original_v12", "ctg:68:18", DATA / "activsg2000.json"),

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
    """The registered case of that name."""
    return CASES[name]


def add_case_arg(parser: argparse.ArgumentParser, default=None, allow_all=False) -> None:
    """Add the shared `--case` argument every entry point takes."""
    choices = list(CASES) + (["all"] if allow_all else [])
    parser.add_argument("--case", default=default, choices=choices, required=default is None,
                        help="registered case name")


def resolve(name: str, family: str | None = None) -> list[Case]:
    """Cases named by `--case`, where "all" means every case (of `family`, if given)."""
    if name != "all":
        return [get(name)]
    return [c for c in CASES.values() if family is None or c.family == family]
