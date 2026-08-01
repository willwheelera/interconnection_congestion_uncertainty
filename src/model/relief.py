"""ModelData edits that turn a bare case into the priced, snapshot-scaled solve model.

Every one of these mutates a `ModelData` in place before the OPF is built, and none of
them knows which case family it is working on -- `model.scdcopf` composes them in the
order the recipe requires:

    set_emergency_factor -> add_queue_generators -> scale_loads_by_zone
    -> scale_renewables_by_snapshot -> add_lostload_generators_ordc

The two scaling steps are no-ops when the case has no load snapshots (the IEEE queue
testbeds run at a single load level), so the same sequence serves both families.

Nodal relief is the piece worth understanding.  EGRET's single system-wide slack cannot
represent a pocket that is unservable only under one contingency; without a local outlet
such a case goes infeasible instead of pricing.  A relief generator at each load bus can
inject up to that bus's demand, so the pocket is relieved locally and the relief price
becomes its price ceiling.  Pricing that relief on an ORDC-shaped rising curve rather
than one flat VOLL is what makes mu spread over a range instead of pinning at a single
level -- see notes/ercot_price_formation.md.
"""

from egret.data.model_data import ModelData


def linear_cost(marginal: float) -> dict:
    """Polynomial cost curve with the given linear ($/MWh) coefficient."""
    return {"data_type": "cost_curve", "cost_curve_type": "polynomial",
            "values": {"0": 0.0, "1": marginal, "2": 0.0}}


def set_emergency_factor(md: ModelData, factor: float) -> None:
    """Override every branch's post-contingency rating to factor x its long-term rating.

    Applied at model-build time rather than baked into the case JSON, so a case file can
    be re-solved at a different factor without being rewritten.  A case read without this
    override solves at whatever `rating_emergency` the JSON carries.
    """
    for b in md.data["elements"]["branch"].values():
        b["rating_emergency"] = factor * b["rating_long_term"]


def add_queue_generators(md: ModelData, candidates: list[dict], config) -> None:
    """Inject the candidates whose bit is 1 as linear-cost generators at their buses.

    Candidates come from `cases.registry.Case.candidates()`, so they already carry a
    uniform `id` and `cost` whatever family they came from.  A candidate that names a
    fuel is tagged with it, which is what lets scale_renewables_by_snapshot derate it.
    """
    gens = md.data["elements"]["generator"]
    for on, c in zip(config, candidates):
        if not on:
            continue
        gen = {"bus": c["bus"], "generator_type": "thermal", "in_service": True,
               "p_min": 0.0, "p_max": float(c["capacity_mw"]), "pg": 0.0,
               "p_cost": linear_cost(c["cost"])}
        if c.get("fuel"):
            gen["fuel_group"] = c["fuel"]
        gens[f"queue_{c['id']}"] = gen


def add_lostload_generators(md: ModelData, cost: float, buses=None) -> None:
    """Relief generator at each load bus (or only `buses`), at one flat price.

    Call after scale_loads_by_zone so p_max tracks the scaled demand.
    """
    gens = md.data["elements"]["generator"]
    for name, load in md.data["elements"]["load"].items():
        if buses is not None and load["bus"] not in buses:
            continue
        gens[f"lostload_{name}"] = {
            "bus": load["bus"], "generator_type": "thermal", "in_service": True,
            "p_min": 0.0, "p_max": float(load["p_load"]), "pg": 0.0,
            "p_cost": linear_cost(cost),
        }


def ordc_relief_curve(cap: float, c_min: float, c_max: float, n_blocks: int) -> dict:
    """Convex piecewise shed-cost curve rising c_min -> c_max over [0, cap].

    Emulates the ERCOT ORDC scarcity shape: cheap for a shallow shed, climbing
    geometrically toward VOLL (c_max) for a deep one, so the marginal block that clears
    sets a range of local prices instead of one fixed level.  `values` are
    (MW, cumulative $) breakpoints; block slopes are the marginal shed price.
    """
    edges = [cap * k / n_blocks for k in range(n_blocks + 1)]
    mc = [c_min * (c_max / c_min) ** (k / (n_blocks - 1)) for k in range(n_blocks)]
    values, cum = [(0.0, 0.0)], 0.0
    for k in range(n_blocks):
        cum += mc[k] * (edges[k + 1] - edges[k])
        values.append((edges[k + 1], cum))
    return {"data_type": "cost_curve", "cost_curve_type": "piecewise", "values": values}


def add_lostload_generators_ordc(md: ModelData, c_min: float, c_max: float,
                                 n_blocks: int, buses=None) -> None:
    """Relief generator at each load bus (or only `buses`) on an ORDC-shaped curve.

    Keep c_max below the system slack price so nodal relief still beats EGRET's
    system-wide slack.  Call after scale_loads_by_zone.
    """
    gens = md.data["elements"]["generator"]
    for name, load in md.data["elements"]["load"].items():
        if buses is not None and load["bus"] not in buses:
            continue
        cap = float(load["p_load"])
        if cap <= 0.0:
            continue
        gens[f"lostload_{name}"] = {
            "bus": load["bus"], "generator_type": "thermal", "in_service": True,
            "p_min": 0.0, "p_max": cap, "pg": 0.0,
            "p_cost": ordc_relief_curve(cap, c_min, c_max, n_blocks),
        }


def line_endpoint_buses(md: ModelData, lines) -> set:
    """From/to buses of the given branch names -- the pockets to target relief at."""
    branches = md.data["elements"]["branch"]
    return {b for ln in lines for b in (branches[ln]["from_bus"], branches[ln]["to_bus"])}


def scale_loads_by_zone(md: ModelData, zone_scales: dict[str, float] | None) -> None:
    """Scale each bus load by its weather zone's snapshot scale factor."""
    if zone_scales is None:
        return
    buses = md.data["elements"]["bus"]
    for load in md.data["elements"]["load"].values():
        load["p_load"] *= zone_scales[buses[load["bus"]]["zone"]]


def scale_renewables_by_snapshot(md: ModelData, cf: dict[str, float] | None) -> None:
    """Derate tagged renewable generators to the snapshot's per-fuel capacity factor.

    Acts only on generators carrying a `fuel_group` listed in `cf`, so untagged base
    units and non-listed fuels (e.g. storage) are left alone.
    """
    if cf is None:
        return
    for gen in md.data["elements"]["generator"].values():
        group = gen.get("fuel_group")
        if group in cf:
            gen["p_max"] *= cf[group]
