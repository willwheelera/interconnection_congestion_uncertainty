"""Convert the ACTIVSg2000 synthetic Texas model to EGRET JSON.

Parses the MATPOWER case with EGRET, then attaches per-bus geographic
coordinates (latitude/longitude) and ERCOT weather zone read from the PowerWorld
.aux file. Coordinates match queue projects to the nearest bus; the zone assigns
each bus to one of the eight ERCOT weather zones for per-zone load scaling. The
model's eight areas are exactly the ERCOT weather zones.

The MATPOWER bus ids equal the .aux BusNum, so both are keyed directly. A
load-mismatch cost is set so the feasibility slack is well defined. Emergency
thermal ratings and an N-1 single-branch-outage contingency set (non-bridge
branches only) are added so the security-constrained DCOPF has binding
post-contingency constraints.

Usage:
    uv run python src/texas/convert_model.py
"""

import shlex
from collections import Counter
from pathlib import Path

import networkx as nx
from egret.parsers.matpower_parser import create_ModelData

REPO_ROOT = Path(__file__).parents[2]
MODEL_DIR = REPO_ROOT / "texas_model" / "ACTIVSg2000"
CASE_M = MODEL_DIR / "case_ACTIVSg2000.m"
AUX = MODEL_DIR / "ACTIVSg2000.aux"
OUT = REPO_ROOT / "data" / "activsg2000.json"

# Column positions of BusNum, AreaNum, per-bus Latitude:1, Longitude:1 in the
# .aux "DATA (Bus, [...])" record (whitespace-delimited, quoted name is one token).
BUS_NUM_COL = 0
BUS_AREA_COL = 9
BUS_LAT_COL = 14
BUS_LON_COL = 15

# ACTIVSg2000 area number -> ERCOT weather zone, named to match the columns of
# the actual-load feed (/np6-345-cd/act_sys_load_by_wzn).
AREA_TO_ZONE = {1: "FarWest", 2: "North", 3: "West", 4: "Southern",
                5: "NorthC", 6: "SouthC", 7: "Coast", 8: "East"}

# Post-contingency (emergency) thermal rating as a multiple of the normal rating.
# The MATPOWER data has no emergency rating, so without this no N-1 constraint
# ever binds. 1.5 keeps the base configuration N-1 secure (feasible) across all
# load snapshots while queue-generator additions still drive post-contingency
# congestion -- the sensitivity of interest.
EMERGENCY_FACTOR = 1.5


def parse_bus_geo(aux_path: Path) -> dict[str, tuple[float, float, str]]:
    """Return {bus_num: (lat, lon, zone)} from the .aux DATA (Bus, ...) block."""
    lines = aux_path.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("DATA (Bus,"))
    open_brace = next(i for i in range(start, len(lines)) if lines[i].strip() == "{")
    geo = {}
    for ln in lines[open_brace + 1:]:
        if ln.strip() == "}":
            break
        tok = shlex.split(ln)
        zone = AREA_TO_ZONE[int(tok[BUS_AREA_COL])]
        geo[tok[BUS_NUM_COL]] = (float(tok[BUS_LAT_COL]), float(tok[BUS_LON_COL]), zone)
    return geo


def attach_geo(md, geo: dict[str, tuple[float, float, str]]) -> None:
    """Store latitude/longitude and weather zone on each bus element."""
    for bus_id, bus in md.data["elements"]["bus"].items():
        lat, lon, zone = geo[bus_id]
        bus["lat"] = lat
        bus["lon"] = lon
        bus["zone"] = zone


def add_emergency_ratings(md) -> None:
    """Set each branch's post-contingency rating to EMERGENCY_FACTOR x normal."""
    for branch in md.data["elements"]["branch"].values():
        branch["rating_emergency"] = EMERGENCY_FACTOR * branch["rating_long_term"]


def build_contingencies(md) -> dict:
    """Return an N-1 single-branch-outage contingency per non-bridge branch.

    A branch whose removal disconnects the network (a bridge with no parallel
    circuit) is skipped: its outage merely islands load rather than posing a
    security constraint, and EGRET rejects it.
    """
    branches = md.data["elements"]["branch"]
    graph = nx.Graph()
    pair_count = Counter()
    for b in branches.values():
        pair = frozenset((b["from_bus"], b["to_bus"]))
        pair_count[pair] += 1
        graph.add_edge(b["from_bus"], b["to_bus"])
    bridges = {frozenset(e) for e in nx.bridges(graph)}
    cont = {}
    for name, b in branches.items():
        pair = frozenset((b["from_bus"], b["to_bus"]))
        if pair_count[pair] == 1 and pair in bridges:
            continue
        cont[name] = {"branch_contingency": name}
    return cont


def main():
    md = create_ModelData(str(CASE_M))
    geo = parse_bus_geo(AUX)
    attach_geo(md, geo)
    add_emergency_ratings(md)
    md.data["elements"]["contingency"] = build_contingencies(md)
    # Value of lost load (~ERCOT's system-wide offer cap), so the DCOPF
    # feasibility slack prices unavoidable shed load realistically instead of
    # swamping congestion duals with a huge penalty.
    md.data["system"]["load_mismatch_cost"] = 9000.0
    OUT.parent.mkdir(exist_ok=True)
    md.write(str(OUT))
    n = md.data["elements"]
    print(f"wrote {OUT}")
    print(f"  buses {len(n['bus'])}  branches {len(n['branch'])}  "
          f"generators {len(n['generator'])}  loads {len(n['load'])}  "
          f"contingencies {len(n['contingency'])}")


if __name__ == "__main__":
    main()
