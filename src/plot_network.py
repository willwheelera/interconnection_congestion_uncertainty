"""Plot an EGRET JSON network using pandapower and matplotlib."""

import argparse
import json
import numpy as np

import geojson
import matplotlib.pyplot as plt
import networkx as nx
import pandapower as pp
import pandapower.plotting as pplot
import pandapower.topology as ptopo


def load_egret_json(path):
    with open(path) as f:
        return json.load(f)


def _scalar(v, default=0.0):
    """Extract a scalar from an EGRET value that may be a time-series dict."""
    if isinstance(v, dict) and v.get("data_type") == "time_series":
        return v["values"][0]
    if v is None:
        return default
    return v


def _cumulative(v):
    """Sum a time-series dict's values, or return the scalar."""
    if isinstance(v, dict) and v.get("data_type") == "time_series":
        return sum(v["values"])
    if v is None:
        return 0.0
    return v


def compute_bus_energy(egret_data):
    """Cumulative energy consumption (MWh, assuming hourly) per bus name."""
    energy = {name: 0.0 for name in egret_data["elements"].get("bus", {})}
    for load in egret_data["elements"].get("load", {}).values():
        energy[load["bus"]] += _cumulative(load.get("p_load"))
    return energy


def build_pandapower_network(egret_data):
    elements = egret_data["elements"]
    base_mva = egret_data["system"].get("baseMVA", 100.0)

    net = pp.create_empty_network(sn_mva=base_mva)

    bus_map = {}
    for name, bus in elements.get("bus", {}).items():
        vn_kv = bus.get("base_kv", 1.0)
        idx = pp.create_bus(net, vn_kv=vn_kv, name=name)
        bus_map[name] = idx

    for name, load in elements.get("load", {}).items():
        pp.create_load(
            net,
            bus=bus_map[load["bus"]],
            p_mw=_scalar(load.get("p_load"), 0.0),
            q_mvar=_scalar(load.get("q_load"), 0.0),
            name=name,
        )

    for name, gen in elements.get("generator", {}).items():
        p_mw = _scalar(gen.get("pg", gen.get("p_min", 0.0)))
        pp.create_gen(net, bus=bus_map[gen["bus"]], p_mw=p_mw, name=name)

    for name, branch in elements.get("branch", {}).items():
        from_bus = bus_map[branch["from_bus"]]
        to_bus = bus_map[branch["to_bus"]]
        vn_kv = net.bus.at[from_bus, "vn_kv"]
        rating_mva = branch.get("rating_long_term") or 9999.0
        max_i_ka = rating_mva / (3**0.5 * vn_kv) if vn_kv > 0 else 1.0
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=1.0,
            r_ohm_per_km=branch.get("resistance", 0.01),
            x_ohm_per_km=branch.get("reactance", 0.1),
            c_nf_per_km=0.0,
            max_i_ka=max_i_ka,
            name=name,
        )

    for name, dc in elements.get("dc_branch", {}).items():
        from_bus = bus_map[dc["from_bus"]]
        to_bus = bus_map[dc["to_bus"]]
        rating_mva = dc.get("rating_long_term") or 9999.0
        pp.create_dcline(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            p_mw=0.0,
            loss_percent=0.0,
            loss_mw=0.0,
            vm_from_pu=1.0,
            vm_to_pu=1.0,
            max_p_mw=rating_mva,
            name=name,
        )

    return net


def angle_repulsion(g, positions, strength, iterations):
    """Spread the edges incident to each node toward even angular spacing.

    strength: per-step gain on the angular repulsion.
    iterations: number of relaxation passes.
    """
    pos = {n: np.asarray(positions[n], dtype=float) for n in positions}
    graph = nx.Graph(g)
    for _ in range(iterations):
        disp = {n: np.zeros(2) for n in pos}
        for u in graph.nodes():
            nbrs = list(graph.neighbors(u))
            if len(nbrs) < 2:
                continue
            vecs = {v: pos[v] - pos[u] for v in nbrs}
            ang = {v: np.arctan2(vecs[v][1], vecs[v][0]) for v in nbrs}
            for v in nbrs:
                dtheta = 0.0
                for w in nbrs:
                    if w == v:
                        continue
                    gap = (ang[v] - ang[w] + np.pi) % (2 * np.pi) - np.pi
                    gap = gap if abs(gap) > 1e-3 else np.copysign(1e-3, gap or 1.0)
                    dtheta += strength / gap
                r = np.linalg.norm(vecs[v]) or 1.0
                tangent = np.array([-vecs[v][1], vecs[v][0]]) / r
                disp[v] += np.clip(dtheta, -0.5, 0.5) * r * tangent
        for n in pos:
            pos[n] += disp[n]
        pos = nx.spring_layout(g, pos=pos, iterations=1, weight="weight")
    return pos


def neighbor_centering(g, pos, strength, iterations=1):
    for _ in range(iterations):
        disp = {n: np.zeros(2) for n in pos}
        for u in g.nodes():
            nbrs = list(g.neighbors(u))
            if len(nbrs) < 2:
                continue
            center = np.sum([pos[v] for v in nbrs])
            disp[u] = (center - pos[u]) * strength
        for n in pos:
            pos[n] += disp[n]
        pos = nx.spring_layout(g, pos=pos, iterations=1, weight="weight")
    return pos
        


def _case30_nudges(pos):
    """Hand-tuned position offsets for the case30 planar layout."""
    for i in range(11, 13):
        pos[i] += np.array([0.5, -0.6])
    pos[14] += np.array([0.6, 0.6])
    pos[13] += np.array([0.8, 0.0])
    pos[18] += np.array([-0.15, -0.2])
    pos[19] += np.array([-0.25, 0.0])
    pos[16] += np.array([-0.25, 0.0])
    pos[8] += np.array([0.3, 0.])
    pos[10] += np.array([0.8, 0.0])
    for i in range(4):
        pos[i] += np.array([-0.1, -0.3])
    for i in range(4, 8):
        pos[i] += np.array([-0.5, -0.3])
    for i in range(26, 28):
        pos[i] += np.array([-0.3, 0.0])
    pos[29] += np.array([-0.4, 0.4])
    pos[1] += np.array([-0.1, -0.1])
    return pos


def _case14_nudges(pos):
    """Relational tweaks to the case14 layout.

    Fractional pulls on the already-relaxed layout (so node spacing survives):
    buses 4 and 5 toward the layout center to fill the interior, and buses 10
    and 11 toward the 1/3 and 2/3 points of the 9->6 segment (the 9-10-11-6
    chain). Node indices are bus name minus one.
    """
    idx = {name: name - 1 for name in range(1, 15)}
    center = np.mean(list(pos.values()), axis=0)
    p9, p6 = pos[idx[9]], pos[idx[6]]
    pos[idx[10]] += 0.55 * ((p9 + (p6 - p9) / 3.0) - pos[idx[10]])
    pos[idx[11]] += 0.55 * ((p9 + 2.0 * (p6 - p9) / 3.0) - pos[idx[11]])
    pos[idx[4]] += 0.5 * (center - pos[idx[4]])
    pos[idx[5]] += 0.5 * (center - pos[idx[5]])
    return pos


LAYOUT_SEEDS = 400


def _segments_cross(p1, p2, p3, p4):
    """True if segment p1-p2 properly intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def count_edge_crossings(g, pos):
    """Number of non-adjacent edge pairs whose drawn segments intersect."""
    edges = list(g.edges())
    total = 0
    for i, (a, b) in enumerate(edges):
        for c, d in edges[i + 1:]:
            if len({a, b, c, d}) < 4:
                continue
            if _segments_cross(pos[a], pos[b], pos[c], pos[d]):
                total += 1
    return total


def _layout_stress(g, pos, dist):
    """Mean squared mismatch between geometric distance and graph hop count.

    Lower stress pulls central, high-degree nodes inward and fills the interior,
    unlike edge-length uniformity which rewards ring-shaped layouts. dist is the
    precomputed all-pairs shortest-path length dict.
    """
    nodes = list(g.nodes())
    total = 0.0
    count = 0
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            geo = float(np.hypot(*(pos[u] - pos[v])))
            hops = dist[u][v]
            total += (geo - hops) ** 2 / hops**2
            count += 1
    return total / count


def _planar_fallback_layout(g, special):
    """Layout for planar graphs too dense for a crossing-free spring layout."""
    _, embedding = nx.check_planarity(nx.Graph(g))
    init = nx.planar_layout(embedding)
    pos = nx.kamada_kawai_layout(g, pos=init, weight="weight")
    if str(special).startswith("case30"):
        pos = _case30_nudges(pos)
    if str(special).startswith("case14"):
        pos = _case14_nudges(pos)
    return nx.spring_layout(g, pos=pos, iterations=5, weight="weight")


def compute_layout(g, special=None):
    """Node positions: crossing-free for planar graphs where one is reachable.

    For planar graphs, searches unweighted spring layouts and keeps the
    crossing-free one with the lowest graph-distance stress (best space filling).
    Dense planar graphs where no seed is crossing-free fall back to the
    Kamada-Kawai layout; non-planar graphs use a weighted spring layout.
    """
    graph = nx.Graph(g)
    if not nx.check_planarity(graph)[0]:
        return nx.spring_layout(g, seed=42, weight="weight")
    dist = dict(nx.all_pairs_shortest_path_length(graph))
    best = None
    for seed in range(LAYOUT_SEEDS):
        pos = nx.spring_layout(graph, seed=seed)
        if count_edge_crossings(graph, pos) == 0:
            stress = _layout_stress(graph, pos, dist)
            if best is None or stress < best[0]:
                best = (stress, pos)
    if best is not None:
        pos = best[1]
        if str(special).startswith("case14"):
            pos = _case14_nudges(pos)
        return pos
    return _planar_fallback_layout(g, special)


def assign_geodata(net, special=None):
    g = ptopo.create_nxgraph(net, include_dclines=True)
    for u, v, key, data in g.edges(keys=True, data=True):
        etype, line_idx = key
        if etype == "line" and line_idx in net.line.index:
            x = net.line.at[line_idx, "x_ohm_per_km"]
            data["weight"] = (1.0 / x**0.2 if x > 0 else 1.0) * 1e-2
    positions = compute_layout(g, special)
    net.bus["geo"] = [
        geojson.dumps(geojson.Point((positions[i][0], positions[i][1])))
        for i in net.bus.index
    ]


def get_bus_coordinates(net):
    """Extract bus index -> (x, y) mapping from pandapower geodata."""
    return {
        i: geojson.loads(net.bus.at[i, "geo"])["coordinates"]
        for i in net.bus.index
    }


def plot_lines(ax, net, bus_xy):
    """Draw transmission lines and label them."""
    lc = pplot.create_line_collection(net, use_bus_geodata=True, color="gray", zorder=1)
    pplot.draw_collections([lc], ax=ax)

    offset = 0.04
    for _, line in net.line.iterrows():
        x0, y0 = bus_xy[line.from_bus]
        x1, y1 = bus_xy[line.to_bus]
        dx, dy = x1 - x0, y1 - y0
        norm = (dx**2 + dy**2) ** 0.5 or 1.0
        nx_, ny_ = -dy / norm, dx / norm
        ax.annotate(
            line["name"],
            xy=((x0 + x1) / 2 + offset * nx_, (y0 + y1) / 2 + offset * ny_),
            fontsize=8,
            ha="center",
            va="center",
            c="blue",
        )


def plot_buses(fig, ax, net, bus_xy, bus_energy, gen_buses):
    """Draw buses colored by energy consumption, with name labels.

    Buses hosting a generator get a highlighted (red, thicker) edge ring.
    """
    gen_color = "firebrick"
    xs = [bus_xy[i][0] for i in net.bus.index]
    ys = [bus_xy[i][1] for i in net.bus.index]
    cs = [bus_energy.get(net.bus.at[i, "name"], 0.0) for i in net.bus.index]
    has_gen = [net.bus.at[i, "name"] in gen_buses for i in net.bus.index]
    edgecolors = [gen_color if g else "k" for g in has_gen]
    linewidths = [2.2 if g else 0.8 for g in has_gen]
    sc = ax.scatter(
        xs, ys, c=cs, cmap="viridis", s=120, zorder=5,
        edgecolors=edgecolors, linewidths=linewidths,
    )
    fig.colorbar(sc, ax=ax, label="Load (MWh)", shrink=1.0, fraction=0.04, pad=0.10)

    ax.scatter([], [], facecolor="none", edgecolors=gen_color, linewidths=2.2,
               s=120, label="Generator bus")
    ax.legend(loc="best", fontsize=8)

    for i in net.bus.index:
        x, y = bus_xy[i]
        ax.annotate(
            net.bus.at[i, "name"],
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            zorder=7,
        )


def plot_network(egret_data, net):
    """Create the full network figure."""
    bus_xy = get_bus_coordinates(net)
    bus_energy = compute_bus_energy(egret_data)
    gen_buses = {
        gen["bus"] for gen in egret_data["elements"].get("generator", {}).values()
    }

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(egret_data["system"].get("model_name", "Network"))

    plot_lines(ax, net, bus_xy)
    plot_buses(fig, ax, net, bus_xy, bus_energy, gen_buses)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot an EGRET network using pandapower")
    parser.add_argument("json_file", help="Path to EGRET JSON model file")
    args = parser.parse_args()

    egret_data = load_egret_json(args.json_file)
    net = build_pandapower_network(egret_data)
    assign_geodata(net, args.json_file.split("/")[-1])

    plot_network(egret_data, net)
    plt.savefig(args.json_file.replace(".json", ".pdf"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
