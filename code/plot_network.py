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


def assign_geodata(net):
    g = ptopo.create_nxgraph(net, include_dclines=True)
    for u, v, key, data in g.edges(keys=True, data=True):
        etype, line_idx = key
        if etype == "line" and line_idx in net.line.index:
            x = net.line.at[line_idx, "x_ohm_per_km"]
            data["weight"] = (1.0 / x**0.5 if x > 0 else 1.0) * 1e-1
    positions = nx.spring_layout(g, seed=42)
    positions[7] += np.array([0.4, 0.4])
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


def plot_buses(fig, ax, net, bus_xy, bus_energy):
    """Draw buses colored by energy consumption, with name labels."""
    xs = [bus_xy[i][0] for i in net.bus.index]
    ys = [bus_xy[i][1] for i in net.bus.index]
    cs = [bus_energy.get(net.bus.at[i, "name"], 0.0) for i in net.bus.index]
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=120, zorder=5, edgecolors="k")
    fig.colorbar(sc, ax=ax, label="Load (MWh)", shrink=1.0, fraction=0.04, pad=0.10)

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


def plot_generators(ax, net, bus_xy, egret_data):
    """Mark generator buses with triangles."""
    name_to_idx = {net.bus.at[i, "name"]: i for i in net.bus.index}
    gen_buses = {
        gen["bus"] for gen in egret_data["elements"].get("generator", {}).values()
    }
    gxs = [bus_xy[name_to_idx[b]][0] for b in gen_buses if b in name_to_idx]
    gys = [bus_xy[name_to_idx[b]][1] for b in gen_buses if b in name_to_idx]
    #ax.scatter(
    #    gxs, gys, marker="^", s=40, facecolor="red", edgecolors="k",
    #    zorder=6, label="Generator",
    #)
    ax.legend(loc="best", fontsize=8)


def plot_network(egret_data, net):
    """Create the full network figure."""
    bus_xy = get_bus_coordinates(net)
    bus_energy = compute_bus_energy(egret_data)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(egret_data["system"].get("model_name", "Network"))

    plot_lines(ax, net, bus_xy)
    plot_buses(fig, ax, net, bus_xy, bus_energy)
    plot_generators(ax, net, bus_xy, egret_data)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot an EGRET network using pandapower")
    parser.add_argument("json_file", help="Path to EGRET JSON model file")
    args = parser.parse_args()

    egret_data = load_egret_json(args.json_file)
    net = build_pandapower_network(egret_data)
    assign_geodata(net)

    plot_network(egret_data, net)
    plt.savefig(args.json_file.replace(".json", ".pdf"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
