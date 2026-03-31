"""Plot an EGRET JSON network using pandapower and matplotlib."""

import argparse
import json

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
        rating_mva = branch.get("rating_long_term", 9999.0)
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
        rating_mva = dc.get("rating_long_term", 9999.0)
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
    positions = nx.spring_layout(g, seed=42)
    net.bus["geo"] = [
        geojson.dumps(geojson.Point((positions[i][0], positions[i][1])))
        for i in net.bus.index
    ]


def main():
    parser = argparse.ArgumentParser(description="Plot an EGRET network using pandapower")
    parser.add_argument("json_file", help="Path to EGRET JSON model file")
    args = parser.parse_args()

    egret_data = load_egret_json(args.json_file)
    net = build_pandapower_network(egret_data)
    assign_geodata(net)

    fig, ax = plt.subplots(figsize=(10, 8))
    pplot.simple_plot(net, ax=ax, show_plot=False)
    ax.set_title(egret_data["system"].get("model_name", "Network"))

    outputfile = None
    if outputfile:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
