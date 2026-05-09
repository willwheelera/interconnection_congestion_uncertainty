"""
Visualize OPF solution: bar charts of bus-level and branch-level quantities.

Four panels:
  1. Bus net power injection (generation − load) per bus
  2. Bus LMPs ($/MWh)
  3. Branch power flows (MW)
  4. Branch congestion duals mu ($/MWh)

Usage:
    uv run python code/plot_opf_result.py
    uv run python code/plot_opf_result.py --solution results/case14_gen_per_bus/opf_case14_gen_per_bus_avail16383.json.gz
    uv run python code/plot_opf_result.py --solution path/to/result.json.gz --out fig.pdf
"""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def read_json(path: Path) -> dict:
    if path.suffix == ".gz" or str(path).endswith(".json.gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def load_bus_data(raw: dict) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return bus names, generation, load, net injection, and LMP arrays."""
    buses = raw["elements"]["bus"]
    gens = raw["elements"]["generator"]
    loads = raw["elements"]["load"]
    bus_names = sorted(buses.keys(), key=lambda b: int(b))

    gen_at_bus = {b: 0.0 for b in bus_names}
    for g in gens.values():
        gen_at_bus[g["bus"]] += g["pg"]

    load_at_bus = {b: 0.0 for b in bus_names}
    for ld in loads.values():
        load_at_bus[ld["bus"]] += ld["p_load"]

    generation = np.array([gen_at_bus[b] for b in bus_names])
    load = np.array([load_at_bus[b] for b in bus_names])
    net = generation - load
    lmp = np.array([buses[b].get("lmp", 0.0) for b in bus_names])
    return bus_names, generation, load, net, lmp


def load_branch_data(raw: dict) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return branch names, labels, flow, mu, and rating arrays."""
    branches = raw["elements"]["branch"]
    br_names = sorted(branches.keys(), key=lambda b: int(b))

    labels = [f"{branches[b]['from_bus']}\u2192{branches[b]['to_bus']}" for b in br_names]
    pf = np.array([branches[b].get("pf", 0.0) or 0.0 for b in br_names])
    mu_upper = np.array([branches[b].get("mu_upper", 0.0) for b in br_names])
    mu_lower = np.array([branches[b].get("mu_lower", 0.0) for b in br_names])
    mu = mu_upper - mu_lower
    rating = np.array([branches[b].get("rating_long_term") or np.nan for b in br_names])
    return br_names, labels, pf, mu, rating


def plot_opf_result(raw: dict, out_path: Path) -> None:
    bus_names, generation, load, net, lmp = load_bus_data(raw)
    br_names, br_labels, pf, mu, rating = load_branch_data(raw)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    model_name = raw.get("system", {}).get("model_name", "")
    total_cost = raw.get("system", {}).get("total_cost")
    title = f"OPF Result: {model_name}"
    if total_cost is not None:
        title += f"  (total cost: ${total_cost:,.2f})"
    fig.suptitle(title, fontsize=13)

    plot_bus_power(axes[0, 0], bus_names, generation, load, net)
    plot_bus_lmp(axes[1, 0], bus_names, lmp)
    plot_branch_flow(axes[0, 1], br_names, br_labels, pf, rating)
    plot_branch_mu(axes[1, 1], br_names, br_labels, mu)

    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def plot_bus_power(ax, bus_names, generation, load, net):
    x = np.arange(len(bus_names))
    w = 0.35
    ax.bar(x - w / 2, generation, w, label="Generation", color="#2ecc71")
    ax.bar(x + w / 2, load, w, label="Load", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(bus_names, fontsize=8)
    ax.set_xlabel("Bus")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Generation and load by bus")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", linewidth=0.5)


def plot_bus_lmp(ax, bus_names, lmp):
    x = np.arange(len(bus_names))
    colors = ["#3498db" if v >= 0 else "#e67e22" for v in lmp]
    ax.bar(x, lmp, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(bus_names, fontsize=8)
    ax.set_xlabel("Bus")
    ax.set_ylabel("LMP ($/MWh)")
    ax.set_title("Locational marginal prices")
    ax.axhline(0, color="k", linewidth=0.5)


def plot_branch_flow(ax, br_names, br_labels, pf, rating):
    x = np.arange(len(br_names))
    ax.bar(x, pf, color="#7f8c8d")
    has_rating = ~np.isnan(rating)
    if has_rating.any():
        s = np.sign(pf)
        ax.bar(x[has_rating], s * rating[has_rating], width=0.8, fill=False,
               edgecolor="red", linestyle="-", linewidth=0.5, label="Rating")
        ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(br_labels, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Branch")
    ax.set_ylabel("Flow (MW)")
    ax.set_title("Branch power flows")
    ax.axhline(0, color="k", linewidth=0.5)


def plot_branch_mu(ax, br_names, br_labels, mu):
    x = np.arange(len(br_names))
    colors = ["#9b59b6" if abs(v) > 1e-6 else "#bdc3c7" for v in mu]
    ax.bar(x, mu, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(br_labels, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Branch")
    ax.set_ylabel("\u03bc ($/MWh)")
    ax.set_title("Congestion duals (+ = upper, \u2212 = lower)")
    ax.axhline(0, color="k", linewidth=0.5)


def main():
    parser = argparse.ArgumentParser(
        description="Plot OPF solution: LMPs, flows, congestion duals."
    )
    parser.add_argument("ID", type=str, default=None,
                        help="Path to OPF solution JSON file.")
    args = parser.parse_args()

    ID = args.ID if args.ID is not None else "16383"
    solution_file = f"opf_case14_gen_per_bus_avail{ID}.json.gz"
    solution_path = RESULTS_DIR / "case14_gen_per_bus" / solution_file
    out_path = (
        RESULTS_DIR / f"{solution_path.stem.split('.')[0]}_plot.pdf"
    )
    raw = read_json(solution_path)
    plot_opf_result(raw, out_path)


if __name__ == "__main__":
    main()
