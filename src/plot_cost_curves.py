"""Plot piecewise-linear cost curves of thermal generators in an EGRET UC JSON.

Usage:
    uv run python src/plot_cost_curves.py [path_to_json]

Default input: Egret/egret/models/tests/uc_test_instances/test_scuc_masked.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_JSON = (
    REPO_ROOT
    / "Egret/egret/models/tests/uc_test_instances/test_scuc_masked.json"
)
OUT_DIR = REPO_ROOT / "results"

FUEL_LABELS = {"C": "Coal", "G": "Gas", "O": "Oil", "N": "Nuclear"}
FUEL_COLORS = {"C": "black", "G": "tab:orange", "O": "tab:brown", "N": "tab:red"}


def load_thermal_generators(path: Path):
    with open(path) as f:
        data = json.load(f)
    gens = data["elements"]["generator"]
    return {n: g for n, g in gens.items() if g.get("generator_type") == "thermal"}


def piecewise_points(p_cost):
    if p_cost["cost_curve_type"] != "piecewise":
        raise ValueError(f"unsupported cost_curve_type: {p_cost['cost_curve_type']}")
    pts = np.array(p_cost["values"], dtype=float)
    return pts[:, 0], pts[:, 1]


def marginal_segments(p, c):
    """Return (p_mid, slope) for each segment of a piecewise-linear curve."""
    dp = np.diff(p)
    dc = np.diff(c)
    return p, dc / dp


def plot_total_cost(gens, ax):
    for name, g in gens.items():
        p, c = piecewise_points(g["p_cost"])
        color = FUEL_COLORS.get(g["fuel"], "gray")
        ax.plot(p, c, color=color, alpha=0.8, linewidth=1.2)
        ax.plot(p, c, "o", color=color, markersize=3)
    ax.set_xlabel("Power output $P$ [MW]")
    ax.set_ylabel("Production cost [\\$/h]")
    ax.set_title("Total cost curves")
    ax.grid(True, alpha=0.3)


def plot_marginal_cost(gens, ax):
    for name, g in gens.items():
        p, c = piecewise_points(g["p_cost"])
        p_bp, slopes = marginal_segments(p, c)
        color = FUEL_COLORS.get(g["fuel"], "gray")
        # Step plot: slope is constant on each segment [p_i, p_{i+1}]
        ax.step(p_bp, np.r_[slopes, slopes[-1]], where="post",
                color=color, alpha=0.8, linewidth=1.2)
    ax.set_xlabel("Power output $P$ [MW]")
    ax.set_ylabel("Marginal cost [\\$/MWh]")
    ax.set_title("Marginal cost curves")
    ax.grid(True, alpha=0.3)


def add_fuel_legend(ax, fuels_present):
    handles = [
        plt.Line2D([0], [0], color=FUEL_COLORS[f], label=FUEL_LABELS[f])
        for f in fuels_present
    ]
    ax.legend(handles=handles, loc="best", frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", nargs="?", type=Path, default=DEFAULT_JSON)
    #parser.add_argument("--out", type=Path, default=OUT_DIR / "cost_curves.pdf")
    args = parser.parse_args()
    out = OUT_DIR / f"{args.json_path.stem}_cost_curves.pdf"

    gens = load_thermal_generators(args.json_path)
    fuels_present = sorted({g["fuel"] for g in gens.values()})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_total_cost(gens, axes[0])
    plot_marginal_cost(gens, axes[1])
    add_fuel_legend(axes[0], fuels_present)
    fig.suptitle(f"{args.json_path.name}: {len(gens)} thermal generators")
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.show()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
