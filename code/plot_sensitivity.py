"""
Plot LMP sensitivity results from sensitivity_analysis.py or
finite_difference_sensitivity.py.

Accepts one or two JSON result files.  When two files are given (one KKT, one
finite-difference), they are displayed side-by-side for comparison.

Usage:
    # Single file
    uv run python code/plot_sensitivity.py results/sensitivity_kkt_<case>.json
    uv run python code/plot_sensitivity.py results/sensitivity_fd_<case>.json

    # Side-by-side comparison
    uv run python code/plot_sensitivity.py \\
        results/sensitivity_kkt_<case>.json \\
        results/sensitivity_fd_<case>.json

    # Custom output path
    uv run python code/plot_sensitivity.py results/sensitivity_kkt_<case>.json --out fig.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> dict:
    """Load a sensitivity JSON file produced by sensitivity_analysis.py or
    finite_difference_sensitivity.py."""
    return json.loads(path.read_text())


def _heatmap_title(results: dict) -> str:
    method = results["method"]
    case = results["case"]
    if method == "kkt":
        label = "Analytical (KKT)" if results.get("is_qp") else "Analytical (KKT) — LP, zeros"
        return f"{label}\n{case}"
    delta = results.get("delta_mw", "?")
    return f"Finite-diff ($\\Delta P$ = {delta:.0f} MW)\n{case}"


def _heatmap_xlabel(results: dict) -> str:
    if results["method"] == "finite_difference":
        delta = results.get("delta_mw", "?")
        return f"Injection bus $j$ ($\\Delta P_j$ = +{delta:.0f} MW)"
    return "Injection bus $j$"


def _draw_heatmap(ax, values: np.ndarray, bus_names: list[str], title: str, xlabel: str) -> None:
    """Draw a diverging heatmap with annotated cells onto ax."""
    n = len(bus_names)
    vmax = np.abs(values).max()
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(n):
        for j in range(n):
            val = values[i, j]
            color = "white" if abs(val) > 0.6 * vmax else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(n))
    ax.set_xticklabels(bus_names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(bus_names)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Observed bus $i$ (LMP$_i$)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="$\\partial \\lambda / \\partial P$ (\\$/MWh per MW)")


def plot_sensitivity(result_sets: list[dict], out_path: Path) -> None:
    """Render and save a figure from one or two sensitivity result dicts.

    Parameters
    ----------
    result_sets:
        List of one or two dicts as returned by ``load_results``.
    out_path:
        Destination PDF path.
    """
    # Use the first result's bus names and base LMPs for the bar chart
    primary = result_sets[0]
    bus_names = primary["bus_names"]
    lmp_base = primary["lmp_base"]
    n = len(bus_names)

    n_heat = len(result_sets)
    n_panels = 1 + n_heat  # bar chart + one heatmap per result file
    width_ratios = [1] + [n] * n_heat

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(max(8, n * 1.5 * n_panels / 2 + 4), max(5, n * 0.8)),
        gridspec_kw={"width_ratios": width_ratios},
    )

    # ---- Base LMPs bar chart ----
    ax_bar = axes[0]
    base_vals = [lmp_base.get(b, 0.0) for b in bus_names]
    ax_bar.barh(range(n), base_vals, color="steelblue", edgecolor="white")
    ax_bar.set_yticks(range(n))
    ax_bar.set_yticklabels(bus_names)
    ax_bar.set_xlabel("LMP ($/MWh)")
    ax_bar.set_title("Base LMPs")
    ax_bar.invert_yaxis()
    vmax_bar = max(base_vals) if base_vals else 1.0
    for i, v in enumerate(base_vals):
        ax_bar.text(v + 0.02 * vmax_bar, i, f"{v:.2f}", va="center", fontsize=8)

    # ---- Heatmap(s) ----
    for ax, results in zip(axes[1:], result_sets):
        _draw_heatmap(
            ax,
            np.array(results["sensitivity"]),
            results["bus_names"],
            title=_heatmap_title(results),
            xlabel=_heatmap_xlabel(results),
        )

    plt.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot LMP sensitivity results from sensitivity_analysis.py "
                    "or finite_difference_sensitivity.py."
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="One or two sensitivity JSON files to plot.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PDF path.  Defaults to the stem of the first input file + .pdf.",
    )
    args = parser.parse_args()

    if len(args.results) > 2:
        parser.error("At most two result files can be compared at once.")

    result_sets = []
    for r in args.results:
        path = Path(r)
        print(f"Loading {path.name}...")
        data = load_results(path)
        print(f"  method={data['method']}, case={data['case']}, buses={data['bus_names']}")
        result_sets.append(data)

    out_path = (
        Path(args.out)
        if args.out
        else Path(args.results[0]).with_suffix(".pdf")
    )

    plot_sensitivity(result_sets, out_path)


if __name__ == "__main__":
    main()
