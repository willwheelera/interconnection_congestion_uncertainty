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
from matplotlib.backends.backend_pdf import PdfPages


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
    if method == "barrier":
        return f"Barrier-regularised\n{case}"
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


_MU_LABELS = {
    "branch_upper": r"Branch upper $\partial\mu^+_\ell/\partial P$",
    "branch_lower": r"Branch lower $\partial\mu^-_\ell/\partial P$",
}


def _get_mu_matrices(results: dict) -> list[tuple[str, str, list, list, np.ndarray]]:
    """Return (key, label, row_names, col_names, matrix) for each non-empty mu sensitivity."""
    out = []
    for key, label in _MU_LABELS.items():
        entry = results.get("mu_sensitivity", {}).get(key)
        if entry is None:
            continue
        arr = np.array(entry["values"])
        if arr.size == 0 or not np.any(arr != 0):
            continue
        out.append((key, label, entry["index"], entry["columns"], arr))
    return out


def _draw_mu_heatmap(
    ax, values: np.ndarray, row_names: list, col_names: list, title: str,
    vmax: float, show_ylabel: bool = True,
):
    """Draw a mu sensitivity heatmap (constraints × buses).

    Returns the AxesImage so the caller can attach a shared colorbar.
    vmax is supplied externally so all panels in a row share the same scale.
    """
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=7)
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    ax.set_xlabel("Injection bus $j$")
    if show_ylabel:
        ax.set_ylabel("Constraint")
    ax.set_title(title, fontsize=9)
    return im


def _lmp_figure(result_sets: list[dict]) -> plt.Figure:
    """Build the LMP sensitivity figure (base LMPs bar + dλ/dP heatmap(s))."""
    primary = result_sets[0]
    bus_names = primary["bus_names"]
    lmp_base = primary["lmp_base"]
    n = len(bus_names)
    n_heat = len(result_sets)
    width_ratios = [1] + [n] * n_heat

    fig, axes = plt.subplots(
        1, 1 + n_heat,
        figsize=(max(6, n * 0.5 * (n_heat + 1) / 2 + 2), max(3, n * 0.4)),
        gridspec_kw={"width_ratios": width_ratios},
    )

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

    for ax, results in zip(axes[1:], result_sets):
        matrix = results.get("sensitivity") or results.get("lmp_sensitivity")
        _draw_heatmap(
            ax,
            np.array(matrix),
            results["bus_names"],
            title=_heatmap_title(results),
            xlabel=_heatmap_xlabel(results),
        )

    fig.tight_layout()
    return fig


def _mu_figure(results: dict) -> plt.Figure | None:
    """Build a figure with one heatmap per non-empty mu sensitivity matrix, 2 per row.

    Panels in the same row share the y-axis and a single colorbar scaled to their
    common vmax.  When ``mu_base`` is present in *results*, a narrow bar chart
    showing the current μ value is placed to the left of each heatmap.
    """
    mu_matrices = _get_mu_matrices(results)
    if not mu_matrices:
        return None

    mu_base = results.get("mu_base", {})
    has_bar = bool(mu_base)
    n_heatmap_cols = 2
    n_rows = (len(mu_matrices) + n_heatmap_cols - 1) // n_heatmap_cols
    n_buses = len(results["bus_names"])

    # Layout per row: [bar, heat, bar, heat] when has_bar, else [heat, heat]
    if has_bar:
        total_cols = n_heatmap_cols * 2
        width_ratios = [v for _ in range(n_heatmap_cols) for v in (1, n_buses)]
    else:
        total_cols = n_heatmap_cols
        width_ratios = [n_buses] * n_heatmap_cols

    fig, axes = plt.subplots(
        n_rows, total_cols,
        figsize=(max(4, n_buses * 0.3) * n_heatmap_cols / 2 + (2 * n_heatmap_cols if has_bar else 0), n_rows * 4),
        squeeze=False,
        sharey="row",
        layout="constrained",
        gridspec_kw={"width_ratios": width_ratios},
    )

    for row_idx in range(n_rows):
        row_items = mu_matrices[row_idx * n_heatmap_cols : (row_idx + 1) * n_heatmap_cols]
        row_vmax = max(np.abs(arr).max() for _, _, _, _, arr in row_items) or 1.0

        last_im = None
        heatmap_axes = []
        for col_idx, (key, label, row_names, col_names, arr) in enumerate(row_items):
            if has_bar:
                ax_bar = axes[row_idx][col_idx * 2]
                base_vals = [mu_base.get(key, {}).get(name, 0.0) for name in row_names]
                ax_bar.barh(range(len(row_names)), base_vals, color="steelblue", edgecolor="white")
                ax_bar.set_yticks(range(len(row_names)))
                ax_bar.set_yticklabels(row_names, fontsize=7)
                ax_bar.set_xlabel("μ ($/MWh)")
                ax_bar.set_title("Base μ", fontsize=8)
                ax_heat = axes[row_idx][col_idx * 2 + 1]
            else:
                ax_heat = axes[row_idx][col_idx]

            last_im = _draw_mu_heatmap(
                ax_heat, arr, row_names, col_names, label,
                vmax=row_vmax, show_ylabel=(col_idx == 0 and not has_bar),
            )
            heatmap_axes.append(ax_heat)

        # One colorbar spanning all heatmap panels in this row
        fig.colorbar(
            last_im,
            ax=heatmap_axes,
            label=r"$\partial\mu/\partial P$ (\$/MWh per MW)",
        )

        # Hide unused panels (last row may be short)
        for col_idx in range(len(row_items), n_heatmap_cols):
            if has_bar:
                axes[row_idx][col_idx * 2].set_visible(False)
                axes[row_idx][col_idx * 2 + 1].set_visible(False)
            else:
                axes[row_idx][col_idx].set_visible(False)

    case = results.get("case", "")
    fig.suptitle(f"Congestion dual sensitivity — {case}", fontsize=10)
    return fig


def plot_sensitivity(result_sets: list[dict], out_path: Path) -> None:
    """Render and save sensitivity figures to a multi-page PDF.

    Page 1: base LMPs bar chart + dλ/dP heatmap(s).
    Page 2: dμ/dP heatmaps for each non-empty constraint type (barrier results only).

    Parameters
    ----------
    result_sets:
        List of one or two dicts as returned by ``load_results``.
    out_path:
        Destination PDF path.
    """
    out_path.parent.mkdir(exist_ok=True)
    with PdfPages(out_path) as pdf:
        fig_lmp = _lmp_figure(result_sets)
        pdf.savefig(fig_lmp, bbox_inches="tight")

        fig_mu = _mu_figure(result_sets[0])
        if fig_mu is not None:
            pdf.savefig(fig_mu, bbox_inches="tight")

        plt.show()
        plt.close(fig_lmp)
        plt.close(fig_mu)

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
    #main()
    for h in [18, 20]:#range(24):
        #fname = f"../results/sensitivity_barrier_uc_test_scuc_masked_h{h:02d}.json"
        #fname = f"../results/sensitivity_barrier_uc_three_bus_uc_h{h:02d}.json"
        fname = f"../results/sensitivity_barrier_uc_many_gen_four_bus_uc_avail42924_h{h:02d}.json"
        out_path = Path(fname).with_suffix(".pdf")
        data = load_results(Path(fname))
        plot_sensitivity([data], out_path)
