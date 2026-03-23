"""
Plot optimal primal and dual variables from a UC solution against time.

Three panels:
  1. Generation dispatch (MW) — stacked area chart by fuel type
  2. Power-balance duals λ (LMP, $/MWh) — one line per bus
  3. Branch congestion duals μ ($/MWh) — one line per branch;
     positive = upper limit binding, negative = lower limit binding

Usage:
    uv run python code/plot_optimal_values.py
    uv run python code/plot_optimal_values.py --solution results/uc_test_scuc_masked.json
    uv run python code/plot_optimal_values.py --solution results/uc_test_scuc_masked.json --out fig.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_SOLUTION = RESULTS_DIR / "uc_test_scuc_masked.json"

# Human-readable fuel labels
FUEL_LABELS = {
    "C": "Coal",
    "G": "Gas CC",
    "N": "Nuclear",
    "O": "Oil / CT",
    "H": "Hydro",
    "S": "Solar",
    "W": "Wind",
}

# Consistent colors per fuel type
FUEL_COLORS = {
    "Coal":     "#4e4e4e",
    "Gas CC":   "#b5651d",
    "Nuclear":  "#9b59b6",
    "Oil / CT": "#e67e22",
    "Hydro":    "#3498db",
    "Solar":    "#f1c40f",
    "Wind":     "#2ecc71",
}


def load_pg(solution_path: Path) -> pd.DataFrame:
    """Return (n_hours × n_generators) DataFrame of pg dispatch in MW."""
    raw = json.loads(solution_path.read_text())
    gens = raw["elements"]["generator"]
    n_hours = len(raw["system"]["time_keys"])
    data = {}
    for name, g in gens.items():
        pg = g.get("pg", 0.0)
        data[name] = pg["values"] if isinstance(pg, dict) else [float(pg)] * n_hours
    return pd.DataFrame(data, index=range(n_hours))


def load_generator_metadata(solution_path: Path) -> pd.DataFrame:
    """Return DataFrame with 'fuel' and 'generator_type' columns indexed by generator name."""
    raw = json.loads(solution_path.read_text())
    gens = raw["elements"]["generator"]
    return pd.DataFrame(
        {name: {"fuel": g.get("fuel", "?"), "type": g.get("generator_type", "?")}
         for name, g in gens.items()}
    ).T


def load_lmp_mu(solution_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-hour LMP and branch congestion duals from sensitivity result files.

    Returns:
        lmp_df  — (n_hours × n_buses) DataFrame of λ ($/MWh)
        mu_df   — (n_hours × n_branches) DataFrame of μ ($/MWh),
                  positive = upper limit binding, negative = lower binding
    """
    stem = solution_path.stem
    lmp_rows: dict[int, dict] = {}
    mu_rows: dict[int, dict] = {}

    for h in range(24):
        fpath = RESULTS_DIR / f"sensitivity_barrier_{stem}_h{h:02d}.json"
        if not fpath.exists():
            continue
        data = json.loads(fpath.read_text())
        lmp_rows[h] = data.get("lmp_base", {})
        mu_base = data.get("mu_base", {})
        if mu_base:
            upper = mu_base.get("branch_upper", {})
            lower = mu_base.get("branch_lower", {})
            branches = sorted(set(upper) | set(lower))
            mu_rows[h] = {br: upper.get(br, 0.0) - lower.get(br, 0.0) for br in branches}

    lmp_df = pd.DataFrame.from_dict(lmp_rows, orient="index").sort_index()
    mu_df = pd.DataFrame.from_dict(mu_rows, orient="index").sort_index()
    return lmp_df, mu_df


def _pg_by_fuel(pg_df: pd.DataFrame, meta: pd.DataFrame) -> dict[str, pd.Series]:
    """Aggregate pg by fuel type label, returning {label: hourly_total_MW}."""
    result = {}
    for fuel_code, label in FUEL_LABELS.items():
        gens = meta.index[meta["fuel"] == fuel_code].tolist()
        cols = [g for g in gens if g in pg_df.columns]
        if cols:
            result[label] = pg_df[cols].sum(axis=1)
    return result


def plot_optimal_values(solution_path: Path, out_path: Path) -> None:
    pg_df = load_pg(solution_path)
    meta = load_generator_metadata(solution_path)
    lmp_df, mu_df = load_lmp_mu(solution_path)

    hours = np.arange(len(pg_df))
    pg_by_fuel = _pg_by_fuel(pg_df, meta)

    fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)

    # ---- Panel 1: generation dispatch (stacked area) ----
    ax_pg = axes[0]
    labels = list(pg_by_fuel.keys())
    values = np.array([pg_by_fuel[lab].reindex(hours, fill_value=0.0).values for lab in labels])
    colors = [FUEL_COLORS.get(lab, "grey") for lab in labels]
    ax_pg.stackplot(hours, values, labels=labels, colors=colors, alpha=0.85)
    ax_pg.set_ylabel("Dispatch (MW)")
    ax_pg.set_title("Generation dispatch by fuel type")
    ax_pg.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
    ax_pg.set_xlim(hours[0], hours[-1])

    # ---- Panel 2: LMP / power-balance dual λ ----
    ax_lmp = axes[1]
    if not lmp_df.empty:
        cmap = plt.colormaps["tab20"].resampled(len(lmp_df.columns))
        for i, bus in enumerate(lmp_df.columns):
            ax_lmp.plot(lmp_df.index, lmp_df[bus], color=cmap(i), linewidth=0.8, alpha=0.7)
        mean_lmp = lmp_df.mean(axis=1)
        ax_lmp.plot(lmp_df.index, mean_lmp, color="black", linewidth=1.5, linestyle="--",
                    label="System avg")
        ax_lmp.legend(loc="upper left", fontsize=8)
    ax_lmp.set_ylabel("λ ($/MWh)")
    ax_lmp.set_title("Power-balance dual (LMP)")

    # ---- Panel 3: branch congestion dual μ ----
    ax_mu = axes[2]
    mu_legend_handles = []
    if not mu_df.empty:
        # Only draw branches that are ever non-zero
        active = mu_df.columns[(mu_df != 0).any(axis=0)]
        inactive = mu_df.columns[(mu_df == 0).all(axis=0)]
        if len(inactive) > 0:
            h_line, = ax_mu.plot(mu_df.index, np.zeros(len(mu_df)), color="lightgrey",
                                 linewidth=0.5, label=f"{len(inactive)} uncongested")
            mu_legend_handles.append(h_line)
        cmap2 = plt.colormaps["Set1"].resampled(max(len(active), 1))
        for i, br in enumerate(active):
            h_line, = ax_mu.plot(mu_df.index, mu_df[br], color=cmap2(i), linewidth=1.2,
                                 label=br, marker="o", markersize=3)
            mu_legend_handles.append(h_line)
        ax_mu.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax_mu.set_ylabel("μ ($/MWh)")
    ax_mu.set_title("Branch congestion dual (+ = upper limit, − = lower limit)")
    ax_mu.set_xlabel("Hour")
    ax_mu.set_xticks(hours)

    fig.tight_layout(rect=[0, 0, 0.84, 1])

    if mu_legend_handles:
        # Anchor flush to actual right edge of axes (after tight_layout moves them)
        pos = ax_lmp.get_position()
        fig.legend(
            handles=mu_legend_handles,
            loc="upper left",
            bbox_to_anchor=(pos.x1 + 0.005, pos.y1),
            bbox_transform=fig.transFigure,
            borderaxespad=0,
            fontsize=7,
            labelspacing=0.2,
            handlelength=1.2,
            handletextpad=0.3,
            title="Branches",
            title_fontsize=8,
            ncol=2,
        )
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot generation dispatch, LMP, and congestion duals against time."
    )
    parser.add_argument(
        "--solution", type=str, default=str(DEFAULT_SOLUTION),
        help="Path to the UC/DCOPF solution JSON file.",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output PDF path. Defaults to <solution_stem>_optimal_values.pdf in results/.",
    )
    args = parser.parse_args()
    solution_path = Path(args.solution)
    out_path = (
        Path(args.out) if args.out
        else RESULTS_DIR / f"{solution_path.stem}_optimal_values.pdf"
    )
    plot_optimal_values(solution_path, out_path)


if __name__ == "__main__":
    main()
