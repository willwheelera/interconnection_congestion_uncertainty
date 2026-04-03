"""
Plot optimal primal and dual variables from a UC solution against time.

Four panels:
  1. Generation dispatch (MW) — stacked area chart by fuel type
  2. Power-balance duals λ (LMP, $/MWh) — one line per bus
  3. Branch congestion duals μ ($/MWh) — one line per branch;
     positive = upper limit binding, negative = lower limit binding
  4. Line limit slack (MW) — rating − |pf| per branch; zero = congested

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
    """Load per-hour LMP and branch congestion duals from the UC solution file.

    Reads ``lmp`` from bus elements and ``mu_upper`` / ``mu_lower`` from branch
    elements (stored by ``solve_uc.py``).

    Returns:
        lmp_df  — (n_hours × n_buses) DataFrame of λ ($/MWh)
        mu_df   — (n_hours × n_branches) DataFrame of μ ($/MWh),
                  positive = upper limit binding, negative = lower binding
    """
    raw = json.loads(solution_path.read_text())
    n_hours = len(raw["system"]["time_keys"])
    buses = raw["elements"]["bus"]
    branches = raw["elements"]["branch"]

    lmp_data = {}
    for name, b in buses.items():
        lmp = b.get("lmp")
        if lmp is None:
            continue
        lmp_data[name] = lmp["values"] if isinstance(lmp, dict) else [float(lmp)] * n_hours

    mu_data = {}
    for name, br in branches.items():
        mu_u = br.get("mu_upper")
        mu_l = br.get("mu_lower")
        if mu_u is None or mu_l is None:
            continue
        upper = mu_u["values"] if isinstance(mu_u, dict) else [float(mu_u)] * n_hours
        lower = mu_l["values"] if isinstance(mu_l, dict) else [float(mu_l)] * n_hours
        mu_data[name] = [u - l for u, l in zip(upper, lower)]

    lmp_df = pd.DataFrame(lmp_data, index=range(n_hours))
    mu_df = pd.DataFrame(mu_data, index=range(n_hours))
    return lmp_df, mu_df


def load_branch_slacks(solution_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (slack_df, pf_df) from the UC solution file.

    slack = rating − |pf| per branch per hour (MW); zero = congested.
    """
    raw = json.loads(solution_path.read_text())
    n_hours = len(raw["system"]["time_keys"])
    branches = raw["elements"]["branch"]

    pf_data = {}
    slack_data = {}
    for name, br in branches.items():
        pf = br.get("pf")
        if pf is None:
            continue
        pf_vals = pf["values"] if isinstance(pf, dict) else [float(pf)] * n_hours
        pf_data[name] = pf_vals
        rating = br.get("rating_long_term")
        if rating is not None:
            slack_data[name] = [rating - abs(v) for v in pf_vals]

    pf_df = pd.DataFrame(pf_data, index=range(n_hours))
    slack_df = pd.DataFrame(slack_data, index=range(n_hours))
    return slack_df, pf_df


def _pg_by_fuel(pg_df: pd.DataFrame, meta: pd.DataFrame) -> dict[str, pd.Series]:
    """Aggregate pg by fuel type label, returning {label: hourly_total_MW}."""
    result = {}
    for fuel_code, label in FUEL_LABELS.items():
        gens = meta.index[meta["fuel"] == fuel_code].tolist()
        cols = [g for g in gens if g in pg_df.columns]
        if cols:
            result[label] = pg_df[cols].sum(axis=1)
    return result


def plot_optimal_values(solution_path: Path, out_path: Path, by_fuel_type=True) -> None:
    pg_df = load_pg(solution_path)
    meta = load_generator_metadata(solution_path)
    lmp_df, mu_df = load_lmp_mu(solution_path)
    slack_df, pf_df = load_branch_slacks(solution_path)
    #print(lmp_df)
    #print(pf_df)

    hours = np.arange(len(pg_df))
    pg_by_fuel = _pg_by_fuel(pg_df, meta)

    fig, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True)

    # ---- Panel 1: generation dispatch (stacked area) ----
    ax_pg = axes[0]
    _f = lambda x: x.reindex(hours, fill_value=0.0).values
    def _color_f(x):
        theta = x * np.pi * 2
        a = np.arange(3) * 2 * np.pi / 3
        b = np.cos(a + theta) * 0.5 + 0.5
        return tuple(b)

    if by_fuel_type:
        labels = list(pg_by_fuel.keys())
        values = np.array([_f(pg_by_fuel[lab]) for lab in labels])
        colors = [FUEL_COLORS.get(lab, "grey") for lab in labels]
    else:
        labels = list(pg_df.columns)
        values = np.array([_f(pg_df[lab]) for lab in labels])
        n = len(labels)
        colors = [_color_f(i/n) for i in range(n)]
        
    ax_pg.stackplot(hours, values, labels=labels, colors=colors, alpha=0.85)
    ax_pg.set_ylabel("Dispatch (MW)")
    ax_pg.set_title("Generation dispatch by fuel type")
    ax_pg.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8, ncol=2)
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

    # ---- Panel 4: line limit slack = rating - |pf| ----
    ax_slack = axes[3]
    if not slack_df.empty:
        cmap3 = plt.colormaps["tab10"].resampled(max(len(slack_df.columns), 1))
        for i, br in enumerate(slack_df.columns):
            ax_slack.plot(slack_df.index, slack_df[br], color=cmap3(i),
                          linewidth=1.2, marker="o", markersize=3, label=br)
        ax_slack.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_slack.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
                        borderaxespad=0, fontsize=8, title="Branches", title_fontsize=8)
    ax_slack.set_ylabel("Slack (MW)")
    ax_slack.set_title("Line limit slack (rating − |pf|);  0 = congested")
    ax_slack.set_xlabel("Hour")
    ax_slack.set_xticks(hours)

    fig.tight_layout(rect=[0, 0, 0.84, 1])

    if mu_legend_handles:
        # Anchor flush to actual right edge of axes (after tight_layout moves them)
        pos = ax_mu.get_position()
        fig.legend(
            handles=mu_legend_handles,
            loc="lower left",
            bbox_to_anchor=(pos.x1 + 0.005, pos.y0),
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
    plot_optimal_values(solution_path, out_path, False)

def run(ID):
    DIR = RESULTS_DIR / "many_gen_cases"
    solution_path = DIR / f"uc_many_gen_four_bus_uc_avail{ID}.json"
    out_path = DIR / f"optimal_{solution_path.stem}.pdf"
    plot_optimal_values(solution_path, out_path, False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            run(int(a))
    #run(65535)
    #run(63282)
    #run(11778)
    #run(16394)
    #run(672)
