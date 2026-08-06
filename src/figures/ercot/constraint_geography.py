"""
Visualize where ERCOT tail congestion lives geographically.

Two views of the 2025 contingency tail (μ > $300), region-tagged by
constraint_geography.py:
  1. bar chart of tail exposure per ERCOT weather zone,
  2. bubble map of tail constraints over a coarse Texas outline.

Both answer the same question: is high-μ congestion in Far West, in the urban
centers, or somewhere else? Coordinates are town-level approximations; the map is
for regional structure, not siting.

Usage:
    uv run python src/grid_data/plot_constraint_geography.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from src.ercot.constraint_geography import (ZONE_ORDER, attach_geography,
                                  load_tail_constraints)

from src.ercot.data_paths import RESULTS as OUT_DIR
ZONE_COLORS = {
    "FarWest": "#8c1d40", "West": "#e0611a", "North": "#c9a227",
    "NorthCentral": "#4c9f38", "SouthCentral": "#1f8a8a",
    "Southern": "#2b6cb0", "Coast": "#6b4c9a", "East": "#7a7a7a",
    "Unmapped": "#c9c9c9",
}

# coarse Texas border (lon, lat), for map context only.
TEXAS_OUTLINE = [
    (-106.5, 31.8), (-103.0, 32.0), (-103.0, 36.5), (-100.0, 36.5),
    (-100.0, 34.56), (-99.4, 34.4), (-97.5, 33.9), (-96.0, 33.8),
    (-94.5, 33.6), (-94.0, 33.0), (-94.0, 31.0), (-93.5, 30.1),
    (-93.8, 29.7), (-95.0, 29.2), (-96.5, 28.4), (-97.4, 27.3),
    (-97.2, 26.0), (-97.5, 25.9), (-99.1, 26.4), (-99.5, 27.5),
    (-100.4, 28.5), (-101.4, 29.8), (-102.3, 29.9), (-102.8, 29.3),
    (-103.3, 29.0), (-104.5, 29.7), (-104.9, 30.6), (-106.5, 31.8),
]


def plot_zone_bar(prof, ax) -> None:
    """Total tail-μ per ERCOT weather zone, west-to-east, colored by zone."""
    z = prof.groupby("zone")["total_mu"].sum()
    zones = [x for x in ZONE_ORDER if x in z.index]
    vals = [z[x] for x in zones]
    total = z.sum()
    bars = ax.bar(zones, [v / 1e6 for v in vals], color=[ZONE_COLORS[x] for x in zones])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v / 1e6, f"{v/total:.0%}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(r"total tail-μ  (\$M, summed over μ > \$300 intervals, 2025)")
    ax.set_title("Tail congestion by ERCOT weather zone (μ-weighted)")
    ax.tick_params(axis="x", rotation=30)


def plot_texas_map(prof, ax) -> None:
    """Bubble map: mapped constraints at town coords, area ∝ tail events."""
    xs, ys = zip(*TEXAS_OUTLINE)
    ax.fill(xs, ys, color="#f2f0eb", zorder=0)
    ax.plot(xs, ys, color="#999", lw=1, zorder=1)

    mapped = prof[prof["zone"] != "Unmapped"]
    loc = mapped.groupby(["lat", "lon", "zone"])["total_mu"].sum().reset_index()
    ax.scatter(loc["lon"], loc["lat"], s=loc["total_mu"] / loc["total_mu"].max() * 900,
               c=[ZONE_COLORS[z] for z in loc["zone"]],
               alpha=0.7, edgecolors="k", linewidths=0.4, zorder=2)

    for zone in ZONE_ORDER:
        if zone in loc["zone"].values:
            ax.scatter([], [], c=ZONE_COLORS[zone], label=zone, s=60)
    ax.legend(title="zone", fontsize=8, loc="lower left")
    ax.set_aspect(1.15)
    ax.set_xlabel("longitude"), ax.set_ylabel("latitude")
    unmapped = prof[prof["zone"] == "Unmapped"]["total_mu"].sum()
    ax.set_title(f"Tail constraints on the map, bubble area ∝ tail-μ "
                 f"({unmapped/prof['total_mu'].sum():.0%} unmapped, not shown)")


def main() -> None:
    prof = attach_geography(load_tail_constraints())
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_zone_bar(prof, axes[0])
    plot_texas_map(prof, axes[1])
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "constraint_geography.png", dpi=150)
    print(f"saved figure to {OUT_DIR / 'constraint_geography.png'}")
    plt.show()


if __name__ == "__main__":
    main()
