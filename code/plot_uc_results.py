"""
Plot UC solution results from a saved EGRET ModelData JSON file.

Produces three panels:
  1. Load profiles over time
  2. Generator dispatch stacked by fuel type (committed generators shaded)
  3. System marginal cost proxy (avg production cost per MWh) — the closest
     available quantity to a dual variable for a MIP UC solution

Usage:
    uv run python code/plot_uc_results.py [path/to/uc_results.json]

Defaults to data/uc_results.json if no argument is given.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Load result file
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
DEFAULT_RESULTS = REPO_ROOT / "results" / "uc_test_scuc_masked.json"

result_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS

print(f"Loading {result_path} ...")
with open(result_path) as f:
    data = json.load(f)

elements = data["elements"]
system = data["system"]

time_keys = system["time_keys"]          # list of period labels, e.g. ['1','2',...]
T = len(time_keys)
hours = list(range(1, T + 1))


def ts_values(obj, key):
    """Return time-series values list for obj[key], or None."""
    v = obj.get(key)
    if isinstance(v, dict) and v.get("data_type") == "time_series":
        return v["values"]
    return None


# ---------------------------------------------------------------------------
# 1. Load profiles
# ---------------------------------------------------------------------------
loads = elements.get("load", {})
load_names = list(loads.items())

load_matrix = {}
for name, ld in load_names:
    vals = ts_values(ld, "p_load")
    if vals:
        load_matrix[name] = np.array(vals[:T])

total_load = sum(load_matrix.values()) if load_matrix else np.zeros(T)

# ---------------------------------------------------------------------------
# 2. Generator dispatch by fuel type
# ---------------------------------------------------------------------------
generators = elements.get("generator", {})

fuel_dispatch: dict[str, np.ndarray] = {}
fuel_committed: dict[str, np.ndarray] = {}

for name, gen in generators.items():
    pg = ts_values(gen, "pg")
    commitment = ts_values(gen, "commitment")
    if pg is None:
        continue
    fuel = gen.get("fuel", "Unknown")
    arr = np.array(pg[:T], dtype=float)
    if fuel not in fuel_dispatch:
        fuel_dispatch[fuel] = np.zeros(T)
        fuel_committed[fuel] = np.zeros(T)
    fuel_dispatch[fuel] += arr
    if commitment:
        fuel_committed[fuel] += np.array(commitment[:T], dtype=float)

fuel_order = sorted(fuel_dispatch.keys())

# Color palette keyed by common fuel names
FUEL_COLORS = {
    "Nuclear": "#8B0000",
    "Coal": "#4a4a4a",
    "Gas": "#e08c3a",
    "NG": "#e08c3a",
    "Oil": "#b5651d",
    "Hydro": "#1a75c9",
    "Wind": "#2ecc71",
    "Solar": "#f1c40f",
    "Geothermal": "#9b59b6",
    "Other": "#95a5a6",
    "Unknown": "#bdc3c7",
    "G": "#e08c3a",         # generic gas label used in test fixtures
}
def fuel_color(fuel):
    return FUEL_COLORS.get(fuel, "#" + format(hash(fuel) % 0xFFFFFF, "06x"))

# ---------------------------------------------------------------------------
# 3. Dual variable proxy: system marginal cost per MWh
# ---------------------------------------------------------------------------
# For a MIP UC problem EGRET does not expose LP shadow prices directly.
# We approximate the system marginal price as the weighted-average production
# cost per MWh across all committed generators at each time step.
total_prod_cost = np.zeros(T)
total_pg = np.zeros(T)

for name, gen in generators.items():
    pg = ts_values(gen, "pg")
    pc = ts_values(gen, "production_cost")
    if pg and pc:
        total_pg += np.array(pg[:T], dtype=float)
        total_prod_cost += np.array(pc[:T], dtype=float)

# Avoid divide-by-zero
with np.errstate(invalid="ignore", divide="ignore"):
    marginal_cost = np.where(total_pg > 0, total_prod_cost / total_pg, np.nan)

# Also grab reserve shortfall as a secondary dual-variable proxy
reserve_shortfall = ts_values(system, "reserve_shortfall")
reserve_req = ts_values(system, "reserve_requirement")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Unit Commitment Solution", fontsize=14, fontweight="bold")

# --- Panel 1: Load profiles ---
ax1 = axes[0]
if len(load_matrix) <= 5:
    for name, vals in load_matrix.items():
        ax1.plot(hours, vals, label=name, linewidth=1.2)
    if len(load_matrix) > 1:
        ax1.plot(hours, total_load, "k--", linewidth=1.5, label="Total")
    ax1.legend(fontsize=7, ncol=2)
else:
    ax1.fill_between(hours, total_load, alpha=0.6, color="#1a75c9", label="Total load")
    ax1.plot(hours, total_load, color="#1a75c9", linewidth=1.5)
    ax1.legend(fontsize=8)

ax1.set_ylabel("Load (MW)")
ax1.set_title("Load Profiles")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.grid(axis="y", linestyle=":", alpha=0.5)

# --- Panel 2: Generator dispatch stacked by fuel ---
ax2 = axes[1]
bottoms = np.zeros(T)
for fuel in fuel_order:
    vals = fuel_dispatch[fuel]
    ax2.bar(
        hours, vals, bottom=bottoms,
        label=fuel, color=fuel_color(fuel), width=0.8, alpha=0.85,
    )
    bottoms += vals

ax2.plot(hours, total_load, "k--", linewidth=1.2, label="Total load")
ax2.set_ylabel("Dispatch (MW)")
ax2.set_title("Generator Dispatch by Fuel Type")
ax2.legend(fontsize=8, ncol=3, loc="upper right")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.grid(axis="y", linestyle=":", alpha=0.5)

# --- Panel 3: Dual variable proxy ---
ax3 = axes[2]
ax3.plot(hours, marginal_cost, color="#c0392b", linewidth=1.8, label="Avg production cost ($/MWh)")

if reserve_shortfall is not None and any(v > 0 for v in reserve_shortfall):
    ax3b = ax3.twinx()
    ax3b.bar(hours, reserve_shortfall[:T], color="orange", alpha=0.4, width=0.8, label="Reserve shortfall (MW)")
    ax3b.set_ylabel("Reserve shortfall (MW)", color="orange")
    ax3b.tick_params(axis="y", labelcolor="orange")
    ax3b.legend(loc="upper right", fontsize=8)

ax3.set_ylabel("$/MWh")
ax3.set_title("Marginal Cost Proxy (Dual Variable Approximation)")
ax3.legend(fontsize=8, loc="upper left")
ax3.grid(axis="y", linestyle=":", alpha=0.5)

ax3.set_xlabel("Hour")
ax3.set_xlim(0.5, T + 0.5)
ax3.xaxis.set_major_locator(mticker.MultipleLocator(max(1, T // 12)))

plt.tight_layout()

out_path = result_path.with_suffix(".pdf")
plt.savefig(out_path, bbox_inches="tight")
print(f"Plot saved to {out_path}")
plt.show()
