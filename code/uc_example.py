"""
Example: Solve security-constrained unit commitment (SCUC) with EGRET.

Uses the 24-bus IEEE RTS test case bundled with EGRET
(Egret/egret/models/tests/uc_test_instances/test_scuc_masked.json).

Solver: Gurobi
"""

import os
from pathlib import Path

from egret.data.model_data import ModelData
from egret.models.unit_commitment import solve_unit_commitment

import json
import pandas as pd
import numpy as np

# Paths
REPO_ROOT = Path(__file__).parent.parent
EGRET_DIR = REPO_ROOT / "Egret"
TEST_CASE = EGRET_DIR / "egret" / "models" / "tests" / "uc_test_instances" / "test_scuc_masked.json"

# Load the test case into an EGRET ModelData object
print(f"Loading {TEST_CASE.name}...")
md = ModelData.read(str(TEST_CASE))

#################################################
FUEL_LABELS = {
    "C": "Coal",
    "G": "Gas CC",
    "N": "Nuclear",
    "O": "Oil / CT",
    "H": "Hydro",
    "S": "Solar",
    "W": "Wind",
}

def load_pg(raw) -> pd.DataFrame:
    """Return (n_hours × n_generators) DataFrame of pg dispatch in MW."""
    gens = raw["elements"]["generator"]
    n_hours = len(raw["system"]["time_keys"])
    data = {}
    for name, g in gens.items():
        pg = g.get("pg", 0.0)
        data[name] = pg["values"] if isinstance(pg, dict) else [float(pg)] * n_hours
    return pd.DataFrame(data, index=range(n_hours))


def _pg_by_fuel(pg_df: pd.DataFrame, meta: pd.DataFrame) -> dict[str, pd.Series]:
    """Aggregate pg by fuel type label, returning {label: hourly_total_MW}."""
    result = {}
    for fuel_code, label in FUEL_LABELS.items():
        gens = meta.index[meta["fuel"] == fuel_code].tolist()
        cols = [g for g in gens if g in pg_df.columns]
        if cols:
            result[label] = len(cols)#pg_df[cols].sum(axis=1)
    return result

def load_generator_metadata(raw) -> pd.DataFrame:
    """Return DataFrame with 'fuel' and 'generator_type' columns indexed by generator name."""
    gens = raw["elements"]["generator"]
    return pd.DataFrame(
        {name: {"fuel": g.get("fuel", "?"), "type": g.get("generator_type", "?")}
         for name, g in gens.items()}
    ).T


pg_df = load_pg(md.data)
gen_meta = load_generator_metadata(md.data)
info = _pg_by_fuel(pg_df, gen_meta)
print(pd.Series(info))

elements = md.data["elements"]
system = md.data["system"]
for key in system.keys():
    print(f"  {key+':':<12}  {(system.get(key, {}))}")
for key in elements.keys():
    print(f"  {key+':':<12}  {len(elements.get(key, {}))}")
gens = elements["generator"]
print("Generator", type(gens))
print(gens.keys())

pmax_df = []
table = {k: [] for k in "CGNOHSW"}
table["RTPV"] = []
print("p_max")
for key, val in elements["generator"].items():
    pmax = val["p_max"]
    if hasattr(pmax, "__len__"):
        pmax = np.around(np.amax(pmax["values"]))
    if "RTPV" in key:
        table["RTPV"].append(pmax)
    else:
        table[val["fuel"]].append(pmax)
    pmax_df.append(dict(name=key, fuel=val["fuel"], p_max=pmax))
#pmax_df = pd.DataFrame(pmax_df)
table = pd.DataFrame({key: pd.Series(value) for key, value in table.items()})
print(table)
    
quit()

#################################################
# Solve unit commitment using the Gurobi solver
print("Solving unit commitment...")
md_sol = solve_unit_commitment(
    md,
    solver="gurobi",
    mipgap=0.01,
    timelimit=300,
    solver_tee=True,
)

# Report results
total_cost = md_sol.data["system"]["total_cost"]
print(f"\nObjective value (total cost): ${total_cost:,.2f}")

# Summarize generator dispatch for the first few time periods
generators = md_sol.data["elements"]["generator"]
print("\nGenerator dispatch summary (first 3 time periods):")
for gen_name, gen_data in list(generators.items())[:5]:
    pg = gen_data.get("pg", {}).get("values", [])
    print(f"  {gen_name}: {[round(v, 1) for v in pg[:3]]} MW")

# Save results
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
case_id = TEST_CASE.stem
RESULTS_FILE = RESULTS_DIR / f"uc_{case_id}.json"
md_sol.write(str(RESULTS_FILE))
print(f"\nResults saved to {RESULTS_FILE}")
