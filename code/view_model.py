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

from utils.solution_io import load_pg, load_generator_metadata, pg_by_fuel

# Paths
REPO_ROOT = Path(__file__).parent.parent
EGRET_DIR = REPO_ROOT / "Egret"
TEST_CASE = EGRET_DIR / "egret" / "models" / "tests" / "uc_test_instances" / "test_scuc_masked.json"
#TEST_CASE = REPO_ROOT / "data" / "four_bus_uc.json"

# Load the test case into an EGRET ModelData object
print(f"Loading {TEST_CASE.name}...")
md = ModelData.read(str(TEST_CASE))

pg_df = load_pg(md.data)
gen_meta = load_generator_metadata(md.data)
info = pg_by_fuel(pg_df, gen_meta)
print(pd.Series({k: v.sum() for k, v in info.items()}))

elements = md.data["elements"]
system = md.data["system"]
for key in system.keys():
    print(f"  {key+':':<12}  {(system.get(key, {}))}")
for key in elements.keys():
    print(f"  {key+':':<12}  {len(elements.get(key, {}))}")
gens = elements["generator"]
print("Generator", type(gens))
print(gens.keys())

load_df = {}
for l, load in elements['load'].items():
    load_df[l] = load["p_load"]["values"]
load_df = pd.DataFrame(load_df)
print(load_df)

bus_stats = {}
for g in elements["generator"].values():
    pmax = g["p_max"]
    if isinstance(pmax, dict):
        pmax = max(pmax["values"])
    row = bus_stats.setdefault(g["bus"], {"n_gens": 0, "p_max_sum": 0.0})
    row["n_gens"] += 1
    row["p_max_sum"] += float(pmax)
print("Generators per bus")
print(pd.DataFrame.from_dict(bus_stats, orient="index").sort_index())

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
