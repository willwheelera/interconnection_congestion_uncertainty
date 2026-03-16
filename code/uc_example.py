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

# Paths
REPO_ROOT = Path(__file__).parent.parent
EGRET_DIR = REPO_ROOT / "Egret"
TEST_CASE = EGRET_DIR / "egret" / "models" / "tests" / "uc_test_instances" / "test_scuc_masked.json"

# Load the test case into an EGRET ModelData object
print(f"Loading {TEST_CASE.name}...")
md = ModelData.read(str(TEST_CASE))

elements = md.data["elements"]
system = md.data["system"]
for key in elements.keys():
    print(f"  {key+':':<12}  {len(elements.get(key, {}))}")

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
