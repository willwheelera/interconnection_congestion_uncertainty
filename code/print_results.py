import json
import re
from pathlib import Path

results_dir = Path(__file__).parent.parent / "results" / "many_gen_cases"

n_infeasible = 0
for path in sorted(results_dir.glob("*.json")):
    match = re.search(r"avail(\d+)", path.stem)
    avail_code = match.group(1) if match else "unknown"
    with open(path) as f:
        data = json.load(f)
    total_cost = data["system"]["total_cost"]
    if total_cost < 1131284.:
        print(f"avail code: {avail_code:>5}  total_cost: {total_cost:.2f}")
    else:
        n_infeasible += 1

print(f"infeasible: {n_infeasible}")
