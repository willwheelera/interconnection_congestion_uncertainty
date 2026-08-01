"""Print a table of branch parameters (resistance, reactance, thermal limit).

Usage:
    uv run python src/print_line_limits.py
    uv run python src/print_line_limits.py --case data/case14_formatted.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from paths import REPO_ROOT
DEFAULT_CASE = REPO_ROOT / "data" / "case14_gen_per_bus.json"


def load_branch_table(case_path: Path) -> pd.DataFrame:
    with open(case_path) as f:
        data = json.load(f)
    branches = data["elements"]["branch"]
    baseMVA = data["system"]["baseMVA"]
    rows = []
    for name in sorted(branches, key=int):
        br = branches[name]
        rows.append({
            "branch": name,
            "from": br["from_bus"],
            "to": br["to_bus"],
            "r (p.u.)": br.get("resistance", 0.0),
            "x (p.u.)": br.get("reactance", 0.0),
            "rating (MW)": br.get("rating_long_term"),
        })
    df = pd.DataFrame(rows).set_index("branch")
    df["thermal loss"] = df["rating (MW)"]**2 * df["r (p.u.)"] / baseMVA**1
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default=str(DEFAULT_CASE))
    args = parser.parse_args()

    df = load_branch_table(Path(args.case))
    print(df.to_string())


if __name__ == "__main__":
    main()
