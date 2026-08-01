import gzip
import json

import pandas as pd

FNAME = "../results/case14_gen_per_bus/opf_case14_gen_per_bus_avail1000.json.gz"


def elements_to_frame(elements):
    """Convert a {name: {attr: value, ...}, ...} dict to a DataFrame indexed by name."""
    return pd.DataFrame.from_dict(elements, orient="index")


def main():
    with gzip.open(FNAME, "rt") as f:
        data = json.load(f)

    print(f"File: {FNAME}")
    print(f"Top-level keys: {list(data.keys())}\n")

    system = pd.Series(data["system"], name="system")
    print("=== system ===")
    print(system)
    print()

    for kind, elements in data["elements"].items():
        df = elements_to_frame(elements)
        print(f"=== elements / {kind} ({len(df)} rows, {len(df.columns)} cols) ===")
        if kind == "generator":
            df.drop(columns="p_cost")
            df = df.T
        print(df)
        print()


if __name__ == "__main__":
    main()
