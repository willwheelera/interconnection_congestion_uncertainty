import numpy as np
import pandas as pd
import sys


def display(df):
    df = pd.read_parquet(fname)
    print(df)
    print(df.columns.get_level_values(0).unique())
    print(df.index)
    print()

if __name__ == "__main__":
    df = pd.read_parquet("../results/case14_gen_per_bus/dual_cache.parquet")
    sens = pd.read_parquet("../results/case14_gen_per_bus/sensitivity_cache.parquet")
    branches = sens.columns.get_level_values("branch").unique().tolist()
    buses = sens.columns.get_level_values("bus").unique().tolist()
    print(branches)
    print(buses)
    print(df["mu"].columns)
    print(df["mu"].reindex(columns=branches).columns)
    
