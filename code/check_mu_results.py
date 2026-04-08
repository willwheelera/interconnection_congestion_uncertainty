import numpy as np
from plot_optimal_values import load_lmp_mu
import itertools
from pathlib import Path
import os
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent

def run(ngen = 8, n_avail = 5):

    out = []
    av = np.zeros(ngen, dtype=int)
    for indices in itertools.combinations(range(ngen), n_avail):
        av[:] = 0
        av[list(indices)] = 1
        ID = int("".join(str(b) for b in av), 2)

        DIR = REPO_ROOT / "results" / "gen8_four_bus"
        solution_path = DIR / f"uc_gen8_four_bus_avail{ID}.json"
        if os.path.exists(solution_path):
            lmp_df, mu_df = load_lmp_mu(solution_path)
            a = (mu_df**2).sum(axis=0)**0.5
            a.name = ID
            out.append(a)

    df = pd.concat(out, axis=1).T
    df["total"] = df.sum(axis=1)
    print(df.round(0))
    print(f"{n_avail} congestion \n{(df>0).sum(axis=0) / len(df)}")

run(8, 4)
quit()
run(8, 5)
run(8, 6)
run(8, 3)
