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
    #for indices in itertools.combinations(range(ngen), n_avail):
    #    av[:] = 0
    #    av[list(indices)] = 1
    for i in range(int(2**ngen)):
        av = list(map(int, np.binary_repr(i, ngen)))
        if np.sum(av) < 3:
            continue
        ID = int("".join(str(b) for b in av), 2)

        DIR = REPO_ROOT / "results" / "gen8_four_bus"
        solution_path = DIR / f"uc_gen8_four_bus_avail{ID}.json"
        if os.path.exists(solution_path):
            lmp_df, mu_df = load_lmp_mu(solution_path)
            if np.any(lmp_df < 0):
                print("negative lmp0", ID)
                if np.any(lmp_df[1:] < -500):
                    print("\t other negatives \n", lmp_df)
            a = (mu_df**2).sum(axis=0)**0.5
            a.name = ID
            out.append(a)

    df = pd.concat(out, axis=1).T
    zeros = df.sum(axis=1) < 1e-3
    print(df[~zeros].round(0))
    threshold = [0, 1e-2, 1e-1]
    congestion = [(df>eps).sum(axis=0) / len(df) for eps in threshold]
    for c,t in zip(congestion, threshold): c.name = f">{t}"
    congestion = pd.concat(congestion, axis=1)
    congestion.loc["total"] = congestion.sum(axis=0)
    print(len(df))
    print(f"avail {n_avail}, congestion \n{congestion}")

run(8, "all")
