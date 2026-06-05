import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_dual_distributions import load_dual_cache
from pathlib import Path
import seaborn as sns

REPO_ROOT = Path(__file__).parent.parent
CASE_DIR = REPO_ROOT / "results" / "case14_gen_per_bus"



def main():
    _, mu_all = load_dual_cache(CASE_DIR)
    mu = mu_all.abs().sum(axis=1)
    #print(mu[:20])
    #for i in [83, 84, 120, 240, 360, 480, 600]:
    #    print(f"greater than {i}: {(mu > i).sum()}")
    ascent = find_max_dir(mu)
    print("check zero", sum(mu[ascent].values < mu.values))
    extrema = (ascent.index==ascent.values)
    print(mu[extrema])

    df = pd.DataFrame(index=ascent.index, data=ascent.values, columns=[1])
    x = ascent.copy()
    for i in range(2, 20):
        done = x.values == ascent[x].values
        print(f"{i} done {done.sum()}")
        if done.sum() == len(x):
            break
        x = ascent[x]
        df[i] = x.values
    bas, counts = np.unique(df[i-1], return_counts=True)
    print()
    basins = pd.DataFrame(index=bas, data=counts, columns=["counts"])
    basins.index.name = "basins"
    basins["mu"] = mu[bas]
    basins["avg"] = [mu[df[i-1]==v].mean() for v in basins.index]
    print(basins)
    sns.pairplot(basins)
    plt.show()
    


def find_max_dir(mu):
    ascent = np.array(mu.index).copy()
    for i in range(14):
        neighbor = np.array(mu.index) ^ 1<<i
        valid = np.isin(neighbor, mu.index)
        bigger = mu[neighbor[valid]].values > mu[ascent[valid]].values
        replace = valid
        replace[valid] = bigger
        ascent[replace] = neighbor[replace]
    return pd.Series(index=mu.index, data=ascent)


def all_vectors(n):
    powers = (2 ** np.arange(n)).astype(int)
    index = np.arange(int(2 ** n), dtype=int)
    vecs = ((index[:, np.newaxis] & powers[::-1]) != 0).astype(int) # all possible availability vectors
    df = pd.DataFrame(index=index, data=vecs)
    return df


if __name__ == "__main__":
    main()
