"""Audit whether the slimmed ORDC relief set is complete for the enumerated sweep.

The full 2^20 sweep solves every config under a $3500 soft cap plus a fixed nodal
relief set. The soft cap lets a monitored line overload rather than force strict
thermal feasibility, so a small overload is normal cap behavior. A *large* overload
or any real system-wide load shed flags a config the slimmed relief set could not
serve -- one that would need a relief bus outside the chosen set. This streams the
snapmax_cache (per config, per snapshot: peak mu, worst overload, fallback shed) and
summarizes cap saturation, overload, and shed over all configs.

Usage:
    uv run python src/texas/audit_relief_completeness.py --case activsg2000_original_v12
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import REPO_ROOT
CAP = 3500.0            # soft shadow-price cap ($/MWh)
CAP_TOL = 0.5           # a cell within this of the cap counts as pinned
SHED_TOL = 1.0          # load shed above this (MW) is a real fallback, not solver noise
OVERLOAD_FLAG = 50.0    # overload above this (MW) flags an incomplete relief set


def stream_snapmax(snapmax_dir):
    """Per-config max over snapshots of (cap-pinned?, overload MW, shed MW), streamed."""
    parts = sorted(glob.glob(str(Path(snapmax_dir) / "part_*.parquet")))
    frames = []
    for part in parts:
        d = pd.read_parquet(part, columns=["id", "max_mu", "max_overload_mw", "load_shed_mw"])
        d["pinned"] = (d["max_mu"] >= CAP - CAP_TOL)
        g = d.groupby("id")
        frames.append(pd.DataFrame({"pinned": g["pinned"].max(),
                                    "overload_mw": g["max_overload_mw"].max(),
                                    "shed_mw": g["load_shed_mw"].max()}))
    return pd.concat(frames).sort_index()


def summarize(per_config):
    """Print cap-saturation, overload, and load-shed statistics over all configs."""
    n = len(per_config)
    pinned = per_config["pinned"].sum()
    shed = per_config["shed_mw"] > SHED_TOL
    flagged = (per_config["overload_mw"] > OVERLOAD_FLAG) | shed
    print(f"configs                     : {n}")
    print(f"any snapshot pinned at cap  : {pinned} ({pinned / n:.1%})")
    print(f"max overload the cap allowed: {per_config['overload_mw'].max():.1f} MW")
    print(f"configs overload > {OVERLOAD_FLAG:.0f} MW    : {(per_config['overload_mw'] > OVERLOAD_FLAG).sum()}")
    print(f"real load shed > {SHED_TOL:.0f} MW       : {shed.sum()}  (max {per_config['shed_mw'].max():.2f} MW)")
    print(f"RELIEF-INCOMPLETE configs   : {flagged.sum()}")
    return flagged


def plot_overload(per_config, out):
    """Histogram of per-config worst overload, with the flag threshold marked."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(per_config["overload_mw"], bins=80, color="tab:blue", log=True)
    ax.axvline(OVERLOAD_FLAG, color="tab:red", ls="--", lw=1.2,
               label=f"relief-incomplete flag = {OVERLOAD_FLAG:.0f} MW")
    ax.set_xlabel("worst thermal overload the $3500 cap allowed (MW)")
    ax.set_ylabel("configs")
    ax.set_title("Relief-set completeness: per-config worst overload")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_original_v12")
    args = parser.parse_args()
    case_dir = REPO_ROOT / "results" / args.case
    per_config = stream_snapmax(case_dir / "snapmax_cache")
    summarize(per_config)
    plot_overload(per_config, case_dir / f"relief_completeness_{args.case}.png")
    plt.show()


if __name__ == "__main__":
    main()
