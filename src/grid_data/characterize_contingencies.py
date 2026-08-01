"""
Characterize the contingencies that drive tail congestion, by their effects.

A shadow-price row (np6-86) pairs a monitored ConstraintName (the line that
overloads) with a ContingencyName (the element assumed out). ERCOT's contingency
names are coded abbreviations -- prefix S/D/M/X/U for single/double/multiple/
special element outages, trailing digit for voltage class -- and they do not map
to the physical element in the public data. So we profile each contingency by
what it *does*: how often and how hard it binds, and which monitored lines it
overloads (with their voltage). This is the input for deciding which contingencies
to reproduce in a test case, and whether a small set covers the tail.

Important -- these are *hypothetical* security constraints, not realized outages. A
nonzero contingency shadow price means SCED is dispatching to protect against that
outage and the protection is binding now; the element is not actually out.

Tail-event definition: conceptually we form the grid of (5-min interval) x (937
contingencies) -- one number per cell, the max shadow price over all lines for
that contingency in that interval. A *tail event* is a cell above TAIL ($300); we
record the (interval, contingency) pairs. Counts below are over these pairs, so a
contingency loading several lines at once in one interval still counts once.

Uses the full-year base scan (shadow_prices_2025.parquet), contingency rows only.

Outputs (in results/):
    contingency_profile.csv        per-contingency effect profile, tail-ranked
    contingency_characterization.png   top contingencies + tail concentration

Usage:
    uv run python src/grid_data/characterize_contingencies.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_shadow_price_distributions import DATA_PATH, load_shadow_prices

OUT_DIR = Path(__file__).parent.parent.parent / "results"
TAIL = 300.0  # $/MWh; a binding interval above this counts as a tail event
TOP_N = 15
TYPE_COLORS = {"S": "tab:blue", "D": "tab:red", "M": "tab:green",
               "X": "tab:orange", "U": "tab:purple"}


def binding_contingencies(df: pd.DataFrame) -> pd.DataFrame:
    """Contingency rows (not base case) that are actually binding (ShadowPrice>0)."""
    cont = df[df["ContingencyName"].str.strip() != "BASE CASE"]
    return cont[cont["ShadowPrice"] > 0]


def top_line_per_contingency(binding: pd.DataFrame) -> pd.DataFrame:
    """For each contingency, the monitored line it loads most and that line's kV."""
    pair_mu = binding.groupby(["ContingencyName", "ConstraintName"])["ShadowPrice"].sum()
    top = pair_mu.groupby("ContingencyName").idxmax().str[1].rename("top_line")
    kv = binding.groupby("ConstraintName")["FromStationkV"].first()
    return pd.DataFrame({"top_line": top, "top_line_kv": top.map(kv)})


def profile(binding: pd.DataFrame) -> pd.DataFrame:
    """Per-contingency effect profile, ranked by tail-event count.

    Metrics use the per-(interval, contingency) max shadow price over lines, so an
    interval counts once however many lines a contingency loads. Columns: ctype,
    tail_events, bind_intervals, max_mu, total_mu, n_lines, top_line, top_line_kv.
    """
    per_ic = binding.groupby(["SCEDTimestamp", "ContingencyName"])["ShadowPrice"].max()
    g = per_ic.groupby("ContingencyName")
    prof = pd.DataFrame({
        "ctype": g.size().index.str[0],
        "bind_intervals": g.size(),
        "tail_events": (per_ic > TAIL).groupby("ContingencyName").sum(),
        "max_mu": g.max(),
        "total_mu": g.sum(),
        "n_lines": binding.groupby("ContingencyName")["ConstraintName"].nunique(),
    })
    prof = prof.join(top_line_per_contingency(binding))
    prof["tail_events"] = prof["tail_events"].astype(int)
    return prof.sort_values("tail_events", ascending=False)


def concentration(prof: pd.DataFrame) -> pd.Series:
    """Cumulative share of tail events vs number of contingencies (descending)."""
    tail = prof["tail_events"].sort_values(ascending=False)
    return (tail.cumsum() / tail.sum()).reset_index(drop=True)


def plot_characterization(prof: pd.DataFrame, conc: pd.Series, path: Path) -> None:
    """Top contingencies by tail events, plus the tail-concentration curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    top = prof.head(TOP_N).iloc[::-1]
    colors = [TYPE_COLORS.get(t, "gray") for t in top["ctype"]]
    axes[0].barh(range(len(top)), top["tail_events"], color=colors)
    axes[0].set_yticks(range(len(top)),
                       [f"{c}  →{l}" for c, l in zip(top.index, top["top_line"])],
                       fontsize=8)
    axes[0].set_xlabel(f"tail events (μ > ${TAIL:.0f})")
    axes[0].set_title("Contingencies driving the tail\n(label: contingency → top line loaded)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TYPE_COLORS.values()]
    axes[0].legend(handles, TYPE_COLORS.keys(), title="type code", fontsize=8)

    axes[1].plot(range(1, len(conc) + 1), conc.values, color="k")
    n80 = int((conc < 0.8).sum() + 1)
    axes[1].axhline(0.8, color="gray", ls="--", lw=0.8)
    axes[1].axvline(n80, color="tab:red", ls="--", lw=0.8)
    axes[1].set_xlabel("number of contingencies (ranked)")
    axes[1].set_ylabel("cumulative share of tail events")
    axes[1].set_title(f"Tail concentration: {n80} contingencies cover 80%")
    axes[1].set_xlim(0, min(120, len(conc)))
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    binding = binding_contingencies(load_shadow_prices(DATA_PATH))
    prof = profile(binding)
    conc = concentration(prof)

    n_tail = (prof["tail_events"] > 0).sum()
    print(f"{len(prof)} binding contingencies; {n_tail} produce tail events (μ>${TAIL:.0f})")
    print(f"total tail events: {int(prof['tail_events'].sum())}")
    print(f"{int((conc < 0.8).sum() + 1)} contingencies cover 80% of tail events\n")
    print("type share of tail events:")
    print((prof.groupby("ctype")["tail_events"].sum()
           / prof["tail_events"].sum()).round(3).to_string())
    print("\ntop contingencies by tail events:")
    print(prof.head(TOP_N)[["ctype", "tail_events", "max_mu", "n_lines",
                            "top_line", "top_line_kv"]].round(1).to_string())

    OUT_DIR.mkdir(exist_ok=True)
    prof.to_csv(OUT_DIR / "contingency_profile.csv")
    plot_characterization(prof, conc, OUT_DIR / "contingency_characterization.png")
    print(f"\nsaved profile and figure to {OUT_DIR}")


if __name__ == "__main__":
    main()
