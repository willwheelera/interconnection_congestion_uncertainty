"""Test whether sensitivity predicts mu changes under generator-availability flips.

The cached sensitivity matrix S(x) holds, for one base configuration x,

    S(x)_{l, j}  =  d mu_l / d load_j   evaluated at the OPF solution of x.

Subtracting load at bus j is the same as adding injection there, so flipping
generator k on->off (at bus b = bus(k)) is approximately an *increase* in
load at bus b by the amount that generator was producing.  The first-order
prediction is therefore

    mu_l(y) - mu_l(x)  ~  sign_flip * |delta P_k| * S(x)_{l, b(k)},

where sign_flip = +1 if x_k=1 -> y_k=0 (gen drops out, load effectively
rises), and sign_flip = -1 in the other direction.  We have no |delta P_k|
on hand, so this script just looks at whether the *signed* delta mu lines
up with the *signed* sensitivity entry, lumping the |delta P| factor into
the scatter.

Picks one base configuration, gathers its 14 one-flip neighbors, and plots
(S, delta mu) for every (branch l, flipped generator k) pair.  Each point
is one branch's dual change for one neighbor; color encodes flip direction.

Usage:
    uv run python code/plot_sensitivity_vs_flip.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.availability import id_to_avail, avail_to_id

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "case14_gen_per_bus"
CASE_PATH = REPO_ROOT / "data" / "case14_gen_per_bus.json"
NGEN = 14
BASE_AVAIL_ID = 127


def load_gen_bus_map(case_path: Path) -> list[str]:
    """Return bus name for each generator, in the canonical generator order."""
    with open(case_path) as f:
        case = json.load(f)
    gens = case["elements"]["generator"]
    return [str(g["bus"]) for g in gens.values()]


def neighbor_ids(avail_id: int) -> list[int]:
    """All NGEN configs differing from avail_id at exactly one bit."""
    x = id_to_avail(avail_id, NGEN)
    out = []
    for k in range(NGEN):
        y = x.copy()
        y[k] = 1 - y[k]
        out.append(avail_to_id(y))
    return out


def collect_points(base_id: int, mu: pd.DataFrame, sens: pd.DataFrame,
                   gen_buses: list[str]) -> pd.DataFrame:
    """Build the (sensitivity, delta_mu) scatter for all neighbors of base_id.

    One row per (flipped generator k, branch l) pair.
    """
    x = id_to_avail(base_id, NGEN)
    mu_x = mu.loc[base_id]
    S_x = sens.loc[base_id].unstack("bus")           # (branch, bus)
    branches = S_x.index.tolist()
    rows = []
    for k in range(NGEN):
        y_id = avail_to_id(np.where(np.arange(NGEN) == k, 1 - x, x))
        if y_id not in mu.index:
            continue
        mu_y = mu.loc[y_id]
        # flip on->off (gen leaves) is +1 (load effectively goes up)
        flip_sign = +1 if x[k] == 1 else -1
        bus_k = gen_buses[k]
        for l in branches:
            rows.append({
                "gen": k + 1,
                "bus": bus_k,
                "branch": l,
                "flip_sign": flip_sign,
                "S": float(S_x.loc[l, bus_k]),
                "dmu": float(mu_y[l] - mu_x[l]),
            })
    return pd.DataFrame(rows)


def report_metrics(S_signed: np.ndarray, dmu: np.ndarray, label: str,
                   eps: float = 1e-6) -> None:
    """Print zero fractions, Spearman rho, and sign-agreement on non-trivial points.

    A point is "non-trivial" when both |S_signed| > eps and |dmu| > eps; the
    Spearman rank correlation and sign-agreement rate are reported on that
    subset only.  Zero-sensitivity and zero-dmu fractions are reported
    separately because they describe different failure modes: a zero
    sensitivity means the linearization says "this flip can't move mu",
    while a zero dmu means the flip didn't actually move it.
    """
    from scipy.stats import spearmanr
    n = len(S_signed)
    zero_S = float(np.mean(np.abs(S_signed) <= eps))
    zero_dmu = float(np.mean(np.abs(dmu) <= eps))
    both_zero = float(np.mean((np.abs(S_signed) <= eps) & (np.abs(dmu) <= eps)))
    m = (np.abs(S_signed) > eps) & (np.abs(dmu) > eps)
    if m.sum() >= 2:
        rho, _ = spearmanr(S_signed[m], dmu[m])
        agree = float(np.mean(np.sign(S_signed[m]) == np.sign(dmu[m])))
    else:
        rho, agree = float("nan"), float("nan")
    print(f"  [{label}] n={n:,}")
    print(f"    zero |S|     = {zero_S:.3f}")
    print(f"    zero |dmu|   = {zero_dmu:.3f}")
    print(f"    both zero    = {both_zero:.3f}")
    print(f"    non-trivial  = {int(m.sum()):,}/{n:,} ({m.mean():.3f})")
    print(f"    Spearman rho = {rho:+.3f}   (non-trivial subset)")
    print(f"    sign-agree   = {agree:.3f}   (non-trivial subset)")


def plot_points(pts: pd.DataFrame, base_id: int, out_path: Path) -> None:
    """Scatter of delta_mu vs (flip-signed) sensitivity, colored by direction."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x_signed = pts["flip_sign"] * pts["S"]
    branches = sorted(pts["branch"].unique(), key=int)
    cmap = plt.get_cmap("tab20", len(branches))
    for i, br in enumerate(branches):
        m = pts["branch"] == br
        ax.scatter(x_signed[m], pts["dmu"][m],
                   s=18, alpha=0.7, color=cmap(i), label=f"{br}")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"flip_sign $\cdot\; \partial \mu_l / \partial \mathrm{load}_{b(k)}$  at base $x$")
    ax.set_ylabel(r"$\mu_l(y_k) - \mu_l(x)$")
    pop = int(id_to_avail(base_id, NGEN).sum())
    ax.set_title(
        f"avail_id={base_id} (pop={pop}/{NGEN}), "
        f"one-flip neighbors, {len(pts)} (branch, gen) points"
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, ncol=1, frameon=False, title="branch")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Wrote {out_path}")
    report_metrics(x_signed.values, pts["dmu"].values, f"avail_id={base_id}")


def collect_points_all(mu: pd.DataFrame, sens: pd.DataFrame,
                       gen_buses: list[str]) -> pd.DataFrame:
    """Same as collect_points, but stacked across every cached config.

    Vectorized: decodes all availability vectors, computes the 14 one-flip
    neighbor ids per config by integer XOR (bit k of the int corresponds to
    avail position k via the MSB-first encoding in avail_to_id), looks up
    neighbor mu rows, and picks the sensitivity column for the bus that
    hosts the flipped generator.  Configs whose neighbor isn't in the
    cache (e.g. the all-off id 0) are dropped.
    """
    branches = list(mu.columns)
    bus_levels = sens.columns.get_level_values("bus").unique().tolist()
    n_br, n_bus = len(branches), len(bus_levels)
    bus_idx = {b: i for i, b in enumerate(bus_levels)}
    gen_bus_idx = np.array([bus_idx[b] for b in gen_buses])

    ids = mu.index.values.astype(np.int64)
    id_to_row = {int(i): r for r, i in enumerate(ids)}
    n = len(ids)

    mu_arr = mu.values                                              # (n, n_br)
    sens_arr = sens.values.reshape(n, n_br, n_bus)                  # (n, n_br, n_bus)
    x_bits = np.stack([id_to_avail(int(i), NGEN) for i in ids])     # (n, NGEN)

    # avail position k is the (NGEN-1-k)-th bit of the integer encoding
    bit_mask = (1 << (NGEN - 1 - np.arange(NGEN))).astype(np.int64)
    neighbor_ids = ids[:, None] ^ bit_mask[None, :]                 # (n, NGEN)

    valid = np.vectorize(lambda nid: int(nid) in id_to_row)(neighbor_ids)
    neighbor_rows = np.vectorize(lambda nid: id_to_row.get(int(nid), 0))(neighbor_ids)
    mu_y = mu_arr[neighbor_rows]                                    # (n, NGEN, n_br)
    dmu = mu_y - mu_arr[:, None, :]                                 # (n, NGEN, n_br)

    # sensitivity column for the bus hosting each flipped generator
    S_per_gen = sens_arr[:, :, gen_bus_idx]                         # (n, n_br, NGEN)
    S_per_gen = np.transpose(S_per_gen, (0, 2, 1))                  # (n, NGEN, n_br)
    flip_sign = (2 * x_bits - 1)[:, :, None]                        # (n, NGEN, 1)
    S_signed = flip_sign * S_per_gen                                # (n, NGEN, n_br)

    mask = np.broadcast_to(valid[:, :, None], dmu.shape)
    branch_arr = np.broadcast_to(np.array(branches), dmu.shape)
    return pd.DataFrame({
        "branch": branch_arr[mask],
        "S_signed": S_signed[mask],
        "dmu": dmu[mask],
    })


def plot_points_all(pts: pd.DataFrame, out_path: Path, eps=1e-6) -> None:
    """Scatter of delta_mu vs (flip-signed) sensitivity over every cached config."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    zero_dmu = pts["dmu"].abs().values <= eps # indices
    ax = axs[0]
    branches = sorted(pts["branch"].unique(), key=int)
    cmap = plt.get_cmap("tab20", len(branches))
    for i, br in enumerate(branches):
        m = pts["branch"] == br
        ax.scatter(pts["S_signed"][m&~zero_dmu], pts["dmu"][m&~zero_dmu],
                   s=5, alpha=0.15, color=cmap(i), label=f"{br}",
                   rasterized=True, linewidths=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"flip_sign $\cdot\; \partial \mu_l / \partial \mathrm{load}_{b(k)}$  at base $x$")
    ax.set_ylabel(r"$\mu_l(y_k) - \mu_l(x)$")
    ax.set_title(
        f"{sum(~zero_dmu):,} (config, gen, branch) points"
    )
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    fontsize=8, ncol=1, frameon=False, title="branch",
                    markerscale=4)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    ax = axs[1]
    ax.hist(pts["S_signed"], log=True, label="all")
    ax.hist(pts["S_signed"][zero_dmu], log=True, label=r"$\Delta\mu=0$")
    ax.set_xlabel("sensitivity")
    ax.set_title(r"$\Delta \mu = 0$, " + f"{sum(zero_dmu):.2g} pts")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")
    report_metrics(pts["S_signed"].values, pts["dmu"].values, "all configs")


def main():
    mu = pd.read_parquet(RESULTS_DIR / "dual_cache.parquet")["mu"]
    sens = pd.read_parquet(RESULTS_DIR / "sensitivity_cache.parquet")
    gen_buses = load_gen_bus_map(CASE_PATH)
    #pts = collect_points(BASE_AVAIL_ID, mu, sens, gen_buses)
    #out_path = RESULTS_DIR / f"sensitivity_vs_flip_avail{BASE_AVAIL_ID}.pdf"
    #plot_points(pts, BASE_AVAIL_ID, out_path)
    pts_all = collect_points_all(mu, sens, gen_buses)
    plot_points_all(pts_all, RESULTS_DIR / "sensitivity_vs_flip_all.pdf")
    plt.show()


if __name__ == "__main__":
    main()
