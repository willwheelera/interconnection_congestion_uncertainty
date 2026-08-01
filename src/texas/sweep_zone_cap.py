"""Scan ZONE_SCALE_CAP to find where the 2025 load pattern starts overloading the network.

build_load_snapshots.py clamps every per-zone load scale at ZONE_SCALE_CAP = 1.5. In
practice only FarWest hits it (true 2025 ratios are 4.9-6.3x the model's 1,307 MW
nominal), and the clamp was justified against a hard-limit model where an uncapped
FarWest shed load and pinned the target at the value of lost load. The current recipe
prices overload instead of forbidding it (soft_limits), so that justification no longer
applies and the onset of overload has to be measured rather than assumed.

For each cap rung this rebuilds the per-zone scales from the raw ERCOT snapshot loads,
solves the same SCDCOPF recipe the sweep uses, and records per snapshot:

  - worst thermal overload the price cap bought, and load shed (nodal relief + system),
  - S = max load-averaged mu and its driving constraint, and whether S pins at the cap,
  - FarWest net export MW (generation + local shed - load; equals net tie flow on a
    lossless DC network) and the largest mu on any FarWest-touching constraint.

FarWest is a large net exporter in the 2025 fleet, so raising its load absorbs local
generation and *reduces* net export. The export columns are here to test whether a
higher cap relieves the FarWest corridors rather than stressing them.

Relief is placed at every load bus (not the slimmed sweep set), so shed at a high cap is
priced on the ORDC curve at the bus that needs it instead of falling through to the
system-wide slack and misreporting where the stress is.

Usage:
    uv run python src/texas/sweep_zone_cap.py
    uv run python src/texas/sweep_zone_cap.py --caps 1.5 3 5 --configs off
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
from egret.data.model_data import ModelData
from egret.models.scopf import create_scopf_model

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "texas"))

from sample_harness import (CASE, QUEUE, RESULTS_NAME, EMERGENCY_FACTOR, ORDC_C_MIN,
                            ORDC_C_MAX, ORDC_BLOCKS, SHADOW_PRICE_CAP, SOLVER_THREADS,
                            extract_mu, set_emergency_factor)
from load_averaged_target import (add_queue_generators, add_lostload_generators_ordc,
                                  scale_loads_by_zone, scale_renewables_by_snapshot)
from soft_limits import enable_soft_limits, thermal_violations
from build_load_snapshots import ZONES
from egret.models.dcopf import _lazy_ptdf_dcopf_model_solve_loop
from egret.common.solver_interface import _solve_model

SNAPSHOTS = REPO_ROOT / "data" / "activsg2000_load_snapshots.csv"
COLLECTION = REPO_ROOT / "data" / "activsg2000_load_collection.json"
OUT = REPO_ROOT / "results" / RESULTS_NAME / "zone_cap_ladder.parquet"

# Rungs to scan. None means uncapped: every zone runs at its true 2025 ERCOT ratio.
DEFAULT_CAPS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, None]
ZONE = "FarWest"


def zone_scales_at_cap(nominal, row, cap):
    """Per-zone scale factors for one snapshot, clamped at cap (None = uncapped)."""
    scales = {z: float(row[z]) / nominal[z] for z in ZONES}
    if cap is None:
        return scales
    return {z: min(s, cap) for z, s in scales.items()}


def zone_buses(md, zone):
    """Bus ids in the named weather zone."""
    return {b for b, rec in md.data["elements"]["bus"].items() if rec["zone"] == zone}


def zone_branch_constraints(md, buses):
    """Branch names with at least one endpoint in the bus set."""
    return {bn for bn, b in md.data["elements"]["branch"].items()
            if b["from_bus"] in buses or b["to_bus"] in buses}


def build_persistent(snapshot, scales, candidates):
    """Build one snapshot's SCDCOPF at the given zone scales, with all candidates on.

    Returns the model, the per-unit ModelData the lazy-PTDF loop needs, the persistent
    solver, per-candidate online bounds, and a `zone` dict holding the FarWest quantities
    read off the MW-unit ModelData before it is scaled: the generator names sitting in the
    zone, the branch names touching it, and its scaled load.
    """
    cf = snapshot.get("renewable_cf") or {}
    md = ModelData.read(str(CASE))
    set_emergency_factor(md, EMERGENCY_FACTOR)
    add_queue_generators(md, candidates, [1] * len(candidates))
    scale_loads_by_zone(md, scales)
    scale_renewables_by_snapshot(md, cf)
    add_lostload_generators_ordc(md, ORDC_C_MIN, ORDC_C_MAX, ORDC_BLOCKS)
    buses = zone_buses(md, ZONE)
    zone = dict(
        gens={n for n, g in md.data["elements"]["generator"].items() if g["bus"] in buses},
        lines=zone_branch_constraints(md, buses),
        load_mw=sum(l["p_load"] for l in md.data["elements"]["load"].values()
                    if l["bus"] in buses))
    m, md_scaled = create_scopf_model(md, include_feasibility_slack=True)
    base_mva = md.data["system"]["baseMVA"]
    enable_soft_limits(m, SHADOW_PRICE_CAP, base_mva)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solver = pyo.SolverFactory("gurobi_persistent")
    solver.options["OutputFlag"] = 0
    if SOLVER_THREADS:
        solver.options["Threads"] = SOLVER_THREADS
    solver.set_instance(m)
    cap_pu = {f"queue_{c['inr']}": cf.get(c["fuel"], 1.0) * c["capacity_mw"] / base_mva
              for c in candidates}
    return m, md_scaled, solver, cap_pu, base_mva, zone


def zone_balance(m, zone, base_mva):
    """(generation, local shed) MW inside the zone at the current solution.

    Iterates m.pg rather than the ModelData generators: EGRET omits out-of-service and
    zero-capacity units from the model, so not every named generator has a pg variable.
    """
    gen = shed = 0.0
    for name in m.pg:
        if name not in zone["gens"]:
            continue
        mw = (m.pg[name].value or 0.0) * base_mva
        if name.startswith("lostload_"):
            shed += mw
        else:
            gen += mw
    return gen, shed


def solve_config(m, md, solver, config, cap_pu, base_mva):
    """Toggle candidate bounds, warm re-solve, return (mu, overload, system shed)."""
    for on, name in zip(config, cap_pu):
        m.pg[name].setub(cap_pu[name] if on else 0.0)
        solver.update_var(m.pg[name])
    _solve_model(m, solver, solver_tee=False, return_solver=True, set_instance=False)
    _lazy_ptdf_dcopf_model_solve_loop(m, md, solver, solver_tee=False)
    sys_shed = float(pyo.value(m.p_load_shed)) * base_mva
    return extract_mu(m, base_mva), thermal_violations(m, base_mva), sys_shed


def worker(snapshot, caps, configs, candidates, nominal, row, result_q):
    """Solve every (cap, config) pair for one pinned snapshot."""
    import logging
    logging.getLogger("egret").setLevel(logging.WARNING)
    for cap in caps:
        scales = zone_scales_at_cap(nominal, row, cap)
        m, md, solver, cap_pu, base_mva, zone = build_persistent(snapshot, scales,
                                                                 candidates)
        for label, config in configs.items():
            mus, overload, sys_shed = solve_config(m, md, solver, config, cap_pu, base_mva)
            gen, shed = zone_balance(m, zone, base_mva)
            zone_mu = {k: v for k, v in mus.items()
                       if k.rsplit(":", 1)[-1] in zone["lines"]}
            ov_line, ov_mw = max(overload.items(), key=lambda kv: kv[1],
                                 default=("none", 0.0))
            result_q.put(dict(
                cap=cap, config=label, snapshot=snapshot["timestamp"],
                zone_scale=scales[ZONE], zone_load_mw=zone["load_mw"], zone_gen_mw=gen,
                zone_shed_mw=shed, zone_net_export_mw=gen + shed - zone["load_mw"],
                zone_max_mu=max(zone_mu.values(), default=0.0),
                n_zone_binding=len(zone_mu),
                max_overload_mw=ov_mw, overload_line=ov_line,
                total_shed_mw=shed_total(m, base_mva) + sys_shed,
                sys_shed_mw=sys_shed, mus=mus))


def shed_total(m, base_mva):
    """System-wide nodal relief dispatch (MW)."""
    return sum((m.pg[n].value or 0.0) * base_mva
               for n in m.pg if n.startswith("lostload_"))


def aggregate(records, n_snapshots):
    """Load-average mu per (cap, config) and attach the per-rung snapshot summaries."""
    totals = defaultdict(lambda: defaultdict(float))
    for r in records:
        for k, v in r["mus"].items():
            totals[(r["cap"], r["config"])][k] += v / n_snapshots
    rows = []
    for key, avg in totals.items():
        cap, label = key
        rung = [r for r in records if r["cap"] == cap and r["config"] == label]
        driver, S = max(avg.items(), key=lambda kv: kv[1], default=("none", 0.0))
        rows.append(dict(
            cap=cap, config=label, S=S, driver=driver,
            pinned_at_cap=sum(1 for r in rung
                              if max(r["mus"].values(), default=0.0) >= SHADOW_PRICE_CAP - 1e-6),
            zone_scale=max(r["zone_scale"] for r in rung),
            zone_load_mw=sum(r["zone_load_mw"] for r in rung) / len(rung),
            zone_gen_mw=sum(r["zone_gen_mw"] for r in rung) / len(rung),
            zone_shed_mw=sum(r["zone_shed_mw"] for r in rung) / len(rung),
            zone_net_export_mw=sum(r["zone_net_export_mw"] for r in rung) / len(rung),
            zone_min_export_mw=min(r["zone_net_export_mw"] for r in rung),
            zone_max_mu=max(r["zone_max_mu"] for r in rung),
            max_overload_mw=max(r["max_overload_mw"] for r in rung),
            overload_line=max(rung, key=lambda r: r["max_overload_mw"])["overload_line"],
            total_shed_mw=max(r["total_shed_mw"] for r in rung),
            n_snapshots_overloaded=sum(1 for r in rung if r["max_overload_mw"] > 1e-3),
            n_snapshots_shedding=sum(1 for r in rung if r["total_shed_mw"] > 1e-3)))
    return pd.DataFrame(rows).sort_values(["config", "cap"])


def report(df):
    """Print the ladder: overload/shed onset beside FarWest export and mu."""
    for label, sub in df.groupby("config"):
        print(f"\nconfig = {label}")
        print(f"  {'cap':>6} {'FW x':>5} {'FW load':>8} {'FW export':>10} {'FW min exp':>11} "
              f"{'FW max mu':>10} {'overload':>9} {'n_ov':>5} {'shed':>8} {'n_shed':>7} "
              f"{'S':>8}  driver")
        for _, r in sub.iterrows():
            cap = "none" if pd.isna(r["cap"]) else f"{r['cap']:.1f}"
            print(f"  {cap:>6} {r['zone_scale']:5.2f} {r['zone_load_mw']:8.0f} "
                  f"{r['zone_net_export_mw']:10.0f} {r['zone_min_export_mw']:11.0f} "
                  f"{r['zone_max_mu']:10.2f} {r['max_overload_mw']:9.1f} "
                  f"{int(r['n_snapshots_overloaded']):5d} {r['total_shed_mw']:8.1f} "
                  f"{int(r['n_snapshots_shedding']):7d} {r['S']:8.1f}  {r['driver']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", nargs="*", type=float, default=None,
                        help="cap rungs to scan; omit for the default ladder")
    parser.add_argument("--configs", nargs="*", default=["off", "on"],
                        choices=["off", "on"], help="queue configs to solve at each rung")
    args = parser.parse_args()

    caps = DEFAULT_CAPS if args.caps is None else list(args.caps)
    candidates = json.loads(QUEUE.read_text())["candidates"]
    collection = json.loads(COLLECTION.read_text())
    nominal, snaps = collection["nominal_zone_mw"], collection["snapshots"]
    raw = pd.read_csv(SNAPSHOTS).set_index("timestamp")
    configs = {"off": [0] * len(candidates), "on": [1] * len(candidates)}
    configs = {k: configs[k] for k in args.configs}

    print(f"case {CASE.name}: {len(caps)} caps x {len(snaps)} snapshots x "
          f"{len(configs)} configs", flush=True)
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(s, caps, configs, candidates, nominal,
                                              raw.loc[s["timestamp"]], result_q))
             for s in snaps]
    t0 = time.time()
    for p in procs:
        p.start()
    records = [result_q.get() for _ in range(len(caps) * len(snaps) * len(configs))]
    for p in procs:
        p.join()

    df = aggregate(records, len(snaps))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    report(df)
    print(f"\n{len(records)} solves in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
