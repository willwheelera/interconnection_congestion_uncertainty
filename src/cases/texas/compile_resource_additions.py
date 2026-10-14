"""Compile ERCOT resources added 2017-2025 from the CDR unit-detail tables.

Source is the most recent Capacity, Demand and Reserves Report (Dec 2025), whose
`Unit Details` sheet lists every operational resource with its in-service date,
technology, fuel, county, ERCOT weather zone, and nameplate capacity. Filtering
that sheet to in-service years 2017-2025 yields the additions still on the grid --
the set relevant to current congestion. The `New CDR-Eligible Resources` sheet
gives the planned (not-yet-online) pipeline, written to a second file.

Note: units added and then retired before Dec 2025 are absent (they are gone from
the current fleet); for current-congestion modeling this is the desired behavior.

Outputs:
    data/ercot_resource_additions_2017_2025.csv  -- operational additions
    data/ercot_resource_planned.csv              -- planned pipeline

Usage:
    uv run python src/texas/compile_resource_additions.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import openpyxl

from src.paths import REPO_ROOT
CDR = REPO_ROOT / "data" / "ercot_cdr" / "CapacityDemandandReservesReport_December2025.xlsx"
OUT_ADDED = REPO_ROOT / "data" / "ercot_resource_additions_2017_2025.csv"
OUT_PLANNED = REPO_ROOT / "data" / "ercot_resource_planned.csv"

YEAR_LO, YEAR_HI = 2017, 2025

# Canonical field -> normalized header text (uppercase, single-spaced).
HEADERS = {
    "unit_name": "UNIT NAME",
    "inr": "INTERCONNECTION REQUEST NUMBER (INR)",
    "unit_code": "UNIT CODE",
    "technology": "TECHNOLOGY",
    "fuel": "FUEL",
    "status": "CDR STATUS",
    "attribute": "CDR RESOURCE ATTRIBUTE",
    "county": "COUNTY",
    "zone": "ZONE",
    "in_service": "IN-SERVICE DATE",
    "installed_mw": "INSTALLED CAPACITY (MW)",
}
CSV_FIELDS = ["unit_name", "unit_code", "inr", "technology", "fuel", "fuel_group",
              "status", "county", "zone", "in_service", "installed_mw"]


def fuel_group(fuel: str) -> str:
    """Collapse the detailed CDR fuel codes into headline resource groups."""
    if fuel.startswith("WIND"):
        return "Wind"
    if fuel.startswith("SOLAR"):
        return "Solar"
    if fuel in ("STORAGE", "MWH"):
        return "Storage"
    return {"GAS": "Gas", "COAL": "Coal", "NUCLEAR": "Nuclear", "HYDRO": "Hydro",
            "BIOMASS": "Biomass", "DIESEL": "Diesel"}.get(fuel, fuel.title())


def norm(cell) -> str:
    """Normalize a header cell to uppercase single-spaced text."""
    return " ".join(str(cell).split()).upper() if cell is not None else ""


def read_table(ws) -> list[dict]:
    """Read a CDR resource table into records, locating the header by 'UNIT NAME'.

    Handles both the `Unit Details` layout and the column-shifted
    `New CDR-Eligible Resources` layout. Rows without a numeric installed
    capacity (section headers, notes) are skipped.
    """
    header_row, col = None, {}
    for i, row in enumerate(ws.iter_rows(min_row=1, max_row=15, values_only=True)):
        cells = [norm(c) for c in row]
        if "UNIT NAME" in cells:
            header_row = i + 1
            col = {f: cells.index(h) for f, h in HEADERS.items() if h in cells}
            break
    records = []
    for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
        mw = row[col["installed_mw"]]
        if not row[col["unit_name"]] or not isinstance(mw, (int, float)):
            continue
        if norm(row[col["unit_name"]]) in ("TOTAL", "TOTALS", "GRAND TOTAL"):
            continue
        rec = {f: row[col[f]] for f in col}
        rec["installed_mw"] = round(float(mw), 2)
        rec["fuel_group"] = fuel_group(rec["fuel"] or "")
        isd = rec["in_service"]
        rec["in_service"] = isd.date().isoformat() if hasattr(isd, "date") else ""
        rec["_year"] = isd.year if hasattr(isd, "year") else None
        records.append(rec)
    return records


def write_csv(records: list[dict], path: Path) -> None:
    """Write records to CSV using the canonical column order."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def print_fuel_year_pivot(added: list[dict]) -> None:
    """Print installed MW by fuel group and in-service year."""
    years = list(range(YEAR_LO, YEAR_HI + 1))
    groups = sorted({r["fuel_group"] for r in added})
    mw = defaultdict(float)
    for r in added:
        mw[(r["fuel_group"], r["_year"])] += r["installed_mw"]
    print(f"\nInstalled MW by fuel group and in-service year:")
    print("  " + "fuel".ljust(9) + "".join(str(y).rjust(8) for y in years) + "   total")
    for g in groups:
        cells = "".join(f"{mw[(g, y)]:8.0f}" for y in years)
        print("  " + g.ljust(9) + cells + f"{sum(mw[(g, y)] for y in years):8.0f}")


def print_location_breakdown(added: list[dict]) -> None:
    """Print additions by ERCOT weather zone and the top counties by MW."""
    zone_mw, zone_n = defaultdict(float), defaultdict(int)
    county_mw = defaultdict(float)
    for r in added:
        zone_mw[r["zone"]] += r["installed_mw"]
        zone_n[r["zone"]] += 1
        county_mw[r["county"]] += r["installed_mw"]
    print("\nAdditions by ERCOT weather zone:")
    for z in sorted(zone_mw, key=lambda k: -zone_mw[k]):
        print(f"  {str(z).ljust(10)} {zone_n[z]:4d} units  {zone_mw[z]:9.0f} MW")
    print("\nTop 12 counties by added MW:")
    for c in sorted(county_mw, key=lambda k: -county_mw[k])[:12]:
        print(f"  {str(c).title().ljust(14)} {county_mw[c]:9.0f} MW")


def main() -> None:
    wb = openpyxl.load_workbook(CDR, read_only=True, data_only=True)
    fleet = read_table(wb["Unit Details"])
    planned = read_table(wb["New CDR-Eligible Resources"])
    wb.close()

    added = [r for r in fleet
             if str(r["status"]).startswith("OPER")
             and r["_year"] is not None and YEAR_LO <= r["_year"] <= YEAR_HI]
    added.sort(key=lambda r: (r["_year"], -r["installed_mw"]))

    write_csv(added, OUT_ADDED)
    write_csv(planned, OUT_PLANNED)

    total = sum(r["installed_mw"] for r in added)
    print(f"Source: {CDR.name}")
    print(f"Operational resources added {YEAR_LO}-{YEAR_HI}: {len(added)} units, "
          f"{total:,.0f} MW installed")
    print_fuel_year_pivot(added)
    print_location_breakdown(added)
    plan_mw = sum(r["installed_mw"] for r in planned)
    print(f"\nPlanned pipeline (New CDR-Eligible): {len(planned)} units, {plan_mw:,.0f} MW")
    print(f"\nWrote {OUT_ADDED.relative_to(REPO_ROOT)} and "
          f"{OUT_PLANNED.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
