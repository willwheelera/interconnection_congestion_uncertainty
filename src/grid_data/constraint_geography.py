"""
Attach approximate geography to ERCOT tail-congestion constraints.

Each np6-86 shadow-price row names the monitored line's physical endpoints
(FromStation/ToStation) but carries no coordinates. ERCOT does not publish
substation coordinates, and the station codes are cryptic abbreviations that do
not fuzzy-match public datasets (HIFLD, EIA) reliably. Since the tail is
concentrated -- the top ~40 constraints cover 80% of tail events -- we place the
tail-dominant stations by hand instead.

STATION_GEO maps a station code to (ERCOT weather zone, approximate town lat,
lon). Coordinates are town centroids, accurate to region not to the substation.
Stations we cannot confidently identify are left out; their constraints fall into
the "Unmapped" bucket so coverage stays visible.

Most codes are placed by recognizing the town name in the abbreviation (BRUNI,
VICTORIA, ODEHV=Odessa). A few high-μ codes are opaque and were instead resolved
from evidence in the data:
  - contingency codes: the ContingencyName that triggers a constraint encodes the
    neighboring elements, which resolve to real stations elsewhere in the feed and
    pin the region. LOYOLA is triggered by SKINKLE8/SKLELOY8 -> the stations
    KINGSVIL + KLEBERG (Kingsville, Coastal Bend). SEA is triggered by
    DSALHUT/DSALTM/DFRYTM -> Salado + Temple + Hutto (Bell/Williamson County).
  - voltage + naming pattern: LPLMK/LPLNE sit on a self-contained 115 kV network
    (LPLDB/LPLMD/LPLMK/LPLNE/LPLNW/LPLPY). The LPL prefix, 115 kV class, and
    compass-point substations identify Lubbock Power & Light (South Plains, in
    ERCOT since 2021); filed under FarWest for lack of a South Plains zone.

Post-processing only: aggregation and geo lookup. Plotting lives in
plot_constraint_geography.py.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "ercot_data" / "shadow_prices_2025.parquet"
TAIL = 300.0  # $/MWh; a binding contingency interval above this is a tail event

# ERCOT weather zones, ordered roughly west -> east for plotting.
ZONE_ORDER = ["FarWest", "West", "North", "NorthCentral",
              "SouthCentral", "Southern", "Coast", "East", "Unmapped"]

# station code -> (zone, lat, lon). Hand-curated, town-level. See module docstring.
STATION_GEO = {
    # --- Far West (Permian Basin) ---
    "ODEHV": ("FarWest", 31.85, -102.37), "ODESA": ("FarWest", 31.85, -102.37),
    "ODNTH": ("FarWest", 31.90, -102.35), "YARBR": ("FarWest", 31.80, -102.30),
    "BIGLAKE": ("FarWest", 31.19, -101.46), "WOLFCAMP": ("FarWest", 31.90, -102.10),
    "PALOUSE": ("FarWest", 31.90, -102.05), "ABNTHWST": ("FarWest", 33.83, -101.84),
    # --- West / West-Central ---
    "TWINBU": ("West", 31.35, -100.52), "HARGROVE": ("West", 31.40, -100.55),
    "VEALMOOR": ("West", 32.42, -101.62), "KOCHTAP": ("West", 32.35, -101.60),
    "SCRCV": ("West", 32.72, -100.92), "KNAPP": ("West", 32.73, -100.90),
    "STER": ("West", 31.83, -101.00), "STCO": ("West", 31.83, -101.00),
    "SONR": ("West", 30.57, -100.64), "ATSO": ("West", 30.60, -100.60),
    "OZONA": ("West", 30.71, -101.20), "MIDW": ("West", 30.80, -101.20),
    "CONCHO": ("West", 31.30, -99.90), "VRBS": ("West", 31.30, -99.90),
    "HEXT": ("West", 30.88, -99.62), "YELWJCKT": ("West", 31.00, -99.50),
    "FORTMA": ("West", 30.75, -99.23), "TREADWEL": ("West", 32.40, -99.70),
    "MRVLY": ("West", 32.40, -98.82), "ESTLD": ("West", 32.40, -98.82),
    "EDWTP": ("West", 30.30, -100.00),
    # --- North Central (DFW metro and south of it) ---
    "ARGYL": ("NorthCentral", 33.12, -97.19), "LWSVH": ("NorthCentral", 33.05, -97.00),
    "BELCNTY": ("NorthCentral", 31.05, -97.46), "SALSW": ("NorthCentral", 31.06, -97.38),
    "ITALY": ("NorthCentral", 32.18, -96.88), "MEXTP": ("NorthCentral", 32.20, -96.90),
    "NVARO": ("NorthCentral", 32.09, -96.47), "HAN1": ("NorthCentral", 32.10, -96.50),
    "HILGR": ("NorthCentral", 32.01, -97.13),
    # --- South Central (Austin / San Antonio metros) ---
    "RRWES": ("SouthCentral", 30.51, -97.68), "RNDRK": ("SouthCentral", 30.51, -97.68),
    "GEORSO": ("SouthCentral", 30.63, -97.68), "HUTTO": ("SouthCentral", 30.54, -97.55),
    "RRNES": ("SouthCentral", 30.55, -97.60), "ELMCREEK": ("SouthCentral", 29.20, -98.70),
    "SANA": ("SouthCentral", 29.42, -98.49),
    # --- Southern (deep South Texas: Eagle Ford, border, Rio Grande Valley) ---
    "LARDVNTH": ("Southern", 27.55, -99.50), "LASCRUCE": ("Southern", 27.60, -99.42),
    "LAREDO": ("Southern", 27.53, -99.49), "DEL_MAR": ("Southern", 27.60, -99.48),
    "MILO": ("Southern", 27.60, -99.30), "NLARSW": ("Southern", 27.62, -99.45),
    "BRUNI": ("Southern", 27.42, -98.85), "LA_PALMA": ("Southern", 26.15, -98.10),
    "HAINE_DR": ("Southern", 26.20, -98.15), "PEARSALL": ("Southern", 28.89, -99.10),
    "POTEETS": ("Southern", 29.04, -98.58), "POTEET": ("Southern", 29.04, -98.58),
    "OAKS9": ("Southern", 29.00, -98.50), "PILONCIL": ("Southern", 28.35, -99.61),
    "CATARINA": ("Southern", 28.35, -99.61), "DILLEYSW": ("Southern", 28.67, -99.17),
    "FREER": ("Southern", 27.88, -98.62), "LOBO": ("Southern", 27.90, -98.60),
    "KENEDSW": ("Southern", 28.82, -97.85), "GANSO": ("Southern", 28.70, -100.40),
    "MAVERICK": ("Southern", 28.70, -100.48), "BIG_FOOT": ("Southern", 28.98, -98.55),
    "UVALDE": ("Southern", 29.21, -99.79), "W_BATESV": ("Southern", 28.95, -99.62),
    # --- Coast ---
    "VICTORIA": ("Coast", 28.80, -97.00), "MAGRUDER": ("Coast", 28.80, -96.90),
    "STP": ("Coast", 28.80, -96.05), "BLESSING": ("Coast", 28.87, -96.21),
    "NAVALBAS": ("Coast", 27.69, -97.29), "N_PADRE": ("Coast", 27.60, -97.22),
    "NUECES_B": ("Coast", 27.85, -97.50), "MORRIS": ("Coast", 27.80, -97.55),
    # --- second pass: confidently-zoned stations from the μ-ranked unmapped tail ---
    "SNYDR": ("West", 32.72, -100.92), "CLCTY": ("West", 32.40, -100.86),
    "BANDER": ("SouthCentral", 29.72, -99.07), "HENLY": ("SouthCentral", 30.21, -98.20),
    "LYTTON_S": ("SouthCentral", 29.88, -97.60), "OLSEN": ("North", 35.20, -101.83),
    "CEDRHILL": ("NorthCentral", 32.59, -96.96), "KFMNW": ("NorthCentral", 32.59, -96.31),
    "DILLY": ("Southern", 28.67, -99.17), "PEARSAL1": ("Southern", 28.89, -99.10),
    # --- third pass: cryptic codes resolved from evidence (see docstring) ---
    "LOYOLA": ("Coast", 27.52, -97.86), "SEA": ("NorthCentral", 31.10, -97.38),
    "LPLMK": ("FarWest", 33.58, -101.85), "LPLNE": ("FarWest", 33.58, -101.85),
}


def load_tail_constraints(path: Path = DATA_PATH, tail: float = TAIL) -> pd.DataFrame:
    """Per-constraint tail profile from the base scan (contingency rows, μ>tail)."""
    df = pd.read_parquet(path)
    b = df[(df["ContingencyName"].str.strip() != "BASE CASE") & (df["ShadowPrice"] > tail)]
    prof = b.groupby("ConstraintName").agg(
        tail_events=("ShadowPrice", "size"),
        total_mu=("ShadowPrice", "sum"),
        max_mu=("ShadowPrice", "max"),
        from_st=("FromStation", "first"),
        to_st=("ToStation", "first"),
        kv=("FromStationkV", "first"),
    )
    return prof.sort_values("tail_events", ascending=False)


def _lookup(from_st: str, to_st: str) -> tuple:
    """Geolocate a constraint by whichever endpoint is in the crosswalk."""
    for code in (from_st, to_st):
        if code in STATION_GEO:
            return STATION_GEO[code]
    return ("Unmapped", None, None)


def attach_geography(prof: pd.DataFrame) -> pd.DataFrame:
    """Add zone, lat, lon columns by looking up each constraint's endpoints."""
    geo = prof.apply(lambda r: _lookup(r["from_st"], r["to_st"]), axis=1)
    prof = prof.copy()
    prof[["zone", "lat", "lon"]] = pd.DataFrame(geo.tolist(), index=prof.index)
    return prof
