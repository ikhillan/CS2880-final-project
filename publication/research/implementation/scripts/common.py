"""Shared paths and helpers for the COMPSCI 2880 data pipeline.

Conventions (match DATA_INVENTORY.md section 7):
- Renters are PUMS occupied rental housing units: TYPEHUGQ=1, NP>0, TEN=3, weighted by WGTP.
- Household income is put in 2024 dollars with ADJINC, then moved to the listing period with
  the US CPI-U (data/reference/cpi_u_us_city_average.csv).
- A listing is feasible for a household if rent <= share * monthly income and
  bedrooms >= max(0, ceil(NP/2) - 1)  (at most two people per room; need capped at 8).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
PUMS_DIR = ROOT / "census_pums"
RENTHOP_DIR = ROOT / "renthop_data"
PUMA_GEOJSON = DATA / "geo" / "nyc_puma2020.geojson"
CPI_CSV = DATA / "reference" / "cpi_u_us_city_average.csv"

BOROUGH_BY_PUMA_PREFIX = {"041": "Manhattan", "042": "Bronx", "043": "Brooklyn",
                          "044": "Queens", "045": "Staten Island"}
BOROUGHS = list(BOROUGH_BY_PUMA_PREFIX.values())
PRICE_MIN, PRICE_MAX = 800, 20_000  # analysis filter used for both catalogs
MAX_NEED_BEDROOMS = 8


# ---------------------------------------------------------------- geography
def load_puma_polygons(path: Path = PUMA_GEOJSON) -> list[tuple[str, MplPath, list[MplPath]]]:
    gj = json.loads(Path(path).read_text())
    polys = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        parts = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for rings in parts:
            polys.append((feat["properties"]["PUMA"], MplPath(np.asarray(rings[0])),
                          [MplPath(np.asarray(h)) for h in rings[1:]]))
    return polys


def assign_puma(lon, lat, polys=None) -> np.ndarray:
    """2020 PUMA code for each (lon, lat) point; None when the point is outside NYC."""
    polys = polys or load_puma_polygons()
    pts = np.column_stack([np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)])
    out = np.full(len(pts), None, dtype=object)
    for puma, exterior, holes in polys:
        hit = exterior.contains_points(pts)
        for hole in holes:
            hit &= ~hole.contains_points(pts)
        out[hit & pd.isna(out)] = puma
    return out


def borough_of_puma(puma) -> str | None:
    return BOROUGH_BY_PUMA_PREFIX.get(puma[:3]) if isinstance(puma, str) else None


# ---------------------------------------------------------------- prices
def cpi(year: int, period: str = "M13") -> float:
    """US CPI-U (CUUR0000SA0). period 'M13' is the annual average, 'M01'..'M12' are months."""
    t = pd.read_csv(CPI_CSV, dtype={"period": str})
    v = t.loc[(t.year == year) & (t.period == period), "value"]
    if v.empty:
        raise KeyError(f"CPI-U {year} {period} not in {CPI_CSV}; run scripts/fetch_cpi.py")
    return float(v.iloc[0])


# ---------------------------------------------------------------- statistics
def weighted_quantiles(x: np.ndarray, w: np.ndarray, qs) -> list[float]:
    m = ~(np.isnan(x) | np.isnan(w))
    x, w = x[m], w[m]
    order = np.argsort(x)
    x, cw = x[order], np.cumsum(w[order]) / w.sum()
    return [float(x[min(np.searchsorted(cw, q), len(x) - 1)]) for q in qs]


def weighted_share(mask, w) -> float:
    return float(np.average(np.asarray(mask, dtype=float), weights=np.asarray(w, dtype=float)))


def md_table(rows: list[list], headers: list[str], align: str | None = None) -> str:
    """Markdown table; `align` is one char per column ('l' or 'r'), default left."""
    align = align or "l" * len(headers)
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------- renters
def load_pums_renters() -> pd.DataFrame:
    """PUMS renter households with income (2024$), bedroom need, weights and income quartile."""
    cols = ["SERIALNO", "PUMA", "WGTP", "NP", "TYPEHUGQ", "TEN", "HINCP", "ADJINC", "MV", "NYC_BOROUGH"]
    h = pd.read_csv(PUMS_DIR / "acs_pums_2020_2024_nyc_households.csv", usecols=cols, dtype=str)
    for c in ["WGTP", "NP", "HINCP", "ADJINC"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    r = h[(h.TYPEHUGQ == "1") & (h.NP > 0) & (h.TEN == "3")].copy()
    r["income_2024"] = r.HINCP * r.ADJINC / 1e6
    r["weight"] = r.WGTP.astype(float)
    r["need_bedrooms"] = np.minimum(np.maximum(0, np.ceil(r.NP / 2) - 1), MAX_NEED_BEDROOMS).astype(int)
    r["recent_mover"] = r.MV == "1"
    cuts = weighted_quantiles(r.income_2024.to_numpy(), r.weight.to_numpy(), [0.25, 0.5, 0.75])
    r["income_quartile"] = pd.cut(r.income_2024, [-np.inf, *cuts, np.inf], labels=["Q1", "Q2", "Q3", "Q4"])
    r = r.rename(columns={"NYC_BOROUGH": "borough"})
    r.attrs["quartile_cuts_2024"] = cuts
    return r.reset_index(drop=True)


def renter_groups(r: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for the groups reported in the feasibility comparison."""
    return {
        "All renters": pd.Series(True, index=r.index),
        "Bottom income quartile (Q1)": r.income_quartile == "Q1",
        "Second income quartile (Q2)": r.income_quartile == "Q2",
        "Top half (Q3-Q4)": r.income_quartile.isin(["Q3", "Q4"]),
        "Recent movers (MV=1)": r.recent_mover,
    }


# ---------------------------------------------------------------- feasibility
def feasible_counts(caps: np.ndarray, need: np.ndarray, rents: np.ndarray, bedrooms: np.ndarray,
                    renter_area: np.ndarray | None = None, listing_area: np.ndarray | None = None) -> np.ndarray:
    """Per renter: number of listings with rent <= cap and bedrooms >= need (optionally same area)."""
    caps, need = np.asarray(caps, dtype=float), np.asarray(need)
    rents, bedrooms = np.asarray(rents, dtype=float), np.asarray(bedrooms)
    out = np.zeros(len(caps), dtype=np.int64)
    areas = [None] if renter_area is None else pd.unique(renter_area)
    for area in areas:
        r_in = np.ones(len(caps), bool) if area is None else (renter_area == area)
        l_in = np.ones(len(rents), bool) if area is None else (listing_area == area)
        for b in np.unique(need[r_in]):
            sorted_rents = np.sort(rents[l_in & (bedrooms >= b)])
            sel = r_in & (need == b)
            out[sel] = np.searchsorted(sorted_rents, caps[sel], side="right")
    return out
