"""Renter profiles (ACS PUMS) and listing catalogs (FirstMover).

Rules follow CS2880/reports/nyc_empirical_design.md:
  renters    TYPEHUGQ=1, NP>0, TEN=3; searcher frame = recent movers (MV=1), weighted by WGTP
  income     HINCP * ADJINC / 1e6 (2024 $), moved to the market month with CPI-U
  feasible   rent <= share * income / 12, bedrooms >= min(8, max(0, ceil(NP/2) - 1)), same borough, same month
  groups     lower income = at or below the NYC renter median; G+ also has >= 1 feasible listing in its cell
Race and ethnicity are kept for descriptive breakdowns only. They never enter feasibility or preferences.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUMS_HOUSEHOLDS = ROOT / "census_pums" / "acs_pums_2020_2024_nyc_households.csv"
CPI_CSV = ROOT / "data" / "reference" / "cpi_u_us_city_average.csv"
FIRSTMOVER_CLEAN = ROOT / "data" / "firstmover" / "clean"
DERIVED = ROOT / "data" / "derived"
RENTER_CACHE = DERIVED / "renter_profiles.csv.gz"

MONTHS = ("2026-06", "2026-07", "2026-08")
MAX_NEED_BEDROOMS = 8
PUMS_COLUMNS = ["SERIALNO", "PUMA", "NYC_BOROUGH", "WGTP", "NP", "TYPEHUGQ", "TEN", "HINCP", "ADJINC", "MV", "HUPAC",
                "HHT", "HHLDRAGEP", "FS", "VEH", "HHLDRRAC1P", "HHLDRHISP"]
LISTING_COLUMNS = ["listing_uid", "unit_id", "building_id", "rent", "bedrooms", "bathrooms_total", "sqft", "neighborhood",
                   "latitude", "longitude", "puma", "borough", "source_type", "source_group", "firm_id", "building_type",
                   "in_analysis_catalog"]


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    x, cw = np.asarray(x)[order], np.cumsum(np.asarray(w)[order]) / np.sum(w)
    return float(x[min(np.searchsorted(cw, q), len(x) - 1)])


def bedroom_need(np_people) -> np.ndarray:
    return np.minimum(np.maximum(0, np.ceil(np.asarray(np_people) / 2) - 1), MAX_NEED_BEDROOMS).astype(int)


def cpi_factor(month: str) -> float:
    """Multiply 2024 dollars by this to get dollars of `month` (YYYY-MM)."""
    t = pd.read_csv(CPI_CSV, dtype={"period": str})
    base = t.loc[(t.year == 2024) & (t.period == "M13"), "value"].iloc[0]
    now = t.loc[(t.year == int(month[:4])) & (t.period == f"M{int(month[5:]):02d}"), "value"].iloc[0]
    return float(now) / float(base)


def build_renter_profiles() -> pd.DataFrame:
    """All PUMS renter households, one row per household, with income groups computed over all renters."""
    h = pd.read_csv(PUMS_HOUSEHOLDS, usecols=PUMS_COLUMNS, dtype=str)
    for c in ["WGTP", "NP", "HINCP", "ADJINC", "HHLDRAGEP", "VEH"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    r = h[(h.TYPEHUGQ == "1") & (h.NP > 0) & (h.TEN == "3")].copy()
    income = r.HINCP * r.ADJINC / 1e6
    weight = r.WGTP.astype(float)
    cuts = [weighted_quantile(income.to_numpy(), weight.to_numpy(), q) for q in (0.25, 0.5, 0.75)]
    out = pd.DataFrame({
        "serialno": r.SERIALNO,
        "survey_year": r.SERIALNO.str[:4].astype(int),
        "puma": r.PUMA,
        "borough": r.NYC_BOROUGH,
        "weight": weight,
        "household_size": r.NP.astype(int),
        "income_2024": income,
        "need_bedrooms": bedroom_need(r.NP),
        "recent_mover": r.MV == "1",
        "has_children": r.HUPAC.isin(["1", "2", "3"]),
        "household_type": r.HHT,
        "householder_age": r.HHLDRAGEP,
        "snap": r.FS == "1",
        "vehicles": r.VEH,
        "race_descriptive_only": r.HHLDRRAC1P,
        "hispanic_descriptive_only": r.HHLDRHISP,
    })
    out["income_quartile"] = pd.cut(out.income_2024, [-np.inf, *cuts, np.inf], labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    out["lower_income"] = out.income_2024 <= cuts[1]
    return out.reset_index(drop=True)


def load_renters(refresh: bool = False) -> pd.DataFrame:
    if RENTER_CACHE.exists() and not refresh:
        return pd.read_csv(RENTER_CACHE, dtype={"puma": str, "serialno": str, "household_type": str,
                                                "race_descriptive_only": str, "hispanic_descriptive_only": str})
    r = build_renter_profiles()
    DERIVED.mkdir(parents=True, exist_ok=True)
    r.to_csv(RENTER_CACHE, index=False)
    return r


def load_listings(month: str) -> pd.DataFrame:
    """The analysis catalog for one month: NYC units, plausible rent and baths, first posting per unit."""
    d = pd.read_csv(FIRSTMOVER_CLEAN / f"firstmover_listings_{month}.csv.gz", usecols=LISTING_COLUMNS, low_memory=False)
    d = d[d.in_analysis_catalog.astype(bool)].drop(columns="in_analysis_catalog").reset_index(drop=True)
    d["puma"] = d.puma.astype(float).astype(int).map(lambda p: f"{p:05d}")
    d["month"] = month
    return d


def feasible_counts(caps: np.ndarray, need: np.ndarray, rents: np.ndarray, bedrooms: np.ndarray) -> np.ndarray:
    """|E_b| per renter against one listing set: rent <= cap and bedrooms >= need."""
    out = np.zeros(len(caps), dtype=np.int64)
    for b in np.unique(need):
        sorted_rents = np.sort(rents[bedrooms >= b])
        sel = need == b
        out[sel] = np.searchsorted(sorted_rents, caps[sel], side="right")
    return out


def cell_frames(renters: pd.DataFrame, listings: pd.DataFrame, borough: str | None, month: str, share: float,
                recent_movers_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renter frame and listing pool of a cell, with cap_month, n_feasible_cell and group flags.

    borough=None is the citywide scope (sensitivity check): every renter can search every listing.
    """
    r = renters[renters.borough == borough] if borough else renters
    if recent_movers_only:
        r = r[r.recent_mover]
    r = r.copy()
    l = (listings[listings.borough == borough] if borough else listings).copy().reset_index(drop=True)
    r["income_month"] = r.income_2024 * cpi_factor(month)
    r["cap_month"] = np.maximum(share * r.income_month / 12, 0.0)
    r["n_feasible_cell"] = feasible_counts(r.cap_month.to_numpy(), r.need_bedrooms.to_numpy(), l.rent.to_numpy(float),
                                           l.bedrooms.to_numpy())
    r["group"] = np.select([r.lower_income & (r.n_feasible_cell >= 1), r.lower_income], ["G+", "G0"], "higher")
    return r.reset_index(drop=True), l
