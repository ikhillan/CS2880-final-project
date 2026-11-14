"""Renter profiles (ACS PUMS) and listing catalogs (FirstMover), following Todd's protocol (FINAL_EXPERIMENT_PROTOCOL_2.md).

  renters     TYPEHUGQ=1, NP>0, TEN=3; searcher frame = recent movers (MV=1), weighted by WGTP
  donors      a fixed hash of SERIALNO (which starts with the survey year) splits households 70/30 into calibration and
              evaluation donors
  dollars     July 2026 throughout. Incomes: HINCP x ADJINC / 1e6 gives 2024 dollars, then CPI-U to July 2026. Listing rents:
              CPI-U from the listing month to July 2026. Months without a published CPI value are handled by `cpi_value`.
  feasible    rent <= share x income / 12, enough bedrooms, same borough, same month
  groups      lower income = at or below the household-weighted median income of all NYC cash-renting households;
              G+ = lower income with >= 1 feasible listing in its cell, G0 = lower income with none
Race and ethnicity are kept for descriptive breakdowns only. They never enter feasibility, preferences or prompts.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUMS_HOUSEHOLDS = ROOT / "census_pums" / "acs_pums_2020_2024_nyc_households.csv"
CPI_CSV = ROOT / "data" / "reference" / "cpi_u_us_city_average.csv"
FIRSTMOVER_CLEAN = ROOT / "data" / "firstmover" / "clean"
DERIVED = ROOT / "data" / "derived"
RENTER_CACHE = DERIVED / "renter_profiles_v2.csv.gz"

MONTHS = tuple(f"{y}-{m:02d}" for y in (2025, 2026) for m in range(1, 13) if "2025-02" <= f"{y}-{m:02d}" <= "2026-10")
DEV_MONTHS = ("2026-06", "2026-07", "2026-08")
HELDOUT_PRIMARY = ("2026-09", "2026-10")
HELDOUT_REPLICATION = ("2025-06", "2025-07", "2025-08")
BASE_MONTH = "2026-07"  # every dollar amount is expressed in July 2026 dollars
BOROUGHS = ("Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island")
CALIBRATION_SHARE = 0.70
MAX_NEED_BEDROOMS = 8
PUMS_COLUMNS = ["SERIALNO", "PUMA", "NYC_BOROUGH", "WGTP", "NP", "TYPEHUGQ", "TEN", "HINCP", "ADJINC", "ADJHSG", "RNTP", "MV",
                "HUPAC", "HHT", "HHLDRAGEP", "FS", "VEH", "HHLDRRAC1P", "HHLDRHISP"]
LISTING_COLUMNS = ["listing_uid", "unit_id", "building_id", "created_at_utc", "rent", "net_effective_rent", "bedrooms",
                   "bathrooms_total", "sqft", "neighborhood", "latitude", "longitude", "puma", "borough", "source_type",
                   "source_group", "firm_id", "building_type", "in_analysis_catalog"]


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    x, cw = np.asarray(x)[order], np.cumsum(np.asarray(w)[order]) / np.sum(w)
    return float(x[min(np.searchsorted(cw, q), len(x) - 1)])


def bedroom_need(np_people, rule: str = "standard") -> np.ndarray:
    """standard: at most two people per bedroom, a studio for 1-2 people. strict: ceil(people/2) bedrooms for 3 or more."""
    n = np.asarray(np_people)
    need = np.maximum(0, np.ceil(n / 2) - 1)
    if rule == "strict":
        need = np.where(n >= 3, np.ceil(n / 2), need)
    elif rule != "standard":
        raise ValueError(f"unknown bedroom rule {rule}")
    return np.minimum(need, MAX_NEED_BEDROOMS).astype(int)


# ------------------------------------------------------------------------------------------------- prices
@lru_cache(maxsize=1)
def _cpi_monthly() -> pd.Series:
    t = pd.read_csv(CPI_CSV, dtype={"period": str})
    t = t[t.period.str.match(r"M(0[1-9]|1[0-2])$")]
    return pd.Series(t.value.to_numpy(float), index=[f"{y}-{p[1:]}" for y, p in zip(t.year, t.period)]).sort_index()


@lru_cache(maxsize=None)
def cpi_value(month: str) -> tuple[float, str]:
    """CPI-U for a month and how it was obtained.

    observed        published value
    interpolated    missing between two published months (October 2025, lost to the 2025 funding lapse): linear interpolation
    carried_forward after the last observed month: the last published value
    """
    s = _cpi_monthly()
    if month in s.index and np.isfinite(s[month]):
        return float(s[month]), "observed"
    observed = s.dropna()
    before, after = observed[observed.index < month], observed[observed.index > month]
    if len(before) and len(after):
        return float((before.iloc[-1] + after.iloc[0]) / 2), "interpolated"
    if len(before):
        return float(before.iloc[-1]), "carried_forward"
    raise KeyError(f"no CPI-U value at or before {month} in {CPI_CSV}")


@lru_cache(maxsize=1)
def cpi_2024_annual() -> float:
    t = pd.read_csv(CPI_CSV, dtype={"period": str})
    return float(t.loc[(t.year == 2024) & (t.period == "M13"), "value"].iloc[0])


def cpi_factor(month: str) -> float:
    """Multiply 2024 dollars by this to get dollars of `month`. Kept for the jobs code, which mirrors it."""
    return cpi_value(month)[0] / cpi_2024_annual()


def to_base_factor(month: str) -> float:
    """Multiply dollars of `month` by this to get July 2026 dollars."""
    return cpi_value(BASE_MONTH)[0] / cpi_value(month)[0]


# ------------------------------------------------------------------------------------------------- renters
def donor_split(serialno: pd.Series) -> pd.Series:
    """'calibration' (70%) or 'evaluation' (30%), by a fixed hash of SERIALNO, which encodes the survey year and household."""
    u = serialno.map(lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:12], 16) / 16**12)
    return pd.Series(np.where(u < CALIBRATION_SHARE, "calibration", "evaluation"), index=serialno.index)


def build_renter_profiles() -> pd.DataFrame:
    """All PUMS renter households, one row per household, with income groups computed over all renters."""
    h = pd.read_csv(PUMS_HOUSEHOLDS, usecols=PUMS_COLUMNS, dtype=str)
    for c in ["WGTP", "NP", "HINCP", "ADJINC", "ADJHSG", "RNTP", "HHLDRAGEP", "VEH"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    r = h[(h.TYPEHUGQ == "1") & (h.NP > 0) & (h.TEN == "3")].copy()
    income_2024 = r.HINCP * r.ADJINC / 1e6
    weight = r.WGTP.astype(float)
    median_2024 = weighted_quantile(income_2024.to_numpy(), weight.to_numpy(), 0.5)
    cuts = [weighted_quantile(income_2024.to_numpy(), weight.to_numpy(), q) for q in (0.25, 0.5, 0.75)]
    contract_rent_2024 = r.RNTP * r.ADJHSG / 1e6  # monthly contract rent, 2024 dollars
    out = pd.DataFrame({
        "serialno": r.SERIALNO,
        "survey_year": r.SERIALNO.str[:4].astype(int),
        "puma": r.PUMA,
        "borough": r.NYC_BOROUGH,
        "weight": weight,
        "household_size": r.NP.astype(int),
        "income_2024": income_2024,
        "contract_rent_2024": contract_rent_2024,
        "recent_mover": r.MV == "1",
        "has_children": r.HUPAC.isin(["1", "2", "3"]),
        "household_type": r.HHT,
        "householder_age": r.HHLDRAGEP,
        "snap": r.FS == "1",
        "vehicles": r.VEH,
        "race_descriptive_only": r.HHLDRRAC1P,
        "hispanic_descriptive_only": r.HHLDRHISP,
    })
    out["donors"] = donor_split(out.serialno)
    out["income_quartile"] = pd.cut(out.income_2024, [-np.inf, *cuts, np.inf], labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    out["lower_income"] = out.income_2024 <= median_2024
    with np.errstate(divide="ignore", invalid="ignore"):
        out["acs_contract_burden"] = np.where(out.income_2024 > 0, 12 * out.contract_rent_2024 / out.income_2024, np.nan)
    return out.reset_index(drop=True)


def load_renters(refresh: bool = False) -> pd.DataFrame:
    """Renter profiles plus income in July 2026 dollars (`income`)."""
    if RENTER_CACHE.exists() and not refresh:
        r = pd.read_csv(RENTER_CACHE, dtype={"puma": str, "serialno": str, "household_type": str,
                                              "race_descriptive_only": str, "hispanic_descriptive_only": str})
    else:
        r = build_renter_profiles()
        DERIVED.mkdir(parents=True, exist_ok=True)
        r.to_csv(RENTER_CACHE, index=False)
    r["income"] = r.income_2024 * cpi_value(BASE_MONTH)[0] / cpi_2024_annual()
    return r


def lower_income_cutoff(renters: pd.DataFrame) -> float:
    """The household-weighted median income of all NYC cash-renting households, July 2026 dollars."""
    return weighted_quantile(renters.income.to_numpy(), renters.weight.to_numpy(), 0.5)


def borough_weights(renters: pd.DataFrame) -> dict[str, float]:
    """Weighted recent-mover shares by borough: the cell weights for standardized group rates."""
    m = renters[renters.recent_mover]
    return (m.groupby("borough").weight.sum() / m.weight.sum()).to_dict()


# ------------------------------------------------------------------------------------------------- listings
def load_listings(month: str) -> pd.DataFrame:
    """The analysis catalog for one month.

    `rent` is in July 2026 dollars; `rent_asking` is the posted price in that month's dollars (shown to recommenders).
    `gap_day` marks postings on low-volume days flagged by scripts/firstmover_coverage.py.
    """
    d = pd.read_csv(FIRSTMOVER_CLEAN / f"firstmover_listings_{month}.csv.gz", usecols=LISTING_COLUMNS, low_memory=False)
    d = d[d.in_analysis_catalog.astype(bool)].drop(columns="in_analysis_catalog").reset_index(drop=True)
    d["puma"] = d.puma.astype(float).astype(int).map(lambda p: f"{p:05d}")
    d["month"] = month
    d["rent_asking"] = d.rent.astype(float)
    d["rent"] = d.rent_asking * to_base_factor(month)
    d["net_effective_rent_asking"] = d.net_effective_rent.astype(float)
    d["bedbin"] = d.bedrooms.clip(upper=3).astype(int)
    d["sqft"] = d.sqft.where(d.sqft > 0)
    d["gap_day"] = d.created_at_utc.str[:10].isin(gap_days().get(month, set()))
    return d


# ------------------------------------------------------------------------------------------------- held-out unit dedup
HELDOUT_PERIODS = {"primary": HELDOUT_PRIMARY, "replication": HELDOUT_REPLICATION}
IDENTIFIABLE_UNIT_SOURCES = ("url_unit",)  # unit_key from the StreetEasy URL's building/unit path (scripts/firstmover_clean.py)
DEDUP_COUNTS_CSV = ROOT / "data" / "firstmover" / "heldout_dedup_counts.csv"


def heldout_period(month: str) -> str | None:
    return next((name for name, months in HELDOUT_PERIODS.items() if month in months), None)


@lru_cache(maxsize=None)
def unit_key_sources(month: str) -> pd.Series:
    """listing_uid -> unit_key_source (how the cleaner identified the physical unit) for one month."""
    d = pd.read_csv(FIRSTMOVER_CLEAN / f"firstmover_listings_{month}.csv.gz", usecols=["listing_uid", "unit_key_source"])
    return d.set_index("listing_uid").unit_key_source


@lru_cache(maxsize=1)
def development_unit_ids() -> frozenset:
    """Every physical unit in the development months' analysis catalogs."""
    return frozenset(pd.concat([load_listings(m).unit_id for m in DEV_MONTHS]))


def dedup_heldout(listings: pd.DataFrame, month: str, dev_units, earlier_units) -> tuple[pd.DataFrame, dict]:
    """Protocol section 3: drop identifiable units seen in development, and units already seen earlier in the same held-out
    period (keeping each unit's earliest arrival; within a month the cleaner already keeps the first posting).

    Identifiable = unit_key_source in IDENTIFIABLE_UNIT_SOURCES. Ambiguous units are kept as distinct listings with flags
    for a sensitivity check: `unit_ambiguous`, plus `ambiguous_dev_match` / `ambiguous_repeat` when their (weaker) key
    matches a development unit or an earlier held-out arrival.
    """
    source = listings.listing_uid.map(unit_key_sources(month))
    identifiable = source.isin(IDENTIFIABLE_UNIT_SOURCES).to_numpy()
    in_dev = listings.unit_id.isin(dev_units).to_numpy()
    earlier = listings.unit_id.isin(earlier_units).to_numpy()
    drop = identifiable & (in_dev | earlier)
    out = listings.assign(unit_key_source=source.to_numpy(), unit_ambiguous=~identifiable,
                          ambiguous_dev_match=~identifiable & in_dev, ambiguous_repeat=~identifiable & earlier)
    out = out[~drop].reset_index(drop=True)
    counts = {"month": month, "period": heldout_period(month), "listings_before": len(listings),
              "removed_dev_overlap": int((identifiable & in_dev).sum()),
              "removed_earlier_in_period": int((identifiable & earlier & ~in_dev).sum()),
              "listings_after": len(out), "ambiguous_kept": int((~identifiable).sum()),
              "ambiguous_dev_match": int((~identifiable & in_dev).sum()), "ambiguous_repeat": int((~identifiable & earlier).sum())}
    return out, counts


def load_heldout_listings(month: str) -> tuple[pd.DataFrame, dict]:
    """A held-out month's analysis catalog after `dedup_heldout`, with the per-month counts."""
    period = heldout_period(month)
    if period is None:
        raise ValueError(f"{month} is not a held-out month")
    months = HELDOUT_PERIODS[period]
    earlier = set()
    for m in months[: months.index(month)]:
        earlier |= set(load_listings(m).unit_id)
    return dedup_heldout(load_listings(month), month, development_unit_ids(), earlier)


def heldout_dedup_report(path: Path | None = DEDUP_COUNTS_CSV) -> pd.DataFrame:
    """Counts for every held-out month (no outcomes are computed); written to data/firstmover/heldout_dedup_counts.csv."""
    t = pd.DataFrame([load_heldout_listings(m)[1] for months in HELDOUT_PERIODS.values() for m in months])
    if path is not None:
        t.to_csv(path, index=False)
    return t


@lru_cache(maxsize=1)
def gap_days() -> dict[str, set[str]]:
    path = ROOT / "data" / "firstmover" / "coverage_daily.csv"
    if not path.exists():
        return {}
    c = pd.read_csv(path)
    return {m: set(g.date) for m, g in c[c.low.astype(bool)].groupby("month")}


def feasible_counts(caps: np.ndarray, need: np.ndarray, rents: np.ndarray, bedrooms: np.ndarray) -> np.ndarray:
    """|E_b| per renter against one listing set: rent <= cap and bedrooms >= need."""
    out = np.zeros(len(caps), dtype=np.int64)
    for b in np.unique(need):
        sorted_rents = np.sort(rents[bedrooms >= b])
        sel = need == b
        out[sel] = np.searchsorted(sorted_rents, caps[sel], side="right")
    return out


def cell_frames(renters: pd.DataFrame, listings: pd.DataFrame, borough: str | None, month: str, share: float,
                recent_movers_only: bool = True, donors: str | None = "evaluation", bedroom_rule: str = "standard",
                rent_column: str = "rent") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renter frame and listing pool of a cell, with `cap`, `need_bedrooms`, `n_feasible_cell` and `group`.

    borough=None is the citywide scope (sensitivity check): every renter can search every listing.
    donors: 'evaluation', 'calibration' or None for all households.
    """
    r = renters[renters.borough == borough] if borough else renters
    if recent_movers_only:
        r = r[r.recent_mover]
    if donors:
        r = r[r.donors == donors]
    r = r.copy()
    l = (listings[listings.borough == borough] if borough else listings).copy().reset_index(drop=True)
    r["need_bedrooms"] = bedroom_need(r.household_size, bedroom_rule)
    r["cap"] = np.maximum(share * r.income / 12, 0.0)
    r["n_feasible_cell"] = np.where(r.income > 0, feasible_counts(r.cap.to_numpy(), r.need_bedrooms.to_numpy(),
                                                                  l[rent_column].to_numpy(float), l.bedrooms.to_numpy()), 0)
    r["group"] = np.select([r.lower_income & (r.n_feasible_cell >= 1), r.lower_income], ["G+", "G0"], "higher")
    # Older names, kept so code written against them (metrics, the jobs mirror) keeps working.
    r["income_month"], r["cap_month"] = r.income, r.cap
    return r.reset_index(drop=True), l
