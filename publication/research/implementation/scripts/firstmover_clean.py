"""Clean one month of FirstMover listings into a project-ready table.

Input   data/firstmover/raw/<YYYY-MM>.csv      (from scripts/firstmover_download.py)
Output  data/firstmover/clean/firstmover_listings_<YYYY-MM>.parquet  (+ .csv.gz copy)

One output row per raw row (nothing dropped); analysis code should filter on
`in_analysis_catalog`. Column definitions are in data/firstmover/README.md.

ID strategy
- listing_uid = "fm:<id>"   one posting. `id` is unique within a month and never repeats across
                            months (checked for 2026-08 vs 2026-09).
- unit_id     = "fmu:<sha1(unit_key)[:12]>"  one physical apartment, stable across months.
  unit_key comes from the StreetEasy URL path "building/<building-slug>/<unit-slug>" (98% of rows),
  else building slug + normalized `unit`, else "addr/<street>|<zip>/<unit>".
- building_id = "fmb:<sha1(building_key)[:12]>"
- firm_id     = slug of source_group for PARTNER (brokerage) and FEED (manager/feed) listings;
                owner-listed units (source_type OWNER, source_group "Owner") each get a private
                firm "owner:<unit_id>" so they are not pooled into one fake firm.

Usage
    python scripts/firstmover_clean.py --month 2026-08
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `python -I scripts/...` finds common
from common import DATA, PRICE_MAX, PRICE_MIN, assign_puma, borough_of_puma, load_puma_polygons

RAW_DIR = DATA / "firstmover" / "raw"
CLEAN_DIR = DATA / "firstmover" / "clean"
NON_RESIDENTIAL = {"COMMERCIAL", "LAND"}


def _hash(key: str, n: int = 12) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:n]


def _norm_unit(u: str) -> str:
    return re.sub(r"[^a-z0-9]", "", u.lower())


def _norm_street(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def _slug(s: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")


def unit_and_building_keys(url: str, street: str, unit: str, zip_code: str) -> tuple[str, str, str]:
    """(unit_key, building_key, unit_key_source)."""
    path = re.sub(r"^https?://(www\.)?streeteasy\.com/", "", url.strip().lower()).rstrip("/")
    parts = path.split("/") if path else []
    u = _norm_unit(unit)
    if len(parts) == 3 and parts[0] == "building":
        return f"building/{parts[1]}/{parts[2]}", f"building/{parts[1]}", "url_unit"
    if len(parts) >= 2 and parts[0] == "building":
        bkey = f"building/{parts[1]}"
        return (f"{bkey}/{u}", bkey, "url_building+unit_field") if u else (bkey, bkey, "url_building_only")
    bkey = f"addr/{_norm_street(street)}|{zip_code.strip()}"
    return (f"{bkey}/{u}", bkey, "address+unit") if u else (bkey, bkey, "address_only")


def to_bool(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower().map({"true": True, "false": False}).astype("boolean")


def clean(month: str) -> pd.DataFrame:
    raw = pd.read_csv(RAW_DIR / f"{month}.csv", dtype=str, keep_default_na=False)
    num = lambda c: pd.to_numeric(raw[c].replace("", np.nan), errors="coerce")

    keys = [unit_and_building_keys(*t) for t in zip(raw.url, raw.street, raw.unit, raw.zip_code)]
    df = pd.DataFrame({
        "listing_uid": "fm:" + raw.id,
        "source_listing_id": num("id").astype("Int64"),
        "unit_key": [k[0] for k in keys],
        "building_key": [k[1] for k in keys],
        "unit_key_source": [k[2] for k in keys],
    })
    df["unit_id"] = "fmu:" + df.unit_key.map(_hash)
    df["building_id"] = "fmb:" + df.building_key.map(_hash)
    df["created_at_utc"] = pd.to_datetime(raw.created_at_utc, utc=True)
    df["month"] = month
    df["available_date"] = pd.to_datetime(raw.available_date.replace("", np.nan), errors="coerce").dt.date

    # rent and lease terms ("0" means no concession / not reported)
    df["rent"] = num("price").astype("Int64")
    ne = num("net_effective_price")
    df["net_effective_rent"] = ne.where(ne > 0, df.rent.astype(float))
    df["months_free"] = num("months_free").fillna(0.0)
    df["has_concession"] = df.months_free > 0
    df["lease_months"] = num("lease_months").astype("Int64")

    # unit size ("0" sqft means not reported)
    df["bedrooms"] = num("bedrooms").astype("Int64")
    df["bathrooms_full"] = num("bathrooms").astype("Int64")
    df["half_baths"] = num("half_baths").astype("Int64")
    df["bathrooms_total"] = df.bathrooms_full.astype(float) + 0.5 * df.half_baths.astype(float)
    sqft = num("sqft")
    df["sqft"] = sqft.where(sqft > 0)

    # location
    df["street"], df["unit"], df["zip_code"] = raw.street, raw.unit.replace("", pd.NA), raw.zip_code
    df["neighborhood"] = raw.neighborhood.replace("", pd.NA)
    df["borough_source"] = raw.borough.replace("", pd.NA)
    df["state"] = raw.state
    df["latitude"], df["longitude"] = num("latitude"), num("longitude")
    df["puma"] = assign_puma(df.longitude, df.latitude, load_puma_polygons())
    df["borough"] = df.puma.map(borough_of_puma)
    df["in_nyc"] = df.puma.notna()

    # who listed it
    df["source_type"] = raw.source_type
    df["source_group"] = raw.source_group.replace("", pd.NA)
    df["firm_kind"] = raw.source_type.map({"PARTNER": "brokerage", "FEED": "manager_or_feed", "OWNER": "owner"})
    firm = df.source_group.fillna("unknown").map(_slug)
    df["firm_id"] = np.where(df.source_type == "OWNER", "owner:" + df.unit_id, "firm:" + firm)

    # attributes
    df["building_type"] = raw.building_type
    for c in ["furnished", "is_new_development", "has_videos", "has_3d_tour"]:
        df[c] = to_bool(raw[c])
    df["no_fee_raw"] = to_bool(raw.no_fee)  # constant False in 2026-08/09 exports; see README
    df["media_asset_count"] = num("media_asset_count").astype("Int64")
    df["n_photos"] = raw.photo_ids.map(lambda s: len([p for p in s.split(",") if p.strip()])).astype("Int64")
    df["has_open_house"] = raw.open_house_start_utc.str.strip() != ""
    df["open_house_appointment_only"] = to_bool(raw.open_house_appointment_only)
    df["url"] = raw.url.replace("", pd.NA)  # provenance only; do not crawl

    # quality flags
    df["flag_price_outlier"] = ~df.rent.between(PRICE_MIN, PRICE_MAX)
    df["flag_borough_mismatch"] = df.in_nyc & (df.borough_source != df.borough)
    df["flag_bath_outlier"] = (df.bathrooms_full > 6) | (df.half_baths > 4)
    df["flag_sqft_implausible"] = df.sqft.notna() & ~df.sqft.between(100, 10_000)
    df["flag_non_residential"] = df.building_type.isin(NON_RESIDENTIAL)

    # reposts: same unit listed more than once this month -> keep the first posting
    df = df.sort_values(["created_at_utc", "source_listing_id"]).reset_index(drop=True)
    df["unit_postings_in_month"] = df.groupby("unit_id").listing_uid.transform("size")
    df["is_first_in_month"] = ~df.duplicated("unit_id")

    df["in_analysis_catalog"] = (df.in_nyc & ~df.flag_price_outlier & df.is_first_in_month
                                 & ~df.flag_non_residential & ~df.flag_bath_outlier)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", required=True, help="YYYY-MM already downloaded to data/firstmover/raw/")
    args = ap.parse_args()
    df = clean(args.month)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out = CLEAN_DIR / f"firstmover_listings_{args.month}"
    df.to_parquet(out.with_suffix(".parquet"), index=False)
    df.to_csv(out.with_suffix(".csv.gz"), index=False)
    print(f"wrote {out}.parquet / .csv.gz: {len(df):,} rows, {df.in_nyc.sum():,} in NYC, "
          f"{df.in_analysis_catalog.sum():,} in analysis catalog")


if __name__ == "__main__":
    main()
