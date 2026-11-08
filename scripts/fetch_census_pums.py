"""Rebuild census_pums/ from the official Census files.

Downloads the ACS 2020-2024 5-year PUMS for New York State (housing and person files) and the PUMS
data dictionary, keeps the 55 NYC PUMAs (2020 PUMA codes 041xx-045xx), and adds one column,
NYC_BOROUGH, derived from the PUMA prefix. All Census columns and values are kept as published.

Writes
    census_pums/acs_pums_2020_2024_nyc_households.csv
    census_pums/acs_pums_2020_2024_nyc_persons.csv
    census_pums/acs_pums_2020_2024_data_dictionary.csv

Usage
    python scripts/fetch_census_pums.py            # about 200 MB download, a few minutes
    python scripts/fetch_census_pums.py --out DIR  # write somewhere other than census_pums/
"""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www2.census.gov/programs-surveys/acs/data/pums/2024/5-Year"
FILES = {  # zip on the Census site -> (CSV inside it, output name)
    "csv_hny.zip": ("psam_h36.csv", "acs_pums_2020_2024_nyc_households.csv"),
    "csv_pny.zip": ("psam_p36.csv", "acs_pums_2020_2024_nyc_persons.csv"),
}
DICTIONARY = "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2020-2024.csv"
BOROUGH_BY_PUMA_PREFIX = {"041": "Manhattan", "042": "Bronx", "043": "Brooklyn", "044": "Queens", "045": "Staten Island"}
HEADERS = {"User-Agent": "COMPSCI2880-course-research-script"}


def download(url: str, dest: Path) -> None:
    with requests.get(url, headers=HEADERS, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            shutil.copyfileobj(r.raw, fh)


def extract_nyc(zip_path: Path, member: str, out_path: Path) -> tuple[int, int]:
    """Stream the state CSV, keep NYC PUMAs, append NYC_BOROUGH. Returns (rows, distinct PUMAs)."""
    rows, pumas, first = 0, set(), True
    with zipfile.ZipFile(zip_path) as z, z.open(member) as fh:
        for chunk in pd.read_csv(fh, dtype=str, keep_default_na=False, chunksize=200_000):
            nyc = chunk[chunk.PUMA.str[:3].isin(BOROUGH_BY_PUMA_PREFIX)].copy()
            nyc["NYC_BOROUGH"] = nyc.PUMA.str[:3].map(BOROUGH_BY_PUMA_PREFIX)
            nyc.to_csv(out_path, mode="w" if first else "a", header=first, index=False, lineterminator="\n")
            first = False
            rows += len(nyc)
            pumas |= set(nyc.PUMA)
    return rows, len(pumas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "census_pums", help="output directory")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for zname, (member, outname) in FILES.items():
            zpath = Path(tmp) / zname
            print(f"downloading {BASE}/{zname} ...")
            download(f"{BASE}/{zname}", zpath)
            rows, n_pumas = extract_nyc(zpath, member, args.out / outname)
            if n_pumas != 55:
                raise SystemExit(f"{outname}: expected 55 NYC PUMAs, found {n_pumas}")
            print(f"wrote {args.out / outname}: {rows:,} rows, {n_pumas} PUMAs")
    download(DICTIONARY, args.out / "acs_pums_2020_2024_data_dictionary.csv")
    print(f"wrote {args.out / 'acs_pums_2020_2024_data_dictionary.csv'}")


if __name__ == "__main__":
    main()
