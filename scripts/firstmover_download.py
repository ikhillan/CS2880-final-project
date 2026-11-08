"""Download one month of FirstMover NYC open rental-listing data.

Source: the FirstMover Open Data Project (https://www.firstmovernyc.com/open-data). Its
download links point to monthly CSVs in the public GitHub repo
benfwalla/firstmover-open-data-project (MIT license), so this script fetches exactly those
files; no HTML scraping. robots.txt for firstmovernyc.com is "Allow: /" (checked 2026-10-31).

Writes
    data/firstmover/raw/<YYYY-MM>.csv            raw file, byte-for-byte as published
    data/firstmover/raw/<YYYY-MM>.manifest.json  provenance: URLs, download time, sha256,
                                                 upstream commit, row count, license notes

Usage
    python scripts/firstmover_download.py --month 2026-09
"""
import argparse
import csv
import datetime as dt
import hashlib
import html
import io
import json
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "firstmover" / "raw"

OPEN_DATA_PAGE = "https://www.firstmovernyc.com/open-data"
REPO = "benfwalla/firstmover-open-data-project"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/main/public/data/{{month}}.csv"
COMMITS_API = f"https://api.github.com/repos/{REPO}/commits"
HEADERS = {"User-Agent": "COMPSCI2880-course-research-script"}

LICENSE_NOTE = {
    "repository_license": "MIT (LICENSE in github.com/benfwalla/firstmover-open-data-project, "
                          "Copyright (c) 2025-2026 Ben Wallace)",
    "site_statement": "Every month, we collect publicly available rental listings and publish the "
                      "raw data here for anyone to use. ... No copyrighted content like descriptions "
                      "is reproduced. This project is not affiliated with or endorsed by StreetEasy "
                      "or Zillow Group.",
    "upstream_origin": "Listings originate from StreetEasy (Zillow Group); FirstMover collects them.",
    "our_usage_rules": [
        "Non-commercial academic use (COMPSCI 2880); cite the FirstMover Open Data Project.",
        "Do not commit or redistribute the raw CSVs; keep them git-ignored.",
        "Do not use the url / photo-id fields to crawl StreetEasy or Zillow.",
    ],
}


def page_listing_count(month: str) -> int | None:
    """Listing count shown for `month` on the Open Data page (cross-check only)."""
    try:
        r = requests.get(OPEN_DATA_PAGE, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return None
    text = html.unescape(re.sub(r"<[^>]+>", " ", r.text))
    label = dt.date.fromisoformat(month + "-01").strftime("%B %Y")
    # counts are printed with thousands separators; skip notes like "Data not available Nov 1-7"
    m = re.search(re.escape(label) + r".{0,80}?(\d{1,3}(?:,\d{3})+)", text, flags=re.S)
    return int(m.group(1).replace(",", "")) if m else None


def upstream_commit(month: str) -> dict | None:
    try:
        r = requests.get(COMMITS_API, params={"path": f"public/data/{month}.csv", "per_page": 1},
                         headers=HEADERS, timeout=30)
        r.raise_for_status()
        c = r.json()[0]
    except (requests.RequestException, IndexError, KeyError, ValueError):
        return None
    return {"sha": c["sha"], "date": c["commit"]["author"]["date"], "message": c["commit"]["message"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", required=True, help="YYYY-MM, a completed month listed on the Open Data page")
    ap.add_argument("--force", action="store_true", help="overwrite an existing raw file")
    args = ap.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}", args.month):
        sys.exit("--month must look like 2026-09")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{args.month}.csv"
    if out.exists() and not args.force:
        sys.exit(f"{out} already exists; pass --force to re-download")

    url = RAW_URL.format(month=args.month)
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    body = r.content
    out.write_bytes(body)

    rows = list(csv.reader(io.StringIO(body.decode("utf-8"))))
    manifest = {
        "dataset": "FirstMover Open Data Project - monthly NYC rental listings",
        "month": args.month,
        "month_definition": "UTC calendar month of created_at_utc (listing first seen on market)",
        "source_page": OPEN_DATA_PAGE,
        "file_url": url,
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "http_status": r.status_code,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "n_rows": len(rows) - 1,
        "n_columns": len(rows[0]),
        "columns": rows[0],
        "listing_count_shown_on_page": page_listing_count(args.month),
        "upstream_last_commit": upstream_commit(args.month),
        "license_and_usage": LICENSE_NOTE,
    }
    (RAW_DIR / f"{args.month}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("file_url", "bytes", "sha256", "n_rows", "n_columns",
                                               "listing_count_shown_on_page", "upstream_last_commit")}, indent=2))


if __name__ == "__main__":
    main()
