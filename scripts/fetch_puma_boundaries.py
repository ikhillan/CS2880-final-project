"""Download the 55 NYC 2020 PUMA polygons from Census TIGERweb as GeoJSON (EPSG:4326).

These are the same PUMA definitions used by the ACS 2020-2024 PUMS extract in census_pums/,
so listing coordinates can be joined to PUMS geography.

Writes data/geo/nyc_puma2020.geojson and data/geo/nyc_puma2020.provenance.json

Usage
    python scripts/fetch_puma_boundaries.py
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `python -I scripts/...` finds common
from common import DATA

SERVICE = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
           "PUMA_TAD_TAZ_UGA_ZCTA/MapServer/0/query")  # layer 0 = 2020 Census PUMAs
PARAMS = {
    "where": "STATE='36' AND PUMA LIKE '04%'",  # NYC PUMAs are 041xx-045xx
    "outFields": "STATE,PUMA,NAME,BASENAME",
    "outSR": "4326",
    "geometryPrecision": "5",
    "returnGeometry": "true",
    "f": "geojson",
}
HEADERS = {"User-Agent": "COMPSCI2880-course-research-script"}


def main() -> None:
    r = requests.get(SERVICE, params=PARAMS, headers=HEADERS, timeout=120)
    r.raise_for_status()
    gj = r.json()
    pumas = sorted(f["properties"]["PUMA"] for f in gj["features"])
    if len(pumas) != 55:
        raise SystemExit(f"expected 55 NYC PUMAs, got {len(pumas)}")
    out = DATA / "geo" / "nyc_puma2020.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(gj, separators=(",", ":"))
    out.write_text(body)
    prov = {
        "source": "U.S. Census Bureau TIGERweb, layer '2020 Census Public Use Microdata Areas'",
        "request_url": r.url,
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_features": len(pumas),
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "license": "U.S. Government work (public domain)",
    }
    (DATA / "geo" / "nyc_puma2020.provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"wrote {out} ({len(pumas)} PUMAs)")


if __name__ == "__main__":
    main()
