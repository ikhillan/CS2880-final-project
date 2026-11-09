"""Flag collection gaps in every cleaned FirstMover month.

A day counts as low when its NYC analysis-catalog postings fall below 40% of the month's median weekday volume
(24% on weekends, which are naturally quieter). Isolated holiday and weekend lows are normal; runs of low days
are collection gaps.

Writes
    data/firstmover/coverage_daily.csv    one row per month and day: postings, threshold, low flag
    data/firstmover/coverage_months.csv   one row per month: listings, days present, low days, status

Usage
    python scripts/firstmover_coverage.py
"""
from __future__ import annotations

import calendar
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FM = ROOT / "data" / "firstmover"
WEEKDAY_SHARE, WEEKEND_SHARE = 0.40, 0.24
GAP_RUN = 3  # this many low days in a row marks a collection gap


def month_days(month: str, catalog: pd.DataFrame) -> pd.DataFrame:
    y, m = map(int, month.split("-"))
    days = pd.date_range(f"{month}-01", periods=calendar.monthrange(y, m)[1], freq="D")
    counts = pd.to_datetime(catalog.created_at_utc, utc=True).dt.strftime("%Y-%m-%d").value_counts()
    d = pd.DataFrame({"month": month, "date": days.strftime("%Y-%m-%d"), "weekday": days.day_of_week < 5})
    d["postings"] = d.date.map(counts).fillna(0).astype(int)
    median_weekday = float(d.loc[d.weekday, "postings"].median())
    d["threshold"] = median_weekday * d.weekday.map({True: WEEKDAY_SHARE, False: WEEKEND_SHARE})
    d["low"] = d.postings < d.threshold
    return d


def longest_run(flags: pd.Series) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def main() -> None:
    daily, months = [], []
    for path in sorted((FM / "clean").glob("firstmover_listings_*.csv.gz")):
        month = path.name.removeprefix("firstmover_listings_").removesuffix(".csv.gz")
        c = pd.read_csv(path, usecols=["created_at_utc", "in_analysis_catalog"], low_memory=False)
        catalog = c[c.in_analysis_catalog.astype(bool)]
        d = month_days(month, catalog)
        daily.append(d)
        run = longest_run(d.low)
        months.append({"month": month, "nyc_catalog_listings": len(catalog), "days_with_postings": int((d.postings > 0).sum()),
                       "days_in_month": len(d), "median_weekday_postings": int(d.loc[d.weekday, "postings"].median()),
                       "low_days": int(d.low.sum()), "longest_low_run": run,
                       "low_dates": ",".join(d.loc[d.low, "date"].str[8:].str.lstrip("0")),
                       "status": "gap" if run >= GAP_RUN or (d.postings == 0).sum() >= GAP_RUN else "complete"})
    pd.concat(daily).to_csv(FM / "coverage_daily.csv", index=False)
    summary = pd.DataFrame(months)
    summary.to_csv(FM / "coverage_months.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
