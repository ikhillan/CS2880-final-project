"""Count the development/held-out physical-unit dedup for every held-out month (protocol section 3). Counts only: no rents,
markets or outcomes are computed.

For each held-out month (hiddenoptions.data.HELDOUT_PRIMARY, HELDOUT_REPLICATION) it writes the listings before and after,
identifiable units removed because they appear in a development month, those removed because the unit already arrived
earlier in the same held-out period, and the ambiguous units kept with a flag (unit key not from the StreetEasy URL unit
path), with how many of those match a development unit or an earlier arrival.

Writes
    data/firstmover/heldout_dedup_counts.csv

Usage
    python scripts/heldout_dedup_report.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hiddenoptions.data import DEDUP_COUNTS_CSV, heldout_dedup_report  # noqa: E402


def main() -> None:
    t = heldout_dedup_report(DEDUP_COUNTS_CSV)
    print(t.to_string(index=False))
    print(f"wrote {DEDUP_COUNTS_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
