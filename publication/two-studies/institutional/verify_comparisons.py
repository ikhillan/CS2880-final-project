"""Verify published comparisons across the pooled and reconstructed mechanisms."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def main():
    first_changes = []
    for year in range(2022, 2026):
        grouped = {r["group"]: r for r in rows(RESULTS / f"first-{year}.csv")}
        overall = grouped.pop("linked")
        for k in ("linked", "first_hits", "rank_hits", "first_placed", "rank_placed", "exits", "entrants"):
            assert int(overall[k]) == sum(int(r[k]) for r in grouped.values())
        all_rows = [overall] + [grouped[g] for g in ("enrolled_at_initial_offer", "enrolled_elsewhere", "no_initial_offer")]
        changes = []
        for row in all_rows:
            places = int(row["rank_placed"]) - int(row["first_placed"])
            assert places == int(row["entrants"]) - int(row["exits"])
            changes.append(places)
        assert changes[0] > 0 and int(overall["rank_hits"]) > int(overall["first_hits"])
        first_changes.append(changes)
    assert list(map(sum, zip(*first_changes))) == [8143, 6149, 4854, -2860]
    for year in range(2022, 2025):
        old = rows(ROOT.parent / "design/risk" / f"ceiling-{year}.csv")
        regenerated = {(r["support"], r["target"]): r for r in rows(RESULTS / f"pooled-{year}.csv") if r["priority"] == "design"}
        assert len(old) == len(regenerated)
        for row in old:
            replica = regenerated[row["support"], row["target"]]
            assert all(row[key] == replica[key] for key in ("present_at_baseline", "additional_reachable"))
    overlap = [r for r in rows(RESULTS/"ceiling-overlap.csv") if r["model"] == "paired" and r["target"] == "april_school"]
    assert [sum(int(r[k]) for r in overlap) for k in ("both","pooled_only","reconstruction_only")] == [252,26,62]
    receipts = rows(RESULTS / "exact-design-checks.csv")
    for spec, split, count in (("exact","design_check",3072),("paired","holdout",192),("paired","recorded",3)):
        selected = [r for r in receipts if (r["specification"],r["split"],r["certificate"]) == (spec,split,"stable")]
        assert sum(int(row["count"]) for row in selected) == count
        assert all(int(row["changed_detailed"]) == 0 for row in selected)
    print("Verified both comparators, the original pooled ceilings, the overlap and every matched design draw.")

if __name__ == "__main__":
    main()
