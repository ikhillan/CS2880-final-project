"""Recompute published summaries without applicant-level data."""
import csv
from collections import defaultdict
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"

def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def checked_summary(script, args, name, temporary):
    output = temporary / name
    subprocess.run([sys.executable, HERE / "field" / script, *args, "--out", output], check=True)
    assert rows(output) == rows(EVIDENCE / name), name

def main():
    with TemporaryDirectory() as directory:
        scratch = Path(directory)
        institutional = HERE.parent.parent / "institutional" / "results"
        subprocess.run([sys.executable, HERE/"engine/summarize_response.py", "--results", EVIDENCE,
                        "--institutional", institutional,"--out",scratch],check=True)
        assert rows(scratch/"response-summary.csv") == rows(EVIDENCE/"response-summary.csv")
        for script,args,name in [
            ("verify_selection.py",["--inputs",EVIDENCE,"--results",EVIDENCE],"selection-summary.csv"),
            ("verify_participation.py",["--relisted",EVIDENCE,"--results",EVIDENCE],"participation-summary.csv"),
            ("differential_response.py",["--inputs",EVIDENCE],"differential-response.csv"),
            ("coverage_thresholds.py",["--inputs",EVIDENCE],"coverage-thresholds.csv")]:
            checked_summary(script,args,name,scratch)
    old = rows(HERE.parent / "field/round-link-counts.csv")
    for year in range(2022, 2026):
        historical, = [r for r in old if r["group"] == "regular_all" and int(r["year"]) == year]
        current, = [r for r in rows(EVIDENCE/f"relisted-{year}.csv") if r["group"] == "all"]
        assert int(current["returned_same_grade_pupils"]) == int(historical["returns_same_grade"])
        assert int(current["outside_original_school_pupils"]) == int(historical["any_new_school"])
    capacity = rows(EVIDENCE/"second-chances.csv")
    assert len(capacity) == 16
    totals = defaultdict(lambda: defaultdict(int))
    for row in capacity:
        earlier = [r for r in rows(institutional / f"transitions-{row['year']}.csv") if r["stratum"] in ("all","all_linked")]
        diff = {r["stratum"]:int(r["rank_placed"])-int(r["blind_placed"]) for r in earlier}
        v = {key:int(value) for key,value in row.items() if key not in ("choices","supply")}
        assert (v["first_stage_gap"],v["linked_first_stage_gap"]) == (diff["all"],diff["all_linked"])
        assert v["difference_in_maxima"] == v["rank_maximum"] - v["blind_maximum"]
        assert v["combined_gap"] == v["first_stage_gap"] + v["difference_in_maxima"]
        assert -v["blind_only_upper"] <= v["difference_in_maxima"] <= v["rank_only_upper"]
        assert v["linked_combined_gap"] == v["linked_first_stage_gap"] + v["rank_linked_maximum"]-v["blind_linked_maximum"]
        assert v["rank_linked_maximum"]-v["blind_linked_maximum"] <= v["linked_rank_only_upper"]
        assert v["pursued_hit_gap"] == v["rank_pursued_hits"]-v["blind_pursued_hits"]
        aggregate = totals[row["choices"],row["supply"]]
        for k, value in v.items():
            aggregate[k] += value
    for key, expected in {
        ("new_school","published_A2"):(-1558,-1419,4752),
        ("new_school","common_regular_residual"):(-1580,-1439,4743),
        ("all_explicit","published_A2"):(-1550,-1409,4763),
        ("all_explicit","common_regular_residual"):(-1574,-1433,4748)
    }.items():
        assert tuple(totals[key][k] for k in ("combined_gap","linked_combined_gap","pursued_hit_gap")) == expected
    assert totals["new_school","published_A2"]["rank_only_upper"] == 388
    assert totals["new_school","published_A2"]["linked_rank_only_upper"] == 358
    print("Verified response, selection, participation and capacity comparisons against the released annual aggregates.")

if __name__ == "__main__":
    main()
