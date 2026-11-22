"""Check the public enrollment, calibration, freeze and lottery summary tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify(folder):
    expected=json.loads((folder/"policy-freeze.json").read_text())
    for name in ("PLAN.md","retention.cpp","calibration-2021.csv"):
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected[name]
    calibration=rows(folder/"calibration-2021.csv")
    assert len(calibration)==175
    assert sum(int(r["school_pairs"]) for r in calibration)==82467
    assert sum(int(r["enrolled"]) for r in calibration)==9996
    groups={(r["year"],r["group"]):r for r in rows(folder/"coverage.csv")}
    for y in ("2022","2023","2024"):
        for key in groups[y,"all"]:
            if key in ("year","group"):continue
            assert int(groups[y,"all"][key])==int(groups[y,"pre_k"][key])+int(groups[y,"higher"][key])
    gains=[int(groups[y,"all"]["selected_hits"])-int(groups[y,"all"]["baseline_hits"]) for y in ("2022","2023","2024")]
    assert gains==[9,22,19]
    for split,limits,expected_batch,expected_policies in (
        ("design_audit",1024,"513",{"learned_retention"}),
        ("holdout_course",128,"8615729",{"learned_retention","unrestricted_rank"}),
        ("holdout_family",32,"8615731",{"learned_retention"}),
    ):
        all_rows=rows(folder/(split+".csv"))
        assert {r["policy"] for r in all_rows}==expected_policies
        assert all(r["batch"]==expected_batch for r in all_rows)
        by_key={(r["year"],r["policy"],int(r["draw"])):r for r in all_rows}
        keys={(year,arm,d) for year in ("2022","2023","2024") for arm in expected_policies for d in range(1,limits+1)}
        assert set(by_key)==keys and len(by_key)==len(all_rows)
        for (year,arm,d),r in by_key.items():
            base,exits,ent,changes,better,worse,placed,school=(int(r[k]) for k in
                  ("baseline_placed","exits","entries","course_switch","retained_better","retained_worse","placed","school_switch"))
            assert placed==base-exits+ent and changes==better+worse and 0<=school<=changes
            if split=="design_audit":assert exits==0
            if len(expected_policies)>1:
                assert r["baseline_placed"]==by_key[year,"learned_retention",d]["baseline_placed"]
    print("Freeze, coverage, cell counts, 3,072 design observations and held-out national paired cohorts verified.")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory",type=Path,default=Path(__file__).resolve().parent,nargs="?")
    verify(ap.parse_args().directory)
