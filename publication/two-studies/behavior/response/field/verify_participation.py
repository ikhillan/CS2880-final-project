"""Compare the hypothetical choice rules on the identical historical return roster."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(a):
    totals = defaultdict(Counter)
    fields = ["reapplicants","can_add_either","adding_blind","adding_rank","blind_april","rank_april",
              "blind_linked_placed","rank_linked_placed","blind_placed","rank_placed","exits","entries"]
    for year in range(2022, 2026):
        cohort={r["group"]: int(r["returned_same_grade_pupils"]) for r in read(a.relisted/f"relisted-{year}.csv")}
        bounds={r["scenario"]:r for r in read(a.results/f"selection-{year}.csv")}
        annual={}
        for r in read(a.results/f"participation-{year}.csv"):
            name,pos,group = (r[k] for k in ("scenario", "position", "responders"))
            v = {k:int(r[k]) for k in fields}
            bound=bounds[name]
            gap=int(bound["informed_original_hits"])-int(bound["blind_original_hits"])
            assert int(r["year"])==year
            assert v["reapplicants"]==cohort["all" if group=="same_grade" else "never_offered"]
            assert max(v["adding_blind"],v["adding_rank"])<=v["can_add_either"]<=v["reapplicants"]
            assert gap-int(bound["blind_only"])<=v["rank_april"]-v["blind_april"]<=gap+int(bound["informed_only"])
            assert v["rank_placed"]-v["blind_placed"]==v["entries"]-v["exits"]
            key=(name,pos,group)
            assert key not in annual
            annual[key]=v
            totals[key].update(v)
        assert len(annual)==8
        for (name,pos,group),row in annual.items():
            twin=annual[name,"original" if pos=="tail" else "tail",group]
            assert all(row[k]==twin[k] for k in fields[:6])
            if group=="same_grade_never_offered":
                assert all(row[k]<=annual[name,pos,"same_grade"][k] for k in fields[:4])
    with a.out.open("w",newline="") as stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(["candidate","position","responders","reapplicants","can_add_either","adding_blind","adding_rank",
                         "april_gap","linked_placement_gap","placement_gap","exits","entries"])
        for (name,pos,group),v in sorted(totals.items()):
            writer.writerow([name,pos,group,v["reapplicants"],v["can_add_either"],v["adding_blind"],v["adding_rank"],
                             v["rank_april"]-v["blind_april"],v["rank_linked_placed"]-v["blind_linked_placed"],
                             v["rank_placed"]-v["blind_placed"],v["exits"],v["entries"]])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("relisted","results","out"):
        p.add_argument("--"+flag,required=True,type=Path)
    main(p.parse_args())
