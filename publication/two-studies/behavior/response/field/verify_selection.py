"""Check sharp common-response bounds against national re-clears and field endpoints."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(inputs, results, output):
    totals = defaultdict(Counter)
    grid_fields = ("responders", "eligible_union", "neutral_padding", "adding_blind", "adding_rank",
                   "blind_april", "rank_april", "blind_linked_placed", "rank_linked_placed",
                   "blind_placed", "rank_placed", "exits", "entries")
    for year in range(2022, 2026):
        cohorts = {r["scenario"]: r for r in rows(results / f"selection-{year}.csv")}
        field = {r["scenario"]: r for r in rows(inputs / f"field-{year}.csv") if r["permille"] == "1000"}
        assert len(cohorts) == 2
        for name, r in cohorts.items():
            full = field[name + "_original"]
            for prefix, exclusive in (("blind", "blind_only"), ("rank", "informed_only")):
                hits = int(r[("informed" if prefix=="rank" else prefix)+"_original_hits"])
                assert int(full[prefix+"_april"]) == hits + int(r[exclusive]) + int(r["common"])
        keys = set()
        for row in rows(results / f"selection-replay-{year}.csv"):
            key = tuple(row[k] for k in ("scenario", "position", "target_bound", "responder_set"))
            assert key not in keys
            keys.add(key)
            name,position,bound,padding = key
            r = cohorts[name]
            v = {k:int(row[k]) for k in grid_fields}
            gap = int(r["informed_original_hits"]) - int(r["blind_original_hits"])
            correction = -int(r["blind_only"]) if bound=="lower" else int(r["informed_only"])
            assert v["rank_april"] - v["blind_april"] == gap + correction
            assert v["rank_placed"] - v["blind_placed"] == v["entries"] - v["exits"]
            assert v["responders"] - v["neutral_padding"] == abs(correction)
            assert v["eligible_union"] == int(r["blind_eligible"]) + int(r["informed_eligible"]) - int(r["jointly_eligible"])
            assert v["responders"] == (abs(correction) if padding=="only_exclusive" else v["eligible_union"]*199//1000)
            totals[key].update(v)
        assert len(keys)==16
    with output.open("w", newline="") as stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(["scenario","position","bound","responder_set","responders","eligible_union","neutral_padding",
                         "blind_adding","informed_adding","april_gap","linked_placement_gap","placement_gap","exits","entries"])
        for (s,pos,bnd,mode),v in sorted(totals.items()):
            writer.writerow([s,pos,bnd,mode,v["responders"],v["eligible_union"],v["neutral_padding"],
                             v["adding_blind"],v["adding_rank"],v["rank_april"]-v["blind_april"],
                             v["rank_linked_placed"]-v["blind_linked_placed"],v["rank_placed"]-v["blind_placed"],v["exits"],v["entries"]])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("inputs","results","out"):
        parser.add_argument("--"+flag,required=True,type=Path)
    args=parser.parse_args()
    main(args.inputs,args.results,args.out)
