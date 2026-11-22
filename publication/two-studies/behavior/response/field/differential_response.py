"""Compare national counterfactuals when uptake can differ by shortlist."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

YEARS = (2022, 2023, 2024, 2025)
METRICS = ("linked_placed", "april", "placed")


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(args):
    national = defaultdict(lambda: defaultdict(int))
    scenarios = defaultdict(set)
    cohorts = defaultdict(set)
    for yr in YEARS:
        rows = read_rows(args.inputs / f"field-{yr}.csv")
        baseline, = [r for r in read_rows(args.inputs / f"endpoints-{yr}.csv") if r["scenario"] == "binding"]
        for scenario in sorted(set(r["scenario"] for r in rows)):
            rows.append(dict(baseline, scenario=scenario))
        for r in rows:
            key = r["scenario"], int(r["permille"]), int(r["rep"])
            assert int(r["year"]) == yr and yr not in cohorts[key]
            cohorts[key].add(yr)
            scenarios[key[0]].add(key[1])
            for metric in METRICS:
                for policy in ("blind", "rank"):
                    field = f"{policy}_{metric}"
                    national[key][field] += int(r[field])
    national = dict(national)
    for s,q,j in national:
        assert cohorts[s,q,j] == set(YEARS)
    results = []
    for s, qs in sorted(scenarios.items()):
        for qb in sorted(qs):
            for qr in sorted(qs):
                seeds = (0,) if qb in (0,1000) and qr in (0,1000) else tuple(range(8))
                def arm(q, j):
                    return national[s,q,0 if q in (0,1000) else j]
                result = {"scenario":s, "blind_permille":qb, "rank_permille":qr, "draws":len(seeds)}
                deltas = {}
                for metric in METRICS:
                    diffs = [arm(qr,j)["rank_"+metric] - arm(qb,j)["blind_"+metric] for j in seeds]
                    deltas[metric] = diffs
                    result.update({metric+"_mean":mean(diffs), metric+"_min":min(diffs), metric+"_max":max(diffs)})
                result["draws_more_april_fewer_linked_places"] = sum(h>0 and p<0 for h,p in zip(deltas["april"],deltas["linked_placed"]))
                results.append(result)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=results[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("inputs","out"):
        parser.add_argument("--"+arg,required=True,type=Path)
    main(parser.parse_args())
