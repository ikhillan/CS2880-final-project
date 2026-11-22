"""Summarize pairs of national clears. All input and output files contain aggregate counts."""
import argparse
import csv
import math
import statistics
from pathlib import Path

GROUPS = ("all", "all_linked", "entry", "entry_linked")
GRID = (250, 450, 500, 550, 750)
REPS = 64


def read_counts(path):
    with path.open(newline="") as f:
        return [{key: (value if key == "group" else int(value))
                 for key, value in row.items()} for row in csv.DictReader(f)]


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pair_outcomes(row):
    k = row["permille"]
    level = (row["next_placed"] + row["anti_placed"]) / 2
    same_level = (row["next_same_placed"] + row["anti_same_placed"]) / 2
    own = (row["ht_next_numerator"] + row["ht_anti_numerator"]) * 500 / (k * (1000 - k))
    return dict(own_direct=own, placement_level=level, change_all=level-row["base_placed"],
                change_same=same_level-row["base_same_placed"])


def estimates(group, design, measure, data, n):
    mean = statistics.fmean(data)
    se = statistics.stdev(data) / math.sqrt(len(data))
    return dict(group=group,design=design,measure=measure,pairs=len(data),n=n,
                count=f"{mean:.6f}",count_mcse=f"{se:.6f}",
                pp=f"{100*mean/n:.8f}",pp_mcse=f"{100*se/n:.8f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--shards", nargs="+", type=Path,
                    help="merge the partitioned integer outputs into mixed-draws.csv")
    ap.add_argument("--cohort", type=Path, help="cross-check four end points against independent national reconstruction")
    args = ap.parse_args()
    directory = args.directory
    endpoint = {r["group"]: r for r in read_counts(directory / "mixed-endpoints.csv")}
    assert set(endpoint) == set(GROUPS)
    if args.cohort:
        cohort = {r["group"]: r for r in read_counts(args.cohort)}
        for g in GROUPS:
            suffix = "_linked_placed" if g.endswith("_linked") else "_placed"
            r = cohort[g.split("_")[0]]
            e = endpoint[g]
            assert e["n"] == r["linked" if g.endswith("_linked") else "applicants"]
            assert (e["blind_placed"], e["rank_placed"]) == (r["blind"+suffix], r["rank"+suffix])
    input_file = directory / "mixed-draws.csv"
    draws = ([r for path in args.shards for r in read_counts(path)]
             if args.shards else read_counts(input_file))
    draws.sort(key=lambda r: (r["pair"], r["permille"], GROUPS.index(r["group"])))
    assert len(draws) == REPS*len(GRID)*len(GROUPS)
    keys = {(r["pair"], r["permille"], r["group"]):r for r in draws}
    assert len(keys) == len(draws)
    for rep in range(1, REPS+1):
        for k in GRID:
            for g in GROUPS:
                row = keys[rep,k,g]
                e = endpoint[g]
                assert row["population"] == e["n"] and row["changed"] == e["changed"]
                assert (row["base_placed"],row["rollout_placed"],row["base_same_placed"]) == (
                    e["blind_placed"],e["rank_placed"],e["blind_unchanged_placed"])
                flip = keys[rep,1000-k,g]
                assert row["next_treated"] + flip["anti_treated"] == row["changed"]
                for label in ("next", "anti"):
                    total, same = row[label+"_placed"], row[label+"_same_placed"]
                    assert 0 <= total - same <= row["changed"]
                    assert 0 <= same <= e["n"] - row["changed"]
    if args.shards:
        write_rows(input_file, draws)
    summary = []
    for g in GROUPS:
        n = endpoint[g]["n"]
        outcomes = {k:[pair_outcomes(keys[j,k,g]) for j in range(1,REPS+1)] for k in GRID}
        for k in GRID:
            for key in outcomes[k][0]:
                summary.append(estimates(g,k,key,[r[key] for r in outcomes[k]],n))
        secants = [(hi["placement_level"]-lo["placement_level"])/0.1
                   for hi,lo in zip(outcomes[550],outcomes[450])]
        summary.append(estimates(g,"450_550","change_all_secant",secants,n))
        differences = [s-o["own_direct"] for s,o in zip(secants,outcomes[500])]
        summary.append(estimates(g,"450_550_minus_500","secant_minus_midpoint_direct",differences,n))
    write_rows(directory / "mixed-summary.csv", summary)
    print(f"Validated {len(draws)} rows ({REPS*2*len(GRID)} national mixed-policy assignments), four reporting groups.")
    for row in summary:
        if row["group"] == "entry_linked" and str(row["design"]) in {"500","450_550","450_550_minus_500"}:
            print(row["measure"], row["count"], row["count_mcse"], row["pp"], row["pp_mcse"])


if __name__ == "__main__":
    main()
