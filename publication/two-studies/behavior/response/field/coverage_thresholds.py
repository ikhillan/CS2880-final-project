"""Exact finite-cohort April-hit response frontier for one recovered school."""
import argparse
import csv
from fractions import Fraction
from pathlib import Path


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(data, output):
    derived = []
    for year in range(2022, 2026):
        baseline, = [r for r in rows(data / f"endpoints-{year}.csv") if r["scenario"] == "binding"]
        full = [r for r in rows(data / f"field-{year}.csv") if int(r["permille"]) == 1000]
        for variant in ("one_school", "least_congested_school"):
            a, = [r for r in full if r["scenario"] == variant + "_original"]
            b, = [r for r in full if r["scenario"] == variant + "_tail"]
            assert a["blind_april"] == b["blind_april"] and a["rank_april"] == b["rank_april"]
            gap = int(baseline["rank_april"]) - int(baseline["blind_april"])
            blind = int(a["blind_april"]) - int(baseline["blind_april"])
            informed = int(a["rank_april"]) - int(baseline["rank_april"])
            assert min(blind, informed) >= 0
            derived.append((str(year), variant, gap, blind, informed))
    summaries = []
    for variant in ("one_school", "least_congested_school"):
        sums = [sum(row[j] for row in derived if row[1] == variant) for j in (2,3,4)]
        summaries.append(("all", variant, *sums))
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["year","candidate","original_hit_gap","eligible_blind_additional_hits",
                         "eligible_informed_additional_hits","equal_response_min_gap",
                         "q_blind_break_even_when_q_informed_zero",
                         "q_informed_break_even_when_q_blind_one"])
        for year,variant,gap,blind,informed in derived+summaries:
            writer.writerow([year, variant, gap, blind, informed, min(gap, gap + informed - blind),
                             str(Fraction(gap, blind)), str(Fraction(blind-gap, informed))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    main(args.inputs,args.out)
