"""Verify published risk-warning effects against the two archived aggregate tables."""
import argparse
import csv
import hashlib
from decimal import Decimal
from pathlib import Path


def load(path, expected_hash):
    if hashlib.md5(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"File does not match the authors' archived copy: {path}")
    return [line.strip().removesuffix(r"\\").strip().split("&") for line in path.read_text().splitlines()]


def get(table, outcome, col, has_se=True):
    matches = [i for i, row in enumerate(table) if row[0].strip() == outcome]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {outcome}, found {len(matches)}")
    i = matches[0]
    estimate = Decimal(table[i][col].strip())
    se = Decimal(table[i+1][col].strip().strip("()")) if has_se else ""
    return estimate, se


def main(args):
    tab2 = load(args.table2, "0e2651cb3a0ec4086e83432b492fb960")
    tab3 = load(args.table3, "94d3eaa6e7440b07f68df7f03145784f")
    observations = []
    for outcome in ("Add any", "Add as last", "Add to middle", "Schools Added",
                    "Placed to preference", "Add any undersubscribed"):
        observations.append(("II", outcome, "local discontinuity", *get(tab2, outcome, 1), 5862803))
    observations.append(("II", "Schools Added", "IV per induced school adder", *get(tab2, "Schools Added", 2), 5862803))
    below, _ = get(tab3, "Add any", -1, False)
    jump, jump_se = get(tab3, "Add any", 1)
    assert (jump, jump_se) == get(tab2, "Add any", 1) == (Decimal(".216"), Decimal(".010"))
    assert below + jump == Decimal(".415")
    observations.extend([
        ("III", "Add any", "fitted below-threshold mean", below, "", 5862809),
        ("III", "Add any", "derived above-threshold level", below + jump, "", 5862809),
    ])
    with args.out.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["published_table","outcome","estimand","value","reported_se","authors_dataverse_file_id"])
        writer.writerows(observations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("table2", "table3", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    main(parser.parse_args())
