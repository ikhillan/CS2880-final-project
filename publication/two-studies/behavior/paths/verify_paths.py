"""Reconcile the constructive path counts with the earlier menu and placement tables."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def rows(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def check(study, data, paper):
    keys = ["year"] if study == "school" else ["design", "seed", "borough", "month"]
    cases = rows(data / f"{study}-binding-certificates.csv")
    hist = Counter()
    for r in rows(data / f"{study}-binding-lengths.csv"):
        hist[tuple(r[k] for k in keys), r["direction"], int(r["steps"])] += int(
            r["sources"]
        )
    counts = Counter()
    for case in cases:
        market = tuple(case[k] for k in keys)
        group = case[keys[0]]
        for direction in ["loss", "gain"]:
            length = int(case[f"{direction}_longest_path"])
            by_depth = [hist[market, direction, d] for d in range(length + 1)]
            n = int(case[direction])
            assert sum(by_depth) == n and (not n or by_depth[-1] > 0)
            assert by_depth[0] == int(case[f"{direction}_self_witness"])
            assert sum(by_depth[1:]) == int(case[f"{direction}_other_witness"])
            same = int(case[f"{direction}_identical_row"])
            assert same <= sum(by_depth[1:]) and n <= int(case["potential_witnesses"])
            counts[group, direction, "all"] += n
            counts[group, direction, "same"] += same
    assert len(cases) == (4 if study == "school" else 480)
    if study == "school":
        for r in rows(paper / "behavior/chile/school-tail-summary.csv"):
            if r["year"] == "0" or r["response"] != "binding":
                continue
            for d in ["loss", "gain"]:
                key = "losses" if d == "loss" else "gains"
                assert counts[r["year"], d, "all"] == int(r[key])
                assert counts[r["year"], d, "same"] == int(r["same_order_" + key])
    else:
        expected = Counter()
        for r in rows(paper / "evidence/joint.csv"):
            if r["group"] != "all" or r["delta_placement"] == "0":
                continue
            d = "loss" if r["delta_placement"] == "-1" else "gain"
            expected[r["design"], d, "all"] += int(r["count"])
            if r["same_menu"] == "1":
                expected[r["design"], d, "same"] += int(r["count"])
        assert expected == counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path(__file__).parent)
    p.add_argument("--paper", type=Path, required=True)
    args = p.parse_args()
    for study in ["school", "housing"]:
        check(study, args.data, args.paper)
    receipt = json.loads((args.data / "mixed-profile-check.json").read_text())
    assert receipt["ordered_stable_comparisons"] == 9068610
    assert receipt["unequal_graph_pairs"] == 8867592
    print(
        "Verified binding loss and gain paths in four application cohorts and 480 rental markets."
    )


if __name__ == "__main__":
    main()
