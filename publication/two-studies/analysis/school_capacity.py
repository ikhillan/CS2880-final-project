"""Check the maximum feasible placement on each display with integral flows and cuts."""
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from school_referrals import load_module


def assess(market, stem, policy, flow):
    shown = np.fromfile(stem.parent / f"{stem.name}-{policy}.mask", np.uint8).astype(bool)
    partners = np.fromfile(stem.parent / f"{stem.name}-{policy}.partners", "<i4")
    adapter = SimpleNamespace(a=market.i, school=market.j, n=market.n, m=market.m)
    edge_grade = market.grade[market.i]
    used = np.zeros(market.m, bool)
    maximum = 0
    for grade in np.unique(edge_grade):
        ids = np.flatnonzero(shown & (edge_grade == grade))
        programs = np.unique(market.j[ids])
        assert not used[programs].any(), "Grades must have disjoint capacity pools"
        used[programs] = True
        assigned, witness = flow.match_vacancies(adapter, market.cap, ids, 2880781)
        assert len(assigned) == witness["cut"]
        maximum += len(assigned)
    placed = int((partners >= 0).sum())
    assert maximum >= placed
    return dict(year=stem.name, policy=policy, applicants=market.n, visible_capacity=maximum,
                placed=placed, shortfall=maximum - placed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reader = load_module(args.published / "research/policy/equity/transitions.py")
    flow = load_module(args.published / "research/policy/design/repair.py")
    rows = []
    for year in ("2022", "2023"):
        stem = args.numeric / year
        market = reader.Market(stem.with_suffix(".bin"))
        for policy in ("blind", "rank"):
            rows.append(assess(market, stem, policy, flow))
    with args.out.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
