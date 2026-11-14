"""Verify missing-enrollment bounds by comparing actual school sets pupil by pupil."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from check_round_link import records
from link_register import enrollment_register
from round_link import RELEASES


def check(year, archive, numeric, register, published):
    pupils = sorted(int(r["mrun"]) for r in records(archive, "D1"))
    schools = np.array([int(r["rbd"]) for r in records(archive, "A1")])
    with (numeric / f"{year}.bin").open("rb") as f:
        n, m, e = np.fromfile(f, "<i4", count=3)
        offsets = np.fromfile(f, "<i4", count=int(n) + 1)
        programs = np.fromfile(f, "<i4", count=e)
    assert len(set(pupils)) == n and len(schools) == m and offsets[-1] == e
    edges = schools[programs]
    masks = []
    for arm in ("blind", "rank"):
        mask = np.fromfile(numeric / f"{year}-{arm}.mask", np.uint8)
        assert len(mask) == e and np.isin(mask, [0, 1]).all()
        masks.append(mask.astype(bool))
    enrollees, _ = enrollment_register(register, pd.Index(pupils), year + 1)
    attended = enrollees.RBD.to_dict()
    down = up = linked_delta = 0
    for pos, kid in enumerate(pupils):
        row = slice(offsets[pos], offsets[pos + 1])
        school = attended.get(kid)
        # Repeated courses at the same school collapse in each person's own set.
        blind, rank = (set(edges[row][visible[row]]) for visible in masks)
        if school is None:
            down += bool(blind - rank)
            up += bool(rank - blind)
        else:
            linked_delta += int(school in rank) - int(school in blind)
    checked = dict(
        year=year,
        unlinked_blind_exclusive=down,
        unlinked_rank_exclusive=up,
        lower_gain=linked_delta - down,
        upper_gain=linked_delta + up,
        all_applicants=len(pupils),
        linked=len(attended),
    )
    for name, value in checked.items():
        if value != int(published[name]):
            raise ValueError((year, name, value, published[name]))
    if linked_delta != int(published["rank"]) - int(published["blind"]):
        raise ValueError((year, "linked coverage difference"))
    return {**checked, "linked_net_gain": linked_delta}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "early-archives",
        "late-archives",
        "early-numeric",
        "late-numeric",
        "enrollments",
        "target",
        "out",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    with args.target.open() as f:
        published = {int(row["year"]): row for row in csv.DictReader(f)}
    checks = []
    for year, archive in RELEASES.items():
        source = args.early_archives if year < 2024 else args.late_archives
        numeric = args.early_numeric if year < 2024 else args.late_numeric
        checks.append(
            check(
                year,
                source / archive,
                numeric,
                args.enrollments / f"matricula-{year + 1}.rar",
                published[year],
            )
        )
    with args.out.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
