"""Compare observed C1 ticket adjacency with candidate F1 family definitions."""

import argparse
import math
from pathlib import Path

import pandas as pd

from lottery_architecture import read_table


def family_ids(links):
    parent = {}

    def root(person):
        parent.setdefault(person, person)
        while parent[person] != person:
            parent[person] = parent[parent[person]]
            person = parent[person]
        return person

    for first, second in links[["mrun_1", "mrun_2"]].itertuples(index=False):
        first, second = root(first), root(second)
        parent[second] = first
    return {person: root(person) for person in parent}


def check(path, year):
    links = read_table(
        path, "F1", {"mrun_1", "mrun_2", "postula_en_bloque", "es_hermano"}
    )
    applicants = read_table(
        path, "C1", {"mrun", "rbd", "cod_curso", "loteria_original"}
    )
    course = ["rbd", "cod_curso"]
    assert applicants.loteria_original.notna().all()
    assert not applicants.duplicated(course + ["loteria_original"]).any()
    assert not applicants.duplicated(course + ["mrun"]).any()
    classes = applicants.groupby(course)
    sizes = classes.size().rename("total")
    applicants["position"] = classes.loteria_original.rank(method="first")
    for definition, relations in [
        ("all_F1", links),
        ("sibling", links.loc[links.es_hermano == 1]),
        ("opted_block", links.loc[links.postula_en_bloque == 1]),
        (
            "sibling_and_opted_block",
            links.loc[(links.es_hermano == 1) & (links.postula_en_bloque == 1)],
        ),
    ]:
        ids = family_ids(relations)
        members = applicants[applicants.mrun.isin(ids)].copy()
        members["family"] = members.mrun.map(ids)
        groups = members.groupby(course + ["family"]).position.agg(
            ["min", "max", "size"]
        )
        groups = groups.join(sizes, on=course)
        groups = groups.loc[(groups["size"] >= 2) & (groups["total"] > groups["size"])]
        contiguous = groups["max"] - groups["min"] + 1 == groups["size"]
        # Under an individual lottery, the family occupies a uniform k-subset of N ranks.
        expected = sum(
            (n - k + 1) / math.comb(n, k)
            for n, k in zip(groups["total"], groups["size"])
        )
        yield {
            "year": year,
            "definition": definition,
            "nontrivial_within_course_f1_groups": len(groups),
            "consecutive_in_recorded_order": int(contiguous.sum()),
            "nonconsecutive": int((~contiguous).sum()),
            "expected_consecutive_uniform_individual": expected,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", nargs=3, type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        row
        for year, path in zip((2022, 2023, 2024), args.archives)
        for row in check(path, year)
    ]
    results = pd.DataFrame(rows)
    results.to_csv(args.out, index=False)
    print(results.to_string(index=False))
