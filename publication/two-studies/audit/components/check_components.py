"""Assess agreement on entire connected components of the reported application graph.

Only aggregate counts are released. All eligible applications and observed official
assignment edges connect the graph, including courses with no modeled vacancies.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def counts(year, root):
    with (root / f"{year}.bin").open("rb") as handle:

        def read(n):
            return np.fromfile(handle, dtype="<i4", count=int(n))

        n, m, e = map(int, read(3))
        offsets = read(n + 1)
        programs = read(e)
        handle.seek(4 * (e + m), 1)
        official = read(n)
    owners = np.repeat(np.arange(n, dtype=np.int32), np.diff(offsets))
    assert programs.size == len(owners) == e
    receiving = np.flatnonzero(official >= 0)
    # Include actual offers even when they are outside the eligible report.
    source = np.concatenate((owners, receiving))
    target = np.concatenate((programs, official[receiving])) + n
    adjacency = coo_matrix(
        (np.ones(len(source), dtype="u1"), (source, target)), shape=(n + m, n + m)
    ).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    ids = labels[:n]
    assert np.array_equal(ids[owners], labels[n + programs])
    assert np.array_equal(ids[receiving], labels[n + official[receiving]])
    size = np.bincount(ids, minlength=labels.max() + 1)
    full, blind, rank = [
        np.fromfile(root / f"{year}-{arm}.partners", dtype="<i4")
        for arm in ("full", "blind", "rank")
    ]
    assert all(len(v) == n for v in (full, blind, rank))
    discrepancies = np.bincount(ids[full != official], minlength=len(size))
    agree = discrepancies[ids] == 0
    # Also verify coverage deltas in each identified subpopulation.
    hit = {}
    equal = []
    for arm in ("blind", "rank"):
        mask = np.fromfile(root / f"{year}-{arm}.mask", dtype="u1")
        assert mask.size == e and np.isin(mask, [0, 1]).all()
        seen = mask.astype(bool)
        hit[arm] = (
            np.bincount(owners[seen & (programs == official[owners])], minlength=n) > 0
        )
        equal.append(seen)
    changed = np.bincount(owners[equal[0] != equal[1]], minlength=n) > 0
    groups = {
        "all": np.ones(n, dtype=bool),
        "entire_component_agrees": agree,
        "disagreement_in_component": ~agree,
    }
    results = []
    for name, cohort in groups.items():
        subset = np.unique(ids[cohort])

        def count(selector):
            return int((cohort & selector).sum())

        results.append(
            dict(
                year=year,
                group=name,
                applicants=int(cohort.sum()),
                number_of_components=len(subset),
                largest_component=int(size[subset].max(initial=0)),
                changed_menus=count(changed),
                new_historical_hits=count(hit["rank"] & ~hit["blind"]),
                lost_historical_hits=count(hit["blind"] & ~hit["rank"]),
                replay_agreements=count(full == official),
                placed_blind=count(blind >= 0),
                placed_rank=count(rank >= 0),
                placement_gains=count((blind < 0) & (rank >= 0)),
                placement_losses=count((blind >= 0) & (rank < 0)),
            )
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("early-numeric", "late-numeric", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for year in range(2022, 2026):
        rows.extend(
            counts(year, args.early_numeric if year <= 2023 else args.late_numeric)
        )
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
