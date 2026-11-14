"""Trace binding-menu placement changes to people whose assigned options reverse."""

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def pair_holders(old, new):
    """Pair different holders at each program in ascending applicant order."""

    def holders(assignments):
        ids = np.flatnonzero((old != new) & (assignments >= 0))
        rows = pd.DataFrame({"person": ids, "program": assignments[ids]})
        rows.sort_values(["program", "person"], inplace=True)
        rows["slot"] = rows.groupby("program").cumcount()
        return rows

    joint = holders(old).merge(
        holders(new),
        on=["program", "slot"],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    nxt = np.full(old.size, -1, dtype=int)
    nxt[joint.person_old] = joint.person_new
    return nxt


def trace(
    old,
    new,
    old_priority,
    new_priority,
    prefer_old_at_old,
    prefer_new_at_new,
    changed_row,
):
    """Follow a source's old seat while its departing holder still wants it."""
    starts = np.flatnonzero((old >= 0) & (new < 0))
    reverse = prefer_old_at_old & prefer_new_at_new
    nxt = pair_holders(old, new)
    visited = set()
    endpoints = set()
    counts = Counter()
    for source in starts:
        i = source
        steps = 0
        while True:
            assert i not in visited and prefer_old_at_old[i]
            visited.add(i)
            if reverse[i]:
                # Original lists in the binding policies have identical relative
                # orders on any shared options, so a witness must have an edit.
                assert changed_row[i] and i not in endpoints
                endpoints.add(i)
                counts[steps] += 1
                break
            h = nxt[i]
            assert h >= 0 and old[i] == new[h] and new_priority[h] < old_priority[i]
            i = h
            steps += 1
    assert len(starts) == len(endpoints)
    return endpoints, counts


def certificate(
    old, new, priority_a, priority_b, score_aa, score_ab, score_ba, score_bb, changed
):
    """score_ab is the rank under A of the program received under B (smaller is better)."""
    prefer_a = score_aa < score_ab
    prefer_b = score_bb < score_ba
    witness = prefer_a & prefer_b
    assert np.all(~witness | changed)
    counts = {
        "applicants": old.size,
        "changed_menus": changed.sum(),
        "potential_witnesses": witness.sum(),
        "unplaced_both": ((old < 0) & (new < 0)).sum(),
    }
    history = []
    ends = []
    for tag, x, y, px, py, choice_x, choice_y in [
        ("loss", old, new, priority_a, priority_b, prefer_a, prefer_b),
        ("gain", new, old, priority_b, priority_a, prefer_b, prefer_a),
    ]:
        sources = (x >= 0) & (y < 0)
        stop, lengths = trace(x, y, px, py, choice_x, choice_y, changed)
        counts[tag] = int(sources.sum())
        counts[f"{tag}_identical_row"] = int((sources & ~changed).sum())
        counts[f"{tag}_self_witness"] = lengths[0]
        counts[f"{tag}_other_witness"] = len(stop) - lengths[0]
        counts[f"{tag}_longest_path"] = max(lengths, default=0)
        assert counts[f"{tag}_other_witness"] >= counts[f"{tag}_identical_row"]
        ends.append(stop)
        history.extend(
            dict(direction=tag, steps=k, sources=v) for k, v in sorted(lengths.items())
        )
    counts["shared_endpoints"] = len(ends[0] & ends[1])
    return counts, history


def school(repo, early, late):
    sys.path.insert(0, str(repo / "publication/two-studies/analysis"))
    from paired_school import import_market

    model = import_market(repo / "publication")
    levels = []
    distribution = []
    for year in range(2022, 2026):
        folder = early if year < 2024 else late
        market = model(folder / f"{year}.bin")
        a, b = [
            np.fromfile(folder / f"{year}-{policy}.mask", np.uint8).astype(bool)
            for policy in ("blind", "rank")
        ]
        old, new = [
            np.fromfile(folder / f"{year}-{policy}.partners", "<i4")
            for policy in ("blind", "rank")
        ]

        def assignment(target):
            # Recover the original relative rank from the applicant's global edge index.
            edge = np.full(market.n, -1, dtype=int)
            present = target[market.i] == market.j
            assert present.sum() == (target >= 0).sum()
            edge[market.i[present]] = np.flatnonzero(present)
            return edge

        ea, eb = assignment(old), assignment(new)
        assert (a[ea[ea >= 0]]).all() and (b[eb[eb >= 0]]).all()

        def score(mask, edges):
            safe = np.maximum(edges, 0)
            return np.where(
                edges < 0, market.ne, np.where(mask[safe], edges, market.ne + 1)
            )

        changed = np.zeros(market.n, dtype=bool)
        changed[market.i[a != b]] = True
        result, paths = certificate(
            old,
            new,
            market.pr[np.maximum(ea, 0)],
            market.pr[np.maximum(eb, 0)],
            score(a, ea),
            score(a, eb),
            score(b, ea),
            score(b, eb),
            changed,
        )
        levels.append(dict(year=year, **result))
        distribution.extend(dict(year=year, **row) for row in paths)
    return levels, distribution


def housing(repo):
    sys.path.insert(0, str(repo))
    from experiments.config import expand_labeled, load_config
    from experiments.pilot_outcomes import (
        CONFIG,
        DEFAULT_CACHE,
        load_pilot_cache,
        pilot_displays,
    )
    from hiddenoptions.markets import Catalog, build_market, InsufficientSupply
    from hiddenoptions.matching import deferred_acceptance

    catalog = Catalog()
    cache = load_pilot_cache(DEFAULT_CACHE)
    levels, distribution = [], []
    for spec, design in expand_labeled(load_config(CONFIG)):
        try:
            market = build_market(spec, catalog)
        except InsufficientSupply:
            assert spec.borough == "Staten Island"
            continue
        inst = market.inst
        a, b = pilot_displays(market, cache)
        old, _ = deferred_acceptance(inst, a)
        new, _ = deferred_acceptance(inst, b)
        row = np.arange(old.size)
        assert (
            a[row[old >= 0], old[old >= 0]].all()
            and b[row[new >= 0], new[new >= 0]].all()
        )

        def score(mask, targets):
            j = np.maximum(targets, 0)
            outside = int(inst.rank_b.max()) + 1
            return np.where(
                targets < 0,
                outside,
                np.where(mask[row, j], inst.rank_b[row, j], outside + 1),
            )

        result, paths = certificate(
            old,
            new,
            inst.rank_s[row, np.maximum(old, 0)],
            inst.rank_s[row, np.maximum(new, 0)],
            score(a, old),
            score(a, new),
            score(b, old),
            score(b, new),
            np.any(a != b, axis=1),
        )
        key = dict(
            design=design, seed=spec.seed, borough=spec.borough, month=spec.month
        )
        levels.append(dict(**key, **result))
        distribution.extend(dict(**key, **entry) for entry in paths)
    assert len(levels) == 480
    return levels, distribution


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("study", choices=["school", "housing"])
    for name in ("repo", "early", "late", "out"):
        cli.add_argument("--" + name, type=Path, required=name in ("repo", "out"))
    args = cli.parse_args()
    if args.study == "school":
        if args.early is None or args.late is None:
            cli.error("school requires --early and --late")
        counts, paths = school(args.repo.resolve(), args.early, args.late)
    else:
        counts, paths = housing(args.repo.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(counts).to_csv(
        args.out / f"{args.study}-binding-certificates.csv", index=False
    )
    pd.DataFrame(paths).to_csv(
        args.out / f"{args.study}-binding-lengths.csv", index=False
    )


if __name__ == "__main__":
    main()
