"""Score frozen school menus within strata defined by the initial offer and April school."""

import argparse
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from enrollment_target import edge_hit_by_person
from link_register import enrollment_register, numeric_col
from round_link import COURSE, GRADE, ID, RELEASES, SCHOOL, read_file


def cohort(year, archive, numeric, register, market_class):
    names = subprocess.check_output(["bsdtar", "-tf", archive], text=True).splitlines()
    original = (
        read_file(
            archive, names, "D1", [ID, GRADE, "rbd_admitido", "cod_curso_admitido"]
        )
        .set_index(ID)
        .sort_index()
    )
    supply = read_file(archive, names, "A1", [SCHOOL, COURSE])
    market = market_class(numeric / f"{year}.bin")
    assert original.index.is_unique
    np.testing.assert_array_equal(market.grade, original[GRADE])
    offer_keys = (
        original[["rbd_admitido", "cod_curso_admitido"]].apply(numeric_col).fillna(-1)
    )
    index = pd.MultiIndex.from_frame(supply[[SCHOOL, COURSE]])
    assert index.is_unique
    np.testing.assert_array_equal(
        index.get_indexer(pd.MultiIndex.from_frame(offer_keys)), market.off
    )
    observed, _ = enrollment_register(register, original.index, year + 1)
    april = observed.RBD.reindex(original.index).fillna(-1).to_numpy(dtype=int)
    linked = april >= 0
    offer_school = offer_keys.rbd_admitido.to_numpy(dtype=int)
    assert np.array_equal(market.off >= 0, offer_school >= 0)
    schools = supply[SCHOOL].to_numpy()[market.j]
    target_edges = schools == april[market.i]
    groups = {
        "enrolled_at_initial_offer": linked & (april == offer_school),
        "enrolled_elsewhere": linked & (offer_school >= 0) & (april != offer_school),
        "no_initial_offer": linked & (offer_school < 0),
    }
    np.testing.assert_array_equal(sum(v.astype(int) for v in groups.values()), linked)
    assert np.all(
        market.cols[market.j == market.ret[market.i]]
        == market.degree[market.i[market.j == market.ret[market.i]]]
    )
    masks = {"first": (market.cols < 5) | (market.j == market.ret[market.i])}
    for tag in ("blind", "rank"):
        mask = np.fromfile(numeric / f"{year}-{tag}.mask", np.uint8)
        assert mask.size == market.ne and np.isin(mask, [0, 1]).all()
        masks[tag] = mask.astype(bool)
    full = edge_hit_by_person(
        target_edges, np.ones(market.ne, bool), market.i, market.n
    )
    hits, placed = {}, {}
    for tag, mask in masks.items():
        np.testing.assert_array_equal(
            np.bincount(market.i[mask], minlength=market.n),
            np.minimum(market.degree, 5) + (market.ret >= 0),
        )
        hits[tag] = edge_hit_by_person(target_edges, mask, market.i, market.n)
        if tag == "first":
            placed[tag] = market.solve(mask) >= 0
        else:
            vector = np.fromfile(numeric / f"{year}-{tag}.partners", dtype="<i4")
            assert (
                vector.shape == (market.n,)
                and ((vector >= -1) & (vector < market.m)).all()
            )
            placed[tag] = vector >= 0
    rows = []
    for label, people in {**groups, "all_linked": linked}.items():
        row = {
            "year": year,
            "stratum": label,
            "people": int(people.sum()),
            "eligible_hits": int(full[people].sum()),
        }
        for tag in masks:
            row[f"{tag}_hits"] = int(hits[tag][people].sum())
            row[f"{tag}_placed"] = int(placed[tag][people].sum())
        for a, b in [("rank", "blind"), ("rank", "first")]:
            row[f"{a}_only_vs_{b}_hits"] = int((people & hits[a] & ~hits[b]).sum())
            row[f"{b}_only_vs_{a}_hits"] = int((people & hits[b] & ~hits[a]).sum())
        rows.append(row)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in (
        "publication",
        "early-archives",
        "late-archives",
        "early-numeric",
        "late-numeric",
        "enrollments",
        "out",
    ):
        p.add_argument("--" + arg, type=Path, required=True)
    args = p.parse_args()
    solver = args.publication / "research/policy/equity/transitions.py"
    spec = importlib.util.spec_from_file_location("equity_transitions", solver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for year, archive in RELEASES.items():
        source = args.early_archives if year < 2024 else args.late_archives
        numeric = args.early_numeric if year < 2024 else args.late_numeric
        rows.extend(
            cohort(
                year,
                source / archive,
                numeric,
                args.enrollments / f"matricula-{year + 1}.rar",
                module.Market,
            )
        )
        print(f"{year} completed", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
