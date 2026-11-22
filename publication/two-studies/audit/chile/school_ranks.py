"""Count rank changes under the two frozen five-slot policies in Chile's SAE.

Look up each replayed school course in the public C1 application file. Distinguish
a family-entered course from a fallback the system appended. Only aggregate
outcomes are written. Requires NumPy, pandas and bsdtar with RAR support.
"""
import argparse
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


RELEASES = {2022: "SAE_2022.rar", 2023: "SAE-2023.rar",
            2024: "SAE_2024.rar", 2025: "SAE_2025.rar"}


def read_public(archive, prefix, columns):
    members = subprocess.check_output(["bsdtar", "-tf", archive], text=True).splitlines()
    member, = [path for path in members if Path(path).name.startswith(prefix + "_") and path.endswith(".csv")]
    raw = subprocess.check_output(["bsdtar", "-xOf", archive, member])
    return pd.read_csv(io.BytesIO(raw), sep=";", encoding="utf-8-sig", usecols=columns)


def summarize(year, numeric, archive):
    key = ["rbd", "cod_curso"]
    catalog = pd.MultiIndex.from_frame(read_public(archive, "A1", key)[key])
    pupils = pd.Index(read_public(archive, "B1", ["mrun"]).mrun).sort_values()
    apps = read_public(archive, "C1", ["mrun", *key, "preferencia_postulante",
                           "agregada_por_continuidad", "prioridad_matriculado"])
    assert catalog.is_unique and pupils.is_unique
    assert not apps.duplicated(["mrun", *key]).any()
    assert not apps.duplicated(["mrun", "preferencia_postulante"]).any()
    entitlement = apps.loc[apps.prioridad_matriculado.eq(1)].set_index("mrun").preferencia_postulante
    cut = apps.mrun.map(entitlement)
    apps = apps.loc[cut.isna() | apps.preferencia_postulante.le(cut)].sort_values(
        ["mrun", "preferencia_postulante"])
    kid = pupils.get_indexer(apps.mrun)
    course = catalog.get_indexer(pd.MultiIndex.from_frame(apps[key]))
    declared_pos = apps.preferencia_postulante.to_numpy()
    admin = apps.agregada_por_continuidad.eq(1).to_numpy()
    n = len(pupils)
    assert (kid >= 0).all() and (course >= 0).all() and (declared_pos >= 1).all()
    degree = np.bincount(kid, minlength=n)
    offsets = np.r_[0, np.cumsum(degree)]
    position = 1 + np.arange(len(kid)) - offsets[kid]
    # Check where the official rank convention differs from the retained order.
    provided = read_public(archive, "D1", ["mrun", "rbd_admitido", "cod_curso_admitido"]).set_index("mrun")
    assert provided.index.is_unique and pupils.symmetric_difference(provided.index).empty
    provided = provided.loc[pupils, ["rbd_admitido","cod_curso_admitido"]].apply(pd.to_numeric, errors="coerce")
    official = catalog.get_indexer(pd.MultiIndex.from_frame(provided))
    assert np.array_equal(official >= 0, provided.rbd_admitido.ge(0))
    assert np.all(position[admin] == degree[kid[admin]])
    # Independent mapping check against the exact numeric DA input, including
    # row order and program IDs, not just its length.
    with (numeric / f"{year}.bin").open("rb") as source:
        read = lambda count: np.fromfile(source, count=count, dtype="<i4")
        assert tuple(read(3)) == (n, len(catalog), len(course))
        np.testing.assert_array_equal(read(n+1), offsets)
        np.testing.assert_array_equal(read(len(course)), course)
        source.seek(4 * (len(course) + len(catalog)), 1)
        np.testing.assert_array_equal(read(n), official)

    assigned, rank, appended, show, shown_historical = {}, {}, {}, {}, {}
    for policy in ["full", "blind", "rank"]:
        assigned[policy] = np.fromfile(numeric / f"{year}-{policy}.partners", dtype="<i4")
        raw = np.fromfile(numeric / f"{year}-{policy}.mask", dtype=np.uint8)
        assert len(raw) == len(kid) and np.isin(raw, [0,1]).all()
        show[policy] = raw.astype(bool)
        hits = (course == assigned[policy][kid])
        assert np.array_equal(np.bincount(kid[hits], minlength=n), assigned[policy] >= 0)
        assert show[policy][hits].all()
        rank[policy] = np.zeros(n, dtype=int)  # outside has no list rank
        rank[policy][kid[hits]] = declared_pos[hits]
        appended[policy] = np.zeros(n, dtype=bool)
        appended[policy][kid[hits]] = admin[hits]
        shown_historical[policy] = np.bincount(kid[(course == official[kid]) & show[policy]],minlength=n).astype(bool)
    assert show["full"].all()
    same_menu = ~np.bincount(kid[show["blind"] != show["rank"]],minlength=n).astype(bool)
    replay_agrees = assigned["full"] == official
    before, after = rank["blind"], rank["rank"]
    got_old, got_new = before > 0, after > 0
    both = got_old & got_new
    better = both & (after < before)
    worse = both & (after > before)
    gain = got_new & ~got_old
    loss = got_old & ~got_new
    same = assigned["blind"] == assigned["rank"]
    assert np.array_equal(gain | loss | better | worse, ~same)
    # Any rank-one offer must be family entered to count toward "reported first".
    groups = {"all": np.ones(n, bool), "same_menu": same_menu, "changed_menu": ~same_menu,
              "full_replay_agrees": replay_agrees, "full_replay_differs": ~replay_agrees}
    table = []
    for name, group in groups.items():
        def count(z):
            return int(np.sum(group & z))
        old_fallback, new_fallback = appended["blind"], appended["rank"]
        row = dict(year=year, group=name, applicants=int(group.sum()),
                   historical_coverage_delta=count(shown_historical["rank"])-count(shown_historical["blind"]),
                   placed_delta=count(got_new)-count(got_old), gain=count(gain), loss=count(loss),
                   improved_both=count(better), worsened_both=count(worse), same_outcome=count(same),
                   better_including_gains=count(better | gain), worse_including_losses=count(worse | loss),
                   improved_both_reported=count(better & ~old_fallback & ~new_fallback),
                   worsened_both_reported=count(worse & ~old_fallback & ~new_fallback),
                   same_menu_losses=count(loss & same_menu), same_menu_gains=count(gain & same_menu),
                   delta_appended=count(new_fallback)-count(old_fallback))
        assert row["applicants"] == row["same_outcome"]+row["better_including_gains"]+row["worse_including_losses"]
        table.append(row)
    curves = []
    for group, population in groups.items():
        for k in range(1, int(declared_pos.max()) + 1):
            vals = {}
            for policy in ["blind", "rank"]:
                in_top = (rank[policy] > 0) & (rank[policy] <= k) & population
                vals[policy] = int(in_top.sum())
                vals["submitted_"+policy] = int((in_top & ~appended[policy]).sum())
            curves.append(dict(year=year,group=group,k=k,**vals,delta=vals["rank"]-vals["blind"],
                               submitted_delta=vals["submitted_rank"]-vals["submitted_blind"]))
    meta = dict(year=year,applicants=n,eligible_edges=len(course),appended_edges=int(admin.sum()),
                applications_nonconsecutive_position=int((position!=declared_pos).sum()),
                maximum_position=int(declared_pos.max()),full_replay_matches=int(replay_agrees.sum()),
                retained_full_recipient_matches=int(((official >= 0) & replay_agrees).sum()))
    return table, curves, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["early_archives", "early_numeric", "late_archives", "late_numeric", "out"]:
        parser.add_argument("--" + name.replace("_","-"), type=Path, required=True)
    args = parser.parse_args()
    results, curves, metadata = [], [], []
    for year, filename in RELEASES.items():
        numeric = args.early_numeric if year <= 2023 else args.late_numeric
        archive = (args.early_archives if year <= 2023 else args.late_archives) / filename
        rows, thresholds, meta = summarize(year,numeric,archive)
        results.extend(rows)
        curves.extend(thresholds)
        metadata.append(meta)
        print(year, rows[0], [(r["k"],r["delta"],r["submitted_delta"]) for r in thresholds
                            if r["group"] == "all" and r["k"] in (1,3,5)], flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    for filename, rows in [("school-rank-changes.csv",results), ("school-rank-thresholds.csv",curves),
                           ("school-rank-input-checks.csv",metadata)]:
        pd.DataFrame(rows).to_csv(args.out / filename, index=False)


if __name__ == "__main__":
    main()
