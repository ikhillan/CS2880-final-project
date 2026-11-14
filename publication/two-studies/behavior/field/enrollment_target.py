"""Score both frozen menus on the school recorded in next April's enrollment register."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from round_link import ID, SCHOOL, COURSE, GRADE, RELEASES, read_file
from link_register import enrollment_register, numeric_col


def edge_hit_by_person(match, visible, owner, n):
    hit = np.zeros(n, dtype=bool)
    hit[owner[match & visible]] = True
    return hit


def cohort(year, archive, numeric, register_path):
    members = subprocess.check_output(
        ["bsdtar", "-tf", archive], text=True
    ).splitlines()
    d1 = read_file(archive, members, "D1", None).set_index(ID).sort_index()
    a1 = read_file(
        archive, members, "A1", [SCHOOL, COURSE, GRADE, "cod_ense", "cod_grado"]
    )
    with (numeric / f"{year}.bin").open("rb") as f:

        def read(count):
            return np.fromfile(f, dtype="<i4", count=count)

        n, m, e = map(int, read(3))
        at, program = read(n + 1), read(e)
        f.seek(4 * (e + m), 1)
        historical = read(n)
        f.seek(4 * n, 1)
        grade = read(n)
    assert (
        n == len(d1)
        and m == len(a1)
        and at[0] == 0
        and at[-1] == e
        and (np.diff(at) >= 0).all()
    )
    assert (program >= 0).all() and (program < m).all() and d1.index.is_unique
    np.testing.assert_array_equal(grade, d1[GRADE].to_numpy())
    for k in [
        "rbd_admitido",
        "cod_curso_admitido",
        "rbd_admitido_post_resp",
        "respuesta_postulante_post_lista_espera",
    ]:
        d1[k] = numeric_col(d1[k])
    orig = (
        d1[["rbd_admitido", "cod_curso_admitido"]]
        .set_axis([SCHOOL, COURSE], axis=1)
        .fillna(-1)
    )
    np.testing.assert_array_equal(
        historical,
        pd.MultiIndex.from_frame(a1[[SCHOOL, COURSE]]).get_indexer(
            pd.MultiIndex.from_frame(orig)
        ),
    )
    masks = {}
    for arm in ["full", "blind", "rank"]:
        raw = np.fromfile(numeric / f"{year}-{arm}.mask", dtype=np.uint8)
        assert len(raw) == e and np.isin(raw, [0, 1]).all()
        masks[arm] = raw.astype(bool)
    assert masks["full"].all()
    owner = np.repeat(np.arange(n), np.diff(at))
    np.testing.assert_array_equal(grade[owner], a1[GRADE].to_numpy()[program])
    register, _ = enrollment_register(register_path, d1.index, year + 1)
    aligned = register.reindex(d1.index)
    linked = aligned["RBD"].notna().to_numpy()
    flags = {
        "never_offered": (
            d1.respuesta_postulante.eq(6)
            & d1.respuesta_postulante_post_lista_espera.eq(6).fillna(False)
        ).to_numpy(dtype=bool)
    }
    assert np.array_equal(
        flags["never_offered"],
        d1.rbd_admitido.isna() & d1.rbd_admitido_post_resp.isna(),
    )
    flags["other_regular"] = ~flags["never_offered"]
    flags["all"] = np.ones(n, dtype=bool)
    school = a1[SCHOOL].to_numpy()
    edu = a1.cod_ense.to_numpy()
    classes = a1.cod_grado.to_numpy()
    # -1 denotes missing in all target vectors, never a school, program type, or grade.
    # Sentinel lookup of a target outside the official program range is always masked.
    off_idx = historical.clip(0)
    targets = {
        "enrollment": [
            aligned[k].fillna(-1).to_numpy(dtype=int)
            for k in ("RBD", "COD_ENSE", "COD_GRADO")
        ],
        "historical_offer": [
            np.where(historical >= 0, q[off_idx], -1) for q in (school, edu, classes)
        ],
    }
    rows = []
    for label, (rbd, kind, level) in targets.items():
        school_eq = school[program] == rbd[owner]
        type_grade_eq = (
            school_eq
            & (edu[program] == kind[owner])
            & (classes[program] == level[owner])
        )
        for grain, edges in [
            ("school", school_eq),
            ("school_type_grade", type_grade_eq),
        ]:
            hits = {
                arm: edge_hit_by_person(edges, visible, owner, n)
                for arm, visible in masks.items()
            }
            full, blind, rank = hits["full"], hits["blind"], hits["rank"]
            assert (full >= blind).all() and (full >= rank).all()
            for group, stratum in flags.items():
                for cohort_name, included in [
                    ("all_originals", stratum),
                    ("register_linked", stratum & linked),
                ]:

                    def count(v):
                        return int((included & v).sum())

                    gained = count(rank & ~blind)
                    lost = count(blind & ~rank)
                    assert gained - lost == count(rank) - count(blind)
                    rows.append(
                        dict(
                            year=year,
                            group=group,
                            population=cohort_name,
                            target=label,
                            resolution=grain,
                            applicants=int(included.sum()),
                            target_observed=count(rbd >= 0),
                            eligible_graph_hits=count(full),
                            blind_hits=count(blind),
                            rank_hits=count(rank),
                            gained_hits=gained,
                            lost_hits=lost,
                            net_hits=gained - lost,
                        )
                    )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "early-archives",
        "late-archives",
        "early-numeric",
        "late-numeric",
        "enrollments",
        "out",
    ]:
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    results = []
    for year, archive in RELEASES.items():
        root = args.early_archives if year < 2024 else args.late_archives
        numeric = args.early_numeric if year < 2024 else args.late_numeric
        rows = cohort(
            year,
            root / archive,
            numeric,
            args.enrollments / f"matricula-{year + 1}.rar",
        )
        print(
            [
                r
                for r in rows
                if r["group"] == "all"
                and r["population"] == "register_linked"
                and r["resolution"] == "school"
            ],
            flush=True,
        )
        results.extend(rows)
    pd.DataFrame(results).to_csv(args.out / "enrollment-target.csv", index=False)
