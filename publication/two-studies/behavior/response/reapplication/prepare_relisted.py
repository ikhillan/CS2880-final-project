"""Align explicitly re-listed complementary options to the frozen regular-round edge stream."""

import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

PERSON = "mrun"
GRADE = "cod_nivel"
SCHOOL = "rbd"
COURSE = "cod_curso"
RANK = "preferencia_postulante"
AUTO = "agregada_por_continuidad"
KEY = [PERSON, SCHOOL, COURSE]


def read_numeric(prefix, people, courses, edges):
    ids = pd.Index(people[PERSON])
    pg = pd.MultiIndex.from_frame(courses[[SCHOOL, COURSE]])
    with prefix.with_suffix(".bin").open("rb") as f:
        n, m, length = np.fromfile(f, "<i4", 3)
        indptr = np.fromfile(f, "<i4", n + 1)
        programs = np.fromfile(f, "<i4", length)
    assert (n, m, length) == (len(people), len(courses), len(edges))
    assert indptr[0] == 0 and indptr[-1] == length
    encoded = pg.get_indexer(pd.MultiIndex.from_frame(edges[[SCHOOL, COURSE]]))
    np.testing.assert_array_equal(encoded, programs)
    owners = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
    np.testing.assert_array_equal(owners, ids.get_indexer(edges[PERSON]))
    shown = {
        k: np.fromfile(str(prefix) + "-" + k + ".mask", dtype=np.uint8)
        for k in ("blind", "rank")
    }
    for mask in shown.values():
        assert len(mask) == length and mask.max() <= 1
    return owners, shown


def main(a):
    sys.path.insert(0, str(a.publication / "two-studies/behavior/field"))
    from round_link import read_file

    members = subprocess.check_output(
        ["bsdtar", "-tf", a.archive], text=True
    ).splitlines()

    def get(pre, cols):
        return read_file(a.archive, members, pre, cols)

    people = get("B1", [PERSON, GRADE]).sort_values(PERSON, ignore_index=True)
    ids = pd.Index(people[PERSON])
    courses = get("A1", [SCHOOL, COURSE])
    reg = get("C1", KEY + [GRADE, RANK, "prioridad_matriculado"]).sort_values(
        [PERSON, RANK], ignore_index=True
    )
    assert not reg.duplicated(KEY).any() and ids.is_unique
    cutoff = (
        reg[RANK]
        .where(reg.prioridad_matriculado.eq(1))
        .groupby(reg[PERSON])
        .transform("min")
    )
    eligible = reg.loc[cutoff.isna() | reg[RANK].le(cutoff)].copy()
    owners, shown = read_numeric(a.numeric, people, courses, eligible)

    c2 = get("C2", KEY + [RANK, GRADE, AUTO])
    assert not c2.duplicated(KEY).any() and not c2.duplicated([PERSON, RANK]).any()
    roster = get("B2", [PERSON, GRADE])
    assert roster[PERSON].is_unique
    flag = ids.isin(
        roster.loc[
            roster[GRADE].eq(roster[PERSON].map(people.set_index(PERSON)[GRADE])),
            PERSON,
        ]
    )
    obs = c2.loc[c2[AUTO].eq(0) & c2[PERSON].isin(ids[flag])].copy()
    assert (
        obs[GRADE].eq(obs[PERSON].map(people.set_index(PERSON)[GRADE])).all()
        and (obs[RANK] >= 1).all()
    )
    obs_keys = pd.MultiIndex.from_frame(obs[KEY])
    eligible_keys = pd.MultiIndex.from_frame(eligible[KEY])
    edge = eligible_keys.get_indexer(obs_keys)
    rank2 = np.zeros(len(eligible), dtype="<i4")
    existing = edge >= 0
    rank2[edge[existing]] = obs.loc[existing, RANK].astype(int)
    repeated = pd.MultiIndex.from_frame(reg[KEY]).get_indexer(obs_keys) >= 0
    new_school = ~pd.MultiIndex.from_frame(obs[[PERSON, SCHOOL]]).isin(
        pd.MultiIndex.from_frame(reg[[PERSON, SCHOOL]])
    )
    assert (repeated[existing]).all() and not (new_school & repeated).any()
    obs_owners = ids.get_indexer(obs[PERSON])

    def person_count(mask):
        return np.bincount(obs_owners[mask], minlength=len(ids)).astype(np.int32)

    def eligible_counts(mask):
        return np.bincount(owners[mask], minlength=len(ids)).astype(np.int32)

    # Private modeled inputs end here. Later records may be used to select/evaluate
    # historical strata; April enrollment never chooses an edge.
    a.private.mkdir(exist_ok=True, parents=True)
    rank2.tofile(a.private / f"{a.year}-relisted.i4")
    a2 = get("A2", [GRADE, SCHOOL, COURSE, "vacantes"])
    a2_keys = pd.MultiIndex.from_frame(a2[[SCHOOL, COURSE]])
    assert a2_keys.is_unique and not a2.vacantes.isna().any() and a2.vacantes.min() >= 0
    second_indices = a2_keys.get_indexer(
        pd.MultiIndex.from_frame(obs[[SCHOOL, COURSE]])
    )
    assert np.min(second_indices) >= 0
    np.testing.assert_array_equal(a2[GRADE].to_numpy()[second_indices], obs[GRADE])
    edge_order = np.lexsort((obs[RANK], obs_owners))
    np.column_stack((obs_owners, second_indices, obs[RANK], new_school))[
        edge_order
    ].astype("<i4").tofile(a.private / f"{a.year}-second-edges.i4")
    regular_indices = pd.MultiIndex.from_frame(courses[[SCHOOL, COURSE]]).get_indexer(
        a2_keys
    )
    np.column_stack((a2.vacantes, regular_indices, a2[SCHOOL])).astype("<i4").tofile(
        a.private / f"{a.year}-second-courses.i4"
    )
    d1 = (
        get(
            "D1",
            [
                PERSON,
                "respuesta_postulante",
                "respuesta_postulante_post_lista_espera",
                "rbd_admitido",
                "rbd_admitido_post_resp",
            ],
        )
        .set_index(PERSON)
        .reindex(ids)
    )

    def numeric(s):
        return pd.to_numeric(
            s.astype("string").str.strip().replace("", pd.NA), errors="raise"
        )

    no_initial = numeric(d1.rbd_admitido).isna().to_numpy()
    never = no_initial & numeric(d1.rbd_admitido_post_resp).isna().to_numpy()
    check = d1.respuesta_postulante.eq(6) & numeric(
        d1.respuesta_postulante_post_lista_espera
    ).eq(6).fillna(False)
    np.testing.assert_array_equal(never, check.to_numpy(dtype=bool))
    never.astype("u1").tofile(a.private / f"{a.year}-unoffered.u1")

    # Link official initial C2 offers and April schools only for descriptive counts.
    d2 = get("D2", [PERSON, "rbd_admitido", "cod_curso_admitido"]).set_index(PERSON)
    awarded = (
        (
            obs[SCHOOL].eq(obs[PERSON].map(numeric(d2.rbd_admitido)))
            & obs[COURSE].eq(obs[PERSON].map(numeric(d2.cod_curso_admitido)))
        )
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    with a.register.open("rb") as f:
        rbd = np.fromfile(f, "<i4", len(courses))
        april = np.fromfile(f, "<i4")
    np.testing.assert_array_equal(courses[SCHOOL], rbd)
    assert len(april) == len(ids)
    at_offer = awarded & (obs[SCHOOL].to_numpy() == april[obs_owners])
    blind_omits = (rank2 > 0) & ~shown["blind"].astype(bool)
    rank_omits = (rank2 > 0) & ~shown["rank"].astype(bool)
    edits = blind_omits | rank_omits
    hidden = np.zeros(len(obs), dtype=bool)
    hidden[existing] = edits[edge[existing]]
    # A school-choice menu can omit an old option only on reports with alternatives
    # outside its quota. To measure the opportunity to revisit it in C2, further
    # require at least one released A2 vacancy at that same course.
    regular_open = pd.MultiIndex.from_frame(eligible[[SCHOOL, COURSE]]).isin(
        a2_keys[a2.vacantes.gt(0).to_numpy()]
    )
    menu_omits = ~shown["blind"].astype(bool) | ~shown["rank"].astype(bool)
    return_edge = flag[owners]
    counts = dict(
        returner_with_omission=eligible_counts(return_edge & menu_omits),
        returner_with_open_omission=eligible_counts(
            return_edge & menu_omits & regular_open
        ),
        repeated_open_omission=eligible_counts(edits & regular_open),
        returned_same_grade=flag,
        named_any=person_count(np.ones(len(obs), bool)),
        repeated_eligible=person_count(existing),
        repeated_below_continuity=person_count(repeated & ~existing),
        outside_original_course=person_count(~repeated),
        outside_original_school=person_count(new_school),
        omitted_by_blind=eligible_counts(blind_omits),
        omitted_by_rank=eligible_counts(rank_omits),
        omitted_by_either=eligible_counts(edits),
        awarded_then_enrolled_at_omitted=person_count(hidden & at_offer),
        awarded_then_enrolled_at_new_school=person_count(new_school & at_offer),
    )
    # Save only row-level indicators to private scratch for gross-loss accounting.
    person_file = a.private / f"{a.year}-reapplication-counts.npz"
    np.savez_compressed(person_file, never=never, no_initial=no_initial, **counts)
    for name, mask in (("all", np.ones(len(ids), bool)), ("never_offered", never)):
        assert not np.any(counts["awarded_then_enrolled_at_omitted"] > 1)
        row = {"year": a.year, "group": name, "applicants": int(mask.sum())}
        for col, values in counts.items():
            row[col + "_pupils"] = int(np.count_nonzero(values[mask]))
            if col != "returned_same_grade":
                row[col + "_choices"] = int(values[mask].sum())
        yield row


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, required=True)
    for k in ("publication", "archive", "numeric", "private", "register", "out"):
        p.add_argument("--" + k, required=True, type=Path)
    args = p.parse_args()
    result = pd.DataFrame(main(args))
    result.to_csv(args.out, index=False)
    print(
        result[
            [
                "year",
                "group",
                "returned_same_grade_pupils",
                "omitted_by_either_pupils",
                "outside_original_school_pupils",
            ]
        ].to_string(index=False)
    )
