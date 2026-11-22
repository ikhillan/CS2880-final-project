"""Link pooled-DA strata and next-April enrollment to the two actual SAE rounds."""

import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from round_link import APPENDED, COURSE, GRADE, ID, SCHOOL, RELEASES, member, read_file

ENROLL_URLS = {
    2023: "2023/10/Matricula-por-estudiante-2023.rar",
    2024: "2024/11/Matricula-por-estudiante-2024.rar",
    2025: "2025/10/Matricula-por-estudiante-2025.rar",
    2026: "2026/10/Matricula-Oficial-2026.rar",
}


def enrollment_register(path, ids, year):
    members = subprocess.check_output(["bsdtar", "-tf", path], text=True).splitlines()
    (csv_member,) = [f for f in members if f.lower().endswith(".csv")]
    keep = ["AGNO", "MRUN", "RBD", "COD_ENSE", "COD_GRADO"]
    subset = []
    count = 0
    with subprocess.Popen(
        ["bsdtar", "-xOf", path, csv_member], stdout=subprocess.PIPE
    ) as proc:
        for chunk in pd.read_csv(
            proc.stdout, sep=";", usecols=keep, encoding="utf-8-sig", chunksize=100000
        ):
            assert (chunk.AGNO == year).all()
            count += len(chunk)
            subset.append(chunk.loc[chunk.MRUN.isin(ids), keep])
        if proc.wait():
            raise RuntimeError(f"Failed to decompress enrollment file {path}")
    linked = pd.concat(subset, ignore_index=True).set_index("MRUN").sort_index()
    assert linked.index.is_unique and not linked.isna().any().any()
    return linked, count


def policy_groups(numeric, year, regular, courses):
    # The original binary stores applicant records sorted by numeric MRUN, and
    # programs in A1 row order. Check both historical offer keys and all grades.
    with (numeric / f"{year}.bin").open("rb") as f:
        n, m, e = map(int, np.fromfile(f, dtype="<i4", count=3))
        assert n == len(regular) and m == len(courses)
        f.seek(4 * (n + 1 + e + e + m), 1)
        official = np.fromfile(f, dtype="<i4", count=n)
        f.seek(4 * n, 1)
        grade = np.fromfile(f, dtype="<i4", count=n)
    np.testing.assert_array_equal(grade, regular[GRADE])
    raw_keys = (
        regular[["rbd_admitido", "cod_curso_admitido"]]
        .set_axis([SCHOOL, COURSE], axis=1)
        .fillna(-1)
    )
    program_idx = pd.MultiIndex.from_frame(courses[[SCHOOL, COURSE]])
    assert program_idx.is_unique
    np.testing.assert_array_equal(
        official, program_idx.get_indexer(pd.MultiIndex.from_frame(raw_keys))
    )
    partners = [
        np.fromfile(numeric / f"{year}-{arm}.partners", dtype="<i4")
        for arm in ("blind", "rank")
    ]
    for vector in partners:
        assert len(vector) == n and np.logical_and(vector >= -1, vector < m).all()
    blind, rank = [x >= 0 for x in partners]
    return np.select(
        [blind & rank, blind & ~rank, ~blind & rank],
        ["both_placed", "lost", "gained"],
        default="both_unplaced",
    )


def numeric_col(s):
    stripped = s.astype("string").str.strip().replace("", pd.NA)
    return pd.to_numeric(stripped, errors="raise").astype("Int64")


def assess(sae, numeric, enroll, year, target):
    members = subprocess.check_output(["bsdtar", "-tf", sae], text=True).splitlines()

    def get(pre, cols):
        return read_file(sae, members, pre, cols)

    d1 = get("D1", None).set_index(ID).sort_index()
    for k in [
        "rbd_admitido",
        "cod_curso_admitido",
        "rbd_admitido_post_resp",
        "respuesta_postulante_post_lista_espera",
    ]:
        d1[k] = numeric_col(d1[k])
    d2 = get("D2", None).set_index(ID)
    for k in ["rbd_admitido", "cod_curso_admitido"]:
        d2[k] = numeric_col(d2[k])
    prior = get("C1", [ID, SCHOOL])
    later = get("C2", [ID, SCHOOL, COURSE, APPENDED])
    later = later.loc[later[APPENDED].eq(0)].copy()
    later["new"] = ~member(later, [ID, SCHOOL], prior)
    later["exact"] = later[SCHOOL].eq(later[ID].map(d2.rbd_admitido)) & later[
        COURSE
    ].eq(later[ID].map(d2.cod_curso_admitido))
    later["exact_new"] = later["exact"] & later["new"]
    a1 = get("A1", [SCHOOL, COURSE]) if numeric is not None else None
    a2 = get("A2", [GRADE, SCHOOL, COURSE, "cod_ense", "cod_grado"])
    a2 = a2.rename(columns={SCHOOL: "rbd_admitido", COURSE: "cod_curso_admitido"})
    d2 = (
        d2.reset_index()
        .merge(
            a2,
            on=[GRADE, "rbd_admitido", "cod_curso_admitido"],
            how="left",
            validate="many_to_one",
        )
        .set_index(ID)
    )
    assert d2.loc[d2.rbd_admitido.notna(), "cod_ense"].notna().all()
    tally = pd.DataFrame(index=d1.index)
    tally["regular"] = 1
    tally["returned_any_grade"] = tally.index.isin(d2.index)
    same_grade = d2[GRADE].reindex(tally.index).eq(d1[GRADE])
    tally["returned_same_grade"] = same_grade
    tally["any_new_school"] = same_grade & tally.index.isin(later.loc[later["new"], ID])
    offer = same_grade & tally.index.isin(later.loc[later.exact_new.fillna(False), ID])
    tally["offered_explicit_new_school"] = offer
    register, register_rows = enrollment_register(enroll, d1.index, year + 1)
    linked = tally.index.isin(register.index)
    tally["register_any_original_applicant"] = linked
    tally["register_any_among_new_offers"] = offer & linked
    school = (
        register.RBD.reindex(tally.index)
        .eq(d2.rbd_admitido.reindex(tally.index))
        .fillna(False)
    )
    grade = (
        register.COD_GRADO.reindex(tally.index)
        .eq(d2.cod_grado.reindex(tally.index))
        .fillna(False)
    )
    type_ = (
        register.COD_ENSE.reindex(tally.index)
        .eq(d2.cod_ense.reindex(tally.index))
        .fillna(False)
    )
    tally["registered_in_new_offered_school"] = offer & school
    tally["registered_in_new_offered_school_type_grade"] = (
        offer & school & grade & type_
    )
    never = d1.respuesta_postulante.eq(
        6
    ) & d1.respuesta_postulante_post_lista_espera.eq(6).fillna(False)
    initial_none = d1.rbd_admitido.isna()
    assert np.array_equal(never, initial_none & d1.rbd_admitido_post_resp.isna())
    change = policy_groups(numeric, year, d1, a1) if numeric is not None else None
    strata = {
        "all": np.ones(len(d1), dtype=bool),
        "initial_no_offer": initial_none,
        "initial_offer": ~initial_none,
        "never_offered": never,
        "waitlist_offer_after_no_initial": initial_none & ~never,
    }
    rows = []
    for stratum, keep in strata.items():
        group = tally[keep].copy()
        overall = group.sum().astype("int64").to_dict()
        rows.append(
            {
                "year": year,
                "historical_stratum": stratum,
                "menu_transition": "all",
                **overall,
            }
        )
        if change is not None:
            grouped = group.groupby(change[keep]).sum().astype("int64")
            assert grouped.sum().to_dict() == overall
            for status, s in grouped.iterrows():
                rows.append(
                    {
                        "year": year,
                        "historical_stratum": stratum,
                        "menu_transition": status,
                        **s.to_dict(),
                    }
                )
        if stratum == "never_offered":
            for here, there in [
                ("regular", "regular"),
                ("returned_same_grade", "returns_same_grade"),
                ("any_new_school", "any_new_school"),
                ("offered_explicit_new_school", "offered_second_explicit_new_school"),
            ]:
                assert overall[here] == target[there], (
                    year,
                    here,
                    overall[here],
                    target[there],
                )
    # Positive linkage check using initial offers explicitly accepted without waiting.
    accepted = d1.respuesta_postulante.eq(1)
    same_offered_rbd = register.RBD.reindex(d1.index).eq(d1.rbd_admitido).fillna(False)
    meta = {
        "initial_accepted": int(accepted.sum()),
        "initial_accepted_register_link": int((accepted & linked).sum()),
        "initial_accepted_same_school_in_register": int(
            (accepted & same_offered_rbd).sum()
        ),
        "application_year": year,
        "entry_year": year + 1,
        "archive_sha256": hashlib.sha256(enroll.read_bytes()).hexdigest(),
        "archive_bytes": enroll.stat().st_size,
        "source": "https://datosabiertos.mineduc.cl/wp-content/uploads/"
        + ENROLL_URLS[year + 1],
        "register_rows": register_rows,
        "original_ids": len(d1),
        "linked_ids": len(register),
    }
    return rows, meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in [
        "early-archives",
        "late-archives",
        "enrollments",
        "out",
    ]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--early-numeric", type=Path)
    p.add_argument("--late-numeric", type=Path)
    args = p.parse_args()
    if bool(args.early_numeric) != bool(args.late_numeric):
        p.error("Provide both numeric folders for the optional menu comparison")
    out = []
    inputs = []
    target = (
        pd.read_csv(args.out / "round-link-counts.csv")
        .query("group=='regular_never_offered'")
        .set_index("year")
    )
    for yr, f in RELEASES.items():
        root = args.early_archives if yr < 2024 else args.late_archives
        numeric = args.early_numeric if yr < 2024 else args.late_numeric
        rows, meta = assess(
            root / f,
            numeric,
            args.enrollments / f"matricula-{yr + 1}.rar",
            yr,
            target.loc[yr],
        )
        out.extend(rows)
        inputs.append(meta)
        print(
            next(
                x
                for x in rows
                if x["historical_stratum"] == "never_offered"
                and x["menu_transition"] == "all"
            ),
            flush=True,
        )
    results = pd.DataFrame(out)
    results.loc[results.menu_transition.eq("all")].to_csv(
        args.out / "round-link-register.csv", index=False
    )
    if args.early_numeric:
        results.to_csv(args.out / "round-link-policy.csv", index=False)
    pd.DataFrame(inputs).to_csv(args.out / "register-inputs.csv", index=False)
