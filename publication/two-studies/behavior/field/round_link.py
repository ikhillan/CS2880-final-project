"""Link the regular and complementary admissions rounds, emitting aggregate counts only."""

import argparse
import hashlib
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

RELEASES = {
    2022: "SAE_2022.rar",
    2023: "SAE-2023.rar",
    2024: "SAE_2024.rar",
    2025: "SAE_2025.rar",
}
ID = "mrun"
GRADE = "cod_nivel"
SCHOOL = "rbd"
COURSE = "cod_curso"
RANK = "preferencia_postulante"
APPENDED = "agregada_por_continuidad"


def read_file(path, members, prefix, cols):
    (name,) = [
        name
        for name in members
        if Path(name).name.startswith(prefix + "_") and name.endswith(".csv")
    ]
    data = subprocess.check_output(["bsdtar", "-xOf", path, name])
    return pd.read_csv(
        io.BytesIO(data), encoding="utf-8-sig", sep=";", usecols=cols, low_memory=False
    )


def member(df, cols, ref):
    """Exact set membership for tuples of keys, allowing multiple courses within a school."""
    return pd.MultiIndex.from_frame(df[cols]).isin(
        pd.MultiIndex.from_frame(ref[cols].drop_duplicates())
    )


def cohort(path, year):
    members = subprocess.check_output(["bsdtar", "-tf", path], text=True).splitlines()

    def get(pre, cols):
        return read_file(path, members, pre, cols)

    regular = get("D1", None).set_index(ID).sort_index()
    second = get("D2", None).set_index(ID).sort_index()
    b1 = get("B1", [ID, GRADE]).set_index(ID).sort_index()
    b2 = get("B2", [ID, GRADE]).set_index(ID).sort_index()
    assert regular.index.is_unique and second.index.is_unique
    pd.testing.assert_frame_equal(b1, regular[[GRADE]])
    pd.testing.assert_frame_equal(b2, second[[GRADE]])
    original = get("C1", [ID, GRADE, SCHOOL, COURSE, RANK, APPENDED])
    later = get("C2", [ID, GRADE, SCHOOL, COURSE, RANK, APPENDED])
    for apps, roster in [(original, regular), (later, second)]:
        assert pd.Index(apps[ID].unique()).sort_values().equals(roster.index)
        assert (
            not apps.duplicated([ID, RANK]).any()
            and not apps.duplicated([ID, SCHOOL, COURSE]).any()
        )
        assert (apps[ID].map(roster[GRADE]) == apps[GRADE]).all()
    of1 = get("A1", [GRADE, SCHOOL, COURSE, "vacantes"])
    of2 = get("A2", [GRADE, SCHOOL, COURSE, "vacantes"])
    for supply, apps in [(of1, original), (of2, later)]:
        assert not supply.duplicated([GRADE, SCHOOL, COURSE]).any()
        assert member(apps, [GRADE, SCHOOL, COURSE], supply).all()
    for data, cols in [
        (
            regular,
            [
                "rbd_admitido",
                "rbd_admitido_post_resp",
                "respuesta_postulante_post_lista_espera",
            ],
        ),
        (second, ["rbd_admitido", "cod_curso_admitido"]),
    ]:
        for col in cols:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.strip().replace("", None), errors="raise"
            ).astype("Int64")
    eligible = regular.respuesta_postulante.eq(
        6
    ) & regular.respuesta_postulante_post_lista_espera.eq(6).fillna(False)
    assert np.array_equal(
        eligible.to_numpy(dtype=bool),
        (
            regular.rbd_admitido.isna() & regular.rbd_admitido_post_resp.isna()
        ).to_numpy(),
    )
    later = later[later[APPENDED] == 0].copy()
    later["new_school"] = ~member(later, [ID, SCHOOL], original)
    later["new_course"] = ~member(later, [ID, SCHOOL, COURSE], original)
    later["in_prior_supply"] = member(
        later, [GRADE, SCHOOL, COURSE], of1[of1.vacantes > 0]
    )
    later["in_current_supply"] = member(
        later, [GRADE, SCHOOL, COURSE], of2[of2.vacantes > 0]
    )
    later["first_new"] = later["new_school"] & later[RANK].eq(1)
    later["new_prior_positive"] = later.new_school & later.in_prior_supply
    later["new_both_positive"] = later.new_prior_positive & later.in_current_supply
    for name, series in [
        ("got_school", "rbd_admitido"),
        ("got_course", "cod_curso_admitido"),
    ]:
        later[name] = (
            later[ID]
            .map(second[series])
            .eq(later[SCHOOL if name == "got_school" else COURSE])
            .fillna(False)
        )
    later["got_exact"] = later.got_school & later.got_course
    later["got_new"] = later.got_exact & later.new_school
    later["got_new_prior_positive"] = later.got_new & later.in_prior_supply
    later["got_new_both_positive"] = (
        later.got_new_prior_positive & later.in_current_supply
    )
    later["got_new_first"] = later.got_new & later[RANK].eq(1)
    obs = second.copy()
    obs["returned_same_grade"] = obs[GRADE].eq(obs.index.map(regular[GRADE]))
    for key in [
        "new_school",
        "new_course",
        "new_prior_positive",
        "new_both_positive",
        "first_new",
        "got_exact",
        "got_new",
        "got_new_first",
        "got_new_prior_positive",
        "got_new_both_positive",
    ]:
        obs[key] = later.groupby(ID)[key].any().reindex(obs.index, fill_value=False)
    byid = later.groupby(ID).size()
    obs["has_manual"] = byid.reindex(obs.index, fill_value=0) > 0
    prev_open = original[
        original[APPENDED].eq(0)
        & member(original, [GRADE, SCHOOL, COURSE], of2[of2.vacantes > 0])
    ]
    obs["old_option_open"] = obs.index.isin(prev_open[ID])
    obs["new_despite_old_open"] = obs.new_school & obs.old_option_open
    obs["new_both_despite_old_open"] = obs.new_both_positive & obs.old_option_open
    respondents = regular.index[eligible]
    links = regular.index.intersection(second.index)
    results = []
    for name, idx in [
        ("regular_all", regular.index),
        ("regular_never_offered", respondents),
        ("other_regular", regular.index[~eligible]),
    ]:
        same = obs[obs.index.isin(idx) & obs.returned_same_grade]
        grouped = dict(
            year=year,
            group=name,
            regular=len(idx),
            returns_any_grade=int(idx.isin(second.index).sum()),
            returns_same_grade=len(same),
            with_manual=int(same.has_manual.sum()),
            any_new_school=int(same.new_school.sum()),
            any_new_course=int(same.new_course.sum()),
            first_new_school=int(same.first_new.sum()),
            any_new_school_with_prior_positive_seat=int(same.new_prior_positive.sum()),
            any_new_school_with_both_positive_seat=int(same.new_both_positive.sum()),
            prior_manual_option_with_second_vacancy=int(same.old_option_open.sum()),
            new_school_with_prior_option_still_open=int(
                same.new_despite_old_open.sum()
            ),
            new_school_in_both_supplies_with_prior_option_still_open=int(
                same.new_both_despite_old_open.sum()
            ),
            offered_second=int(same.rbd_admitido.notna().sum()),
            offered_second_explicit_course=int(same.got_exact.sum()),
            offered_second_explicit_new_school=int(same.got_new.sum()),
            offered_second_first_explicit_new_school=int(same.got_new_first.sum()),
            offered_second_new_in_prior_positive=int(same.got_new_prior_positive.sum()),
            offered_second_new_in_both_positive=int(same.got_new_both_positive.sum()),
        )
        assert (
            grouped["offered_second"]
            >= grouped["offered_second_explicit_course"]
            >= grouped["offered_second_explicit_new_school"]
        )
        results.append(grouped)
    meta = dict(
        year=year,
        regular=len(regular),
        second=len(second),
        linked=len(links),
        second_only=len(second) - len(links),
        grade_differs=int(
            obs.index.isin(regular.index).sum() - obs.returned_same_grade.sum()
        ),
        original_edges=len(original),
        explicit_second_edges=len(later),
        second_courses_with_positive_vacancy=int((of2.vacantes > 0).sum()),
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return results, meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["early-archives", "late-archives", "out"]:
        p.add_argument("--" + key, required=True, type=Path)
    args = p.parse_args()
    summ = []
    provenance = []
    for year, filename in RELEASES.items():
        result, meta = cohort(
            (args.early_archives if year < 2024 else args.late_archives) / filename,
            year,
        )
        summ.extend(result)
        provenance.append(meta)
        print(result[1], flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summ).to_csv(args.out / "round-link-counts.csv", index=False)
    pd.DataFrame(provenance).to_csv(args.out / "round-link-inputs.csv", index=False)
