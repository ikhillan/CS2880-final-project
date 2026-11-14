"""Check linked-school hits by rebuilding the menu rows from the original C1 applications."""

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from check_round_link import records

RELEASES = {
    2022: "SAE_2022.rar",
    2023: "SAE-2023.rar",
    2024: "SAE_2024.rar",
    2025: "SAE_2025.rar",
}


def independent_check(sae, numeric, enroll, year, expected):
    rosters = {
        r["mrun"]: (
            int(r["rbd_admitido"]) if r["rbd_admitido"].strip() else -1,
            int(r["cod_curso_admitido"]) if r["cod_curso_admitido"].strip() else -1,
            (
                r["respuesta_postulante"].strip(),
                r["respuesta_postulante_post_lista_espera"].strip(),
            )
            == ("6", "6"),
        )
        for r in records(sae, "D1")
    }
    assignment = {}
    fids = set(rosters)
    listing = subprocess.check_output(["bsdtar", "-tf", enroll], text=True).splitlines()
    (member,) = [x for x in listing if x.lower().endswith(".csv")]
    with subprocess.Popen(
        ["bsdtar", "-xOf", enroll, member], stdout=subprocess.PIPE
    ) as p:
        with io.TextIOWrapper(p.stdout, encoding="utf-8-sig") as stream:
            rows = csv.reader(stream, delimiter=";")
            h = next(rows)
            kid, school, edu, gr = [
                h.index(q) for q in ("MRUN", "RBD", "COD_ENSE", "COD_GRADO")
            ]
            for r in rows:
                key = r[kid]
                if key in fids:
                    assert key not in assignment
                    assignment[key] = (int(r[school]), int(r[edu]), int(r[gr]))
        if p.wait():
            raise RuntimeError("Enrollment decode failed")
    program = {
        (int(r["rbd"]), int(r["cod_curso"])): (int(r["cod_ense"]), int(r["cod_grado"]))
        for r in records(sae, "A1")
    }
    # C1 is unsorted in some cohorts; order edges by sorted numeric MRUN and raw rank.
    listing = subprocess.check_output(["bsdtar", "-tf", sae], text=True).splitlines()
    (member,) = [
        x for x in listing if Path(x).name.startswith("C1_") and x.endswith(".csv")
    ]
    with subprocess.Popen(["bsdtar", "-xOf", sae, member], stdout=subprocess.PIPE) as p:
        apps = pd.read_csv(
            p.stdout,
            sep=";",
            usecols=[
                "mrun",
                "preferencia_postulante",
                "rbd",
                "cod_curso",
                "prioridad_matriculado",
            ],
        )
        if p.wait():
            raise RuntimeError("C1 decode failed")
    apps = apps.sort_values(["mrun", "preferencia_postulante"])
    incumbent = (
        apps[apps.prioridad_matriculado == 1].set_index("mrun").preferencia_postulante
    )
    assert not incumbent.index.has_duplicates
    pos = apps.mrun.map(incumbent)
    apps = apps[pos.isna() | (apps.preferencia_postulante <= pos)]
    masks = {
        tag: np.fromfile(numeric / f"{year}-{tag}.mask", dtype=np.uint8)
        for tag in ("blind", "rank")
    }
    assert {len(x) for x in masks.values()} == {len(apps)}

    # Distinct sets of masked pupil identifiers implement one hit per school-level target.
    def empty():
        return {k: set() for k in ("eligible_graph_hits", "blind_hits", "rank_hits")}

    hits = {
        (t, g): empty()
        for t in ("enrollment", "historical_offer")
        for g in ("school", "school_type_grade")
    }
    for row, show_blind, show_rank in zip(
        apps[["mrun", "rbd", "cod_curso"]].itertuples(index=False, name=None),
        masks["blind"],
        masks["rank"],
    ):
        kid, rbd, course = row
        key = str(kid)
        entry = assignment.get(key)
        old_school, old_course, _ = rosters[key]
        if entry is None:
            continue
        for label, school, etype, grade in [
            ("enrollment", *entry),
            (
                "historical_offer",
                old_school,
                *program.get((old_school, old_course), (-1, -1)),
            ),
        ]:
            if school != rbd:
                continue
            for grain in (
                ["school", "school_type_grade"]
                if program[(rbd, course)] == (etype, grade)
                else ["school"]
            ):
                sets = hits[(label, grain)]
                sets["eligible_graph_hits"].add(key)
                if show_blind:
                    sets["blind_hits"].add(key)
                if show_rank:
                    sets["rank_hits"].add(key)
    linked = set(assignment)
    never = {x for x in linked if rosters[x][2]}
    verified = 0
    for group, ids in [
        ("all", linked),
        ("never_offered", never),
        ("other_regular", linked - never),
    ]:
        for (t, g), sets in hits.items():
            raw = expected.loc[(year, group, t, g)]
            assert raw.applicants == len(ids)
            assert raw.target_observed == (
                len(ids) if t == "enrollment" else sum(rosters[x][0] >= 0 for x in ids)
            )
            for col, s in sets.items():
                assert raw[col] == len(s & ids), (year, group, t, g, col)
            assert raw.gained_hits == len(
                ids & (sets["rank_hits"] - sets["blind_hits"])
            )
            assert raw.lost_hits == len(ids & (sets["blind_hits"] - sets["rank_hits"]))
            assert raw.net_hits == raw.gained_hits - raw.lost_hits
            verified += 1
    return {
        "year": year,
        "register_linked_applicants": len(linked),
        "eligible_edges_rebuilt_from_C1": len(apps),
        "checked_linked_population_rows": verified,
    }


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
    checks = []
    target = (
        pd.read_csv(args.out / "enrollment-target.csv")
        .query("population=='register_linked'")
        .set_index(["year", "group", "target", "resolution"])
    )
    for yr, archive in RELEASES.items():
        root = args.early_archives if yr < 2024 else args.late_archives
        num = args.early_numeric if yr < 2024 else args.late_numeric
        result = independent_check(
            root / archive,
            num,
            args.enrollments / f"matricula-{yr + 1}.rar",
            yr,
            target,
        )
        checks.append(result)
        print(result, flush=True)
    (args.out / "enrollment-target-checks.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
