"""Recompute the focal cross-round counts with streaming CSV and sets of school codes."""

import argparse
import csv
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path

NAMES = {
    2022: "SAE_2022.rar",
    2023: "SAE-2023.rar",
    2024: "SAE_2024.rar",
    2025: "SAE_2025.rar",
}


def records(path, prefix):
    listing = subprocess.check_output(["bsdtar", "-tf", path], text=True).splitlines()
    (source,) = [
        name
        for name in listing
        if Path(name).name.startswith(prefix + "_") and name.endswith(".csv")
    ]
    with subprocess.Popen(
        ["bsdtar", "-xOf", path, source], stdout=subprocess.PIPE
    ) as process:
        with io.TextIOWrapper(process.stdout, encoding="utf-8-sig") as stream:
            yield from csv.DictReader(stream, delimiter=";")
        if process.wait():
            raise RuntimeError(f"Failed reading {prefix}")


def verify(path, year, target, enrollments=None, register_target=None):
    # Link keys (ID, grade, school, course) below are always taken
    # directly from the raw files. Every participant is counted at most once.
    eligible = {}
    for row in records(path, "D1"):
        response = row["respuesta_postulante"].strip()
        wait = row["respuesta_postulante_post_lista_espera"].strip()
        if response == "6" and wait == "6":
            assert (
                not row["rbd_admitido"].strip()
                and not row["rbd_admitido_post_resp"].strip()
            )
            eligible[row["mrun"]] = row["cod_nivel"]
    linked = {}
    for row in records(path, "B2"):
        kid = row["mrun"]
        if kid in eligible and row["cod_nivel"] == eligible[kid]:
            linked[kid] = set()
    positive_supply = {}
    for pre in ("A1", "A2"):
        positive_supply[pre] = {
            (row["cod_nivel"], row["rbd"], row["cod_curso"])
            for row in records(path, pre)
            if int(row["vacantes"]) > 0
        }
    still_open = set()
    for row in records(path, "C1"):
        kid = row["mrun"]
        if kid in linked:
            linked[kid].add(row["rbd"])
            if (
                row["agregada_por_continuidad"] == "0"
                and (row["cod_nivel"], row["rbd"], row["cod_curso"])
                in positive_supply["A2"]
            ):
                still_open.add(kid)
    offers = {
        row["mrun"]: (row["rbd_admitido"].strip(), row["cod_curso_admitido"].strip())
        for row in records(path, "D2")
        if row["mrun"] in linked
    }
    flags = defaultdict(set)
    for row in records(path, "C2"):
        kid = row["mrun"]
        if kid not in linked or row["agregada_por_continuidad"] != "0":
            continue
        option = (row["rbd"], row["cod_curso"])
        supply = (row["cod_nivel"], *option)
        if option == offers[kid]:
            flags["offered_second_explicit_course"].add(kid)
        if row["rbd"] not in linked[kid]:
            flags["any_new_school"].add(kid)
            if int(row["preferencia_postulante"]) == 1:
                flags["first_new_school"].add(kid)
            if kid in still_open:
                flags["new_school_with_prior_option_still_open"].add(kid)
            if option == offers[kid]:
                flags["offered_second_explicit_new_school"].add(kid)
                if supply in positive_supply["A1"] and supply in positive_supply["A2"]:
                    flags["offered_second_new_in_both_positive"].add(kid)
    counts = {
        "regular": len(eligible),
        "returns_same_grade": len(linked),
        "prior_manual_option_with_second_vacancy": len(still_open),
        **{k: len(v) for k, v in flags.items()},
    }
    for name, count in counts.items():
        if int(target[name]) != count:
            raise ValueError((year, name, count, target[name]))
    result = {"year": year, "verified_columns": list(counts), "counts": counts}
    if enrollments is not None:
        # A separate implementation scans the official register directly. Match the
        # offered school as a string, then check the exact A2 teaching and grade codes.
        ids = flags["offered_second_explicit_new_school"]
        course_codes = {
            (r["rbd"], r["cod_curso"]): (r["cod_ense"], r["cod_grado"])
            for r in records(path, "A2")
        }
        source = enrollments / f"matricula-{year + 1}.rar"
        variants = subprocess.check_output(
            ["bsdtar", "-tf", source], text=True
        ).splitlines()
        (reg_csv,) = [n for n in variants if n.lower().endswith(".csv")]
        matched, at_school, at_grade = set(), set(), set()
        with subprocess.Popen(
            ["bsdtar", "-xOf", source, reg_csv], stdout=subprocess.PIPE
        ) as p:
            with io.TextIOWrapper(p.stdout, encoding="utf-8-sig") as stream:
                # Select the five relevant column positions once for speed. Some other
                # fields are quoted text. A full CSV parser preserves correct positions.
                rows = csv.reader(stream, delimiter=";")
                header = next(rows)
                i, s, e, g, y = [
                    header.index(k)
                    for k in ["MRUN", "RBD", "COD_ENSE", "COD_GRADO", "AGNO"]
                ]
                for row in rows:
                    kid = row[i]
                    if kid not in ids:
                        continue
                    assert kid not in matched and int(row[y]) == year + 1
                    matched.add(kid)
                    if offers[kid][0] == row[s]:
                        at_school.add(kid)
                        if course_codes[offers[kid]] == (row[e], row[g]):
                            at_grade.add(kid)
            if p.wait():
                raise RuntimeError("Failed to read school register")
        enroll_counts = {
            "register_any_among_new_offers": len(matched),
            "registered_in_new_offered_school": len(at_school),
            "registered_in_new_offered_school_type_grade": len(at_grade),
        }
        for name, value in enroll_counts.items():
            assert value == int(register_target[name]), (
                year,
                name,
                value,
                register_target[name],
            )
        result["verified_register_columns"] = enroll_counts
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("early-archives", "late-archives", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument(
        "--enrollments",
        type=Path,
        help="Also independently check linked April enrollment counts",
    )
    args = p.parse_args()
    enroll_targets = {}
    if args.enrollments:
        with (args.out / "round-link-register.csv").open() as f:
            enroll_targets = {
                int(r["year"]): r
                for r in csv.DictReader(f)
                if r["historical_stratum"] == "never_offered"
                and r["menu_transition"] == "all"
            }
    with (args.out / "round-link-counts.csv").open() as f:
        targets = {
            int(r["year"]): r
            for r in csv.DictReader(f)
            if r["group"] == "regular_never_offered"
        }
    checks = []
    for yr, filename in NAMES.items():
        archive = (args.early_archives if yr < 2024 else args.late_archives) / filename
        checks.append(
            verify(archive, yr, targets[yr], args.enrollments, enroll_targets.get(yr))
        )
        print(yr, "verified", flush=True)
    (args.out / "round-link-checks.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
