"""Check necessary implications of a common per-school applicant order in recorded C1 ranks."""

import argparse
import io
import itertools
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd


def read_table(archive, table, fields):
    names = subprocess.check_output(["bsdtar", "-tf", archive], text=True).splitlines()
    hits = [
        n
        for n in names
        if Path(n).name.startswith(table + "_") and n.lower().endswith(".csv")
    ]
    if len(hits) != 1:
        raise ValueError(f"Expected one {table} in {archive.name}, found {len(hits)}")
    data = subprocess.check_output(["bsdtar", "-xOf", archive, hits[0]])
    first = data.splitlines()[0]
    enc = "utf-8-sig" if first.startswith(b"\xef\xbb\xbf") else "latin1"
    frame = pd.read_csv(
        io.BytesIO(data),
        delimiter=";",
        encoding=enc,
        usecols=lambda s: s.lower() in fields,
    )
    frame.columns = frame.columns.str.lower()
    return frame


def check(archive, year):
    d = read_table(archive, "C1", {"mrun", "rbd", "cod_curso", "loteria_original"})
    key = ["rbd", "mrun"]
    assert (
        not d.duplicated(key + ["cod_curso"]).any() and d.loteria_original.notna().all()
    )
    multi = d[d.duplicated(key, keep=False)]
    peer = multi.merge(multi, on=key, suffixes=("_a", "_b"), validate="many_to_many")
    peer = peer.loc[peer.cod_curso_a < peer.cod_curso_b]
    summary = defaultdict(int)
    summary["schools_courses"] = d[["rbd", "cod_curso"]].drop_duplicates().shape[0]
    summary["multiple_courses_pupil_school"] = len(multi.groupby(key))
    summary["same_numeric_lottery_at_all_courses"] = int(
        multi.groupby(key).loteria_original.nunique().eq(1).sum()
    )
    for _, g in peer.groupby(["rbd", "cod_curso_a", "cod_curso_b"], sort=False):
        if len(g) < 2:
            continue
        reversed_pairs = 0
        total = 0
        for (a1, b1), (a2, b2) in itertools.combinations(
            zip(g.loteria_original_a, g.loteria_original_b), 2
        ):
            if a1 == a2 or b1 == b2:
                summary["tied_pupil_pairs"] += 1
                continue
            total += 1
            reversed_pairs += int((a1 < a2) != (b1 < b2))
        summary["strict_pupil_pairs"] += total
        summary["inverted_pupil_pairs"] += reversed_pairs
        summary["testable_course_pairs"] += int(total > 0)
        summary["course_pairs_with_inversion"] += int(reversed_pairs > 0)
    return {"year": year, **summary}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--archives",
        nargs=3,
        type=Path,
        required=True,
        metavar=("SAE_2022", "SAE_2023", "SAE_2024"),
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = (
        pd.DataFrame(
            [check(a, year) for year, a in zip((2022, 2023, 2024), args.archives)]
        )
        .fillna(0)
        .astype(int)
    )
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))
