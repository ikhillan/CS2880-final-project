"""Prepare coarse priority tiers and historical enrollment label vectors in a private directory."""

import argparse
from pathlib import Path
import subprocess
import sys
import pandas as pd


def prepare(args):
    publication = args.publication.resolve()
    sys.path.insert(0, str(publication / "two-studies/audit/chile"))
    sys.path.insert(0, str(publication / "two-studies/behavior/field"))
    from run_simulated_menu import late_tiers
    from round_link import RELEASES, read_file
    from link_register import enrollment_register

    out = args.out
    out.mkdir(exist_ok=True, parents=True)
    for year in args.years:
        early = year < 2024
        archives = args.early_archives if early else args.late_archives
        numeric = args.early_numeric if early else args.late_numeric
        if early:
            tier = subprocess.check_output(
                [
                    sys.executable,
                    publication
                    / "two-studies/robustness/chile/school_priority_tiers.py",
                    "--school",
                    publication / "research/observed/school",
                    "--archives",
                    archives,
                    "--numeric",
                    numeric,
                    "--cohorts",
                    publication / "research/policy/equity/cohorts.json",
                    "--year",
                    str(year),
                ]
            )
        else:
            tier = late_tiers(archives, numeric, publication, year)
        (out / f"{year}-tiers").write_bytes(tier)
        raw = archives / RELEASES[year]
        files = subprocess.check_output(["bsdtar", "-tf", raw], text=True).splitlines()
        pupils = read_file(raw, files, "B1", ["mrun"]).mrun.sort_values()
        school = read_file(raw, files, "A1", ["rbd"]).rbd
        observed, _ = enrollment_register(
            args.enrollments / f"matricula-{year+1}.rar", pupils, year + 1
        )
        april = observed.RBD.reindex(pd.Index(pupils)).fillna(-1).astype("<i4")
        with (out / f"{year}-register").open("wb") as stream:
            school.to_numpy("<i4").tofile(stream)
            april.to_numpy("<i4").tofile(stream)
        print(year, len(tier), len(school), len(april), (april >= 0).sum(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "publication",
        "early-archives",
        "early-numeric",
        "late-archives",
        "late-numeric",
        "enrollments",
        "out",
    ]:
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument(
        "--years",
        type=int,
        nargs="+",
        choices=[2022, 2023, 2024],
        default=[2022, 2023, 2024],
    )
    prepare(p.parse_args())
