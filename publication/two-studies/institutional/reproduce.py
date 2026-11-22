"""Rebuild the Chile institutional replay from named Ministry archives and frozen fits."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlparse
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
YEARS = (2022, 2023, 2024, 2025)


def read_json(path):
    return json.loads(path.read_text())


def run(*args):
    print("+", *(str(s) for s in args), flush=True)
    subprocess.run(args, check=True)


def sources(pub, sae, registers, template, include_history, years):
    school = pub / "research/observed/school"
    fit_file = school / "fit-and-inputs.json"
    early = read_json(pub / "research/policy/equity/cohorts.json")
    if hashlib.sha256(fit_file.read_bytes()).hexdigest() != early["frozen_fit_sha256"]:
        raise ValueError("The saved predictor coefficients differ from the original study.")
    old = read_json(fit_file)["cohorts"] if include_history else []
    expected = {}
    for row in old + early["sources"]:
        url = row["source"]
        expected[sae / Path(urlparse(url).path).name] = (url, row["source_sha256"])
    for row in read_json(pub / "two-studies/robustness/sources.json"):
        if include_history or any(str(year) in row["filename"] for year in years):
            expected[sae / row["filename"]] = (row["url"], row["sha256"])
    with (pub / "two-studies/behavior/field/register-inputs.csv").open() as f:
        for row in csv.DictReader(f):
            if int(row["entry_year"]) - 1 not in years:
                continue
            path = registers / template.format(year=row["entry_year"])
            if path in expected:
                raise ValueError(f"Two inputs resolve to {path}; include {{year}} in the template.")
            expected[path] = (row["source"], row["archive_sha256"])
    return expected


def verify_source(path, url, digest, download):
    if not path.exists() and download:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urlopen(url, timeout=120) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Downloaded bytes differ from the recorded source at {url}")
        tmp.replace(path)
    if not path.exists():
        raise FileNotFoundError(f"Place {url} at {path}, or select --download")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"SHA-256 mismatch for {path}; expected the archive from {url}")


def main(a):
    pub, sae, reg, private, out = (
        p.resolve() for p in (a.publication, a.archives, a.registers, a.private, a.results)
    )
    if private == out or out in private.parents or private in out.parents:
        raise ValueError("Use separate private and aggregate result directories.")
    eq = pub / "research/policy/equity"
    school = pub / "research/observed/school"
    robust = pub / "two-studies/robustness"
    field = pub / "two-studies/behavior/field"
    audit = pub / "two-studies/audit/chile"
    sys.path.insert(0, str(field))
    from round_link import RELEASES

    if bool(a.early_numeric) != bool(a.late_numeric):
        raise ValueError("Supply both numeric directories together to reuse private frozen files.")
    years = YEARS[:3] if a.stage == "ceilings" else YEARS
    for path, (url, digest) in sources(
        pub, sae, reg, a.register_template, not a.early_numeric, years
    ).items():
        verify_source(path, url, digest, a.download)
    for path in (private, out, private / "raw", private / "bin"):
        path.mkdir(parents=True, exist_ok=True)
    cc = os.environ.get("CXX", "c++")
    exe = private / "bin"
    py = sys.executable
    early = a.early_numeric.resolve() if a.early_numeric else private / "numeric-early"
    late = a.late_numeric.resolve() if a.late_numeric else private / "numeric-late"

    def build(name, src):
        run(
            cc,
            "-O3",
            "-std=c++17",
            f"-I{HERE}",
            f"-I{school}",
            f"-I{audit}",
            src,
            "-o",
            exe / name,
        )

    if not a.early_numeric:
        run(
            py,
            eq / "prepare_later.py",
            school,
            sae,
            sae,
            early,
            private / "early-inputs.json",
        )
        run(
            py,
            robust / "frozen_school.py",
            "--school",
            school,
            "--old",
            sae,
            "--new",
            sae,
            "--numeric",
            late,
            "--sources",
            robust / "sources.json",
            "--record",
            private / "late-inputs.json",
        )
        build("export_menus", eq / "export_menus.cpp")
        for year in years:
            num = early if year < 2024 else late
            run(exe / "export_menus", num / f"{year}.bin", str(year), num)
    for year in years:
        num = early if year < 2024 else late
        for suffix in (".bin", "-blind.mask", "-rank.mask"):
            if not (num / (str(year) + suffix)).is_file():
                raise FileNotFoundError(num / (str(year) + suffix))

    # The archive reader also checks year and columns in every enrollment chunk.
    # Keep register preparation independent of the offer-generating solver.
    if a.stage in ("historical", "all"):
        for tool in (
            "official",
            "export_programs",
            "register_offers",
            "mechanism_audit",
        ):
            build(tool, HERE / (tool + ".cpp"))
        run(py, HERE / "test_institution.py", "--school", school, "--markets", "600")
        run(py, HERE / "family_witness.py")
    if a.stage in ("ceilings", "all"):
        build("ceiling", HERE / "institution_ceiling.cpp")
        build("verify_interleave", HERE / "verify_interleave.cpp")
    for year in years:
        y = str(year)
        num = early if year < 2024 else late
        frozen = num / (y + ".bin")
        prefix = num / y
        side = private / (y + ".side")
        register = private / (y + "-register")
        archive = sae / RELEASES[year]
        run(
            py,
            HERE / "prepare_institution.py",
            "--archive",
            archive,
            "--numeric",
            frozen,
            "--out",
            side,
        )
        run(
            py,
            HERE / "prepare_register.py",
            "--publication",
            pub,
            "--year",
            y,
            "--archive",
            archive,
            "--register",
            reg / a.register_template.format(year=year + 1),
            "--out",
            register,
        )
        if a.stage in ("historical", "all"):
            run(exe / "official", prefix, side, y, out / f"official-{year}.csv")
            run(
                exe / "mechanism_audit",
                prefix,
                side,
                y,
                out / f"ablation-{year}.csv",
                out / f"transitions-{year}.csv",
                register,
            )
            register_args = (
                exe / "register_offers",
                prefix,
                side,
                register,
                y,
                out / f"linked-{year}.csv",
            )
            run(*register_args)
            run(*register_args, out / f"first-{year}.csv")
            raw = private / "raw" / (y + ".bin")
            raw_side = raw.with_suffix(".side")
            run(
                py,
                HERE / "prepare_unabridged.py",
                "--publication",
                pub,
                "--year",
                y,
                "--archives",
                sae,
                "--frozen",
                frozen,
                "--out",
                raw,
            )
            run(
                py,
                HERE / "prepare_institution.py",
                "--archive",
                archive,
                "--numeric",
                raw,
                "--all-reports",
                "--out",
                raw_side,
            )
            for tag, report, market_arg, side_arg in [
                ("cutoff", "policy_pool", frozen, side),
                ("all", "all_submitted", raw, raw_side),
            ]:
                prediction = private / f"{tag}-{year}.pred"
                run(exe / "export_programs", market_arg, side_arg, prediction)
                run(
                    py,
                    HERE / "compare_official.py",
                    "--publication",
                    pub,
                    "--archives",
                    sae,
                    "--year",
                    y,
                    "--programs",
                    prediction,
                    "--report",
                    report,
                    "--out",
                    out / f"{tag}-{year}.csv",
                )
        if a.stage in ("ceilings", "all") and year < 2025:
            # Reuse the original priority encoding, including its release and row checks.
            tier = private / (y + "-tiers")
            if year < 2024:
                with tier.open("wb") as stream:
                    subprocess.run(
                        [
                            py,
                            robust / "chile/school_priority_tiers.py",
                            "--school",
                            school,
                            "--archives",
                            sae,
                            "--numeric",
                            num,
                            "--year",
                            y,
                            "--cohorts",
                            eq / "cohorts.json",
                        ],
                        stdout=stream,
                        check=True,
                    )
            else:
                sys.path.insert(0, str(audit))
                from run_simulated_menu import late_tiers

                tier.write_bytes(late_tiers(sae, num, pub, year))
            run(exe / "verify_interleave", frozen, y, tier)
            family = private / f"school-family-{year}.bin"
            course = private / f"course-family-{year}.bin"
            for dest, opts in [(family, []), (course, ["--course-specific"])]:
                run(
                    py,
                    HERE / "prepare_family.py",
                    "--year",
                    y,
                    "--publication",
                    pub,
                    "--archives",
                    sae,
                    "--numeric",
                    frozen,
                    "--out",
                    dest,
                    *opts,
                )
            designs = [
                ("pooled", "pooled", "-", "-", "0", []),
                ("paired", "institutional", side, "-", "64", []),
                ("family", "family", side, family, "16", []),
                ("course-family", "course_family", side, course, "16", []),
                ("exact", "institutional", side, "-", "0", ["all-design"]),
            ]
            for tag, rule, model, groups, heldout, opts in designs:
                run(
                    exe / "ceiling",
                    rule,
                    num,
                    y,
                    tier,
                    register,
                    model,
                    out / f"{tag}-{year}.csv",
                    private / f"{tag}-{year}.bits",
                    "1024",
                    groups,
                    heldout,
                    *opts,
                )
    if a.stage in ("historical", "all"):
        run(py, HERE / "verify_fidelity.py", "--engine", out, "--official", out)
        for src, filename in [
            ("lottery_architecture.py", "lottery-architecture.csv"),
            ("family_adjacency.py", "family-adjacency.csv"),
        ]:
            run(
                py,
                HERE / src,
                "--archives",
                *(sae / RELEASES[year] for year in YEARS[:3]),
                "--out",
                out / filename,
            )
    if a.stage in ("ceilings", "all"):
        run(
            py,
            HERE / "summarize.py",
            "--folder",
            out,
            "--summary",
            out / "ceiling-summary.csv",
            "--audit",
            out / "exact-design-checks.csv",
            "--private",
            private,
            "--overlap",
            out / "ceiling-overlap.csv",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("publication", "archives", "registers", "private", "results"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--register-template",
        default="matricula-{year}.rar",
        help="Filename with {year} for the April entry year.",
    )
    parser.add_argument("--early-numeric", type=Path)
    parser.add_argument("--late-numeric", type=Path)
    parser.add_argument("--stage", choices=("historical", "ceilings", "all"), default="all")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing source archives after verifying the frozen manifests.",
    )
    main(parser.parse_args())
