"""Rebuild the 2021 entry-grade benchmark and compare each published output byte for byte."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
DIGEST = {
    "SAE_2019.zip": "79df389c05e37b040c728dfc80bdc0dfd7ca1cbdde708710771e3e048e582f6e",
    "SAE_2021.rar": "8bd459562803c6a57e60e51ecf8198beb212100400e8cdd058cbcfbf1aa29261",
    "register": "9d5ecc67b3bf0e70256c001b70d4246d89b7618f857a395a58c2b4b58840bd9a",
    "fit-and-inputs.json": "122413f014977a69ddd213e41b87a3979b4b979dd9414c97b7c4a1fe91ae4f75",
    "2021.bin": "b376961275925006c11ef1800d14f017e71e05a8a8fb75612d6ec9547478a2e7",
    "2021-blind.mask": "6026f7fe02a6921381dd805c725b91b8eace01d9145887e20d132b3108e3c478",
    "2021-rank.mask": "3d9085e7f6a797408d435c44ad828a8a65ad7f33f6781003be967ad28c94c114",
}


def check(path, key):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != DIGEST[key]:
        raise ValueError(f"SHA-256 mismatch for {path}; expected {DIGEST[key]}")


def reproduce(args):
    pub = args.publication.resolve()
    archives = args.archives.resolve()
    enrollment = args.register.resolve()
    sae = archives / "SAE_2021.rar"
    institutional = pub / "two-studies/institutional"
    school = pub / "research/observed/school"
    check(sae, sae.name)
    check(enrollment, "register")
    fit = school / "fit-and-inputs.json"
    check(fit, fit.name)
    if not args.numeric:
        check(archives / "SAE_2019.zip", "SAE_2019.zip")
    results = args.results.resolve()
    if results == HERE:
        raise ValueError("Choose a result directory separate from the reference tables.")
    results.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    compiler = os.environ.get("CXX", "c++")
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}

    def run(*cmd):
        print("+", *cmd, flush=True)
        subprocess.run(cmd, check=True, env=env)

    def build(source, target):
        run(compiler, "-O3", "-std=c++17", f"-I{school}", f"-I{institutional}", source, "-o", target)

    with tempfile.TemporaryDirectory(prefix="entry-2021-", dir=args.scratch) as tmp:
        stage = Path(tmp).resolve()
        numeric = args.numeric.resolve() if args.numeric else stage
        frozen = numeric / "2021.bin"
        if args.numeric:
            check(frozen, frozen.name)
        else:
            run(sys.executable, HERE / "frozen_2021.py", "--school", school, "--archives",
                archives, "--output", frozen)
            build(school / "check_partners.cpp", stage / "menus")
            run(stage / "menus", frozen, "2021", numeric)
        for name in ("2021-blind.mask", "2021-rank.mask"):
            check(numeric / name, name)

        run(sys.executable, institutional / "prepare_institution.py", "--archive", sae,
            "--numeric", frozen, "--out", stage / "frozen.side")
        historical = [sys.executable, HERE / "historical.py"]
        inputs = ["--archives", archives, "--publication", pub]
        run(*historical, "prepare", *inputs, "--frozen", frozen, "--out", stage / "all.bin")
        run(sys.executable, institutional / "prepare_institution.py", "--archive", sae,
            "--numeric", stage / "all.bin", "--all-reports", "--out", stage / "all.side")
        build(institutional / "export_programs.cpp", stage / "export")
        run(stage / "export", stage / "all.bin", stage / "all.side", stage / "all.pred")
        run(*historical, "compare", *inputs, "--programs", stage / "all.pred", "--out",
            stage / "official-2021.csv")
        run(sys.executable, institutional / "prepare_register.py", "--publication", pub,
            "--archive", sae, "--year", "2021", "--register", enrollment,
            "--out", stage / "register")
        build(HERE / "entry_grades.cpp", stage / "entry")
        run(stage / "entry", numeric / "2021", stage / "frozen.side", stage / "register",
            stage / "all.pred", stage / "entry-2021.csv")

        for name in ("official-2021.csv", "entry-2021.csv"):
            if (stage / name).read_bytes() != (HERE / name).read_bytes():
                raise ValueError(f"Rebuilt output differs from reference {name}")
        for name in ("official-2021.csv", "entry-2021.csv"):
            (results / name).write_bytes((stage / name).read_bytes())
    print("Both tables agree byte for byte with the release.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("publication", "archives", "register", "scratch", "results"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--numeric", type=Path, help="Optional original staged market and masks.")
    reproduce(parser.parse_args())
