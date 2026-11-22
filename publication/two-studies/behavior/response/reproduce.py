"""Build the response study from the regular-round institutional replication."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
YEARS = range(2022, 2026)

def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)

def main(args):
    pub, archives, inst, work, results = [path.resolve() for path in
        (args.publication, args.archives, args.institutional_private, args.private, args.results)]
    if work == results or work in results.parents or results in work.parents:
        raise ValueError("Separate applicant-level scratch from aggregate outputs.")
    for folder in (work, results):
        folder.mkdir(exist_ok=True, parents=True)
    sys.path.insert(0, str(pub / "two-studies/behavior/field"))
    from round_link import RELEASES
    engine = ROOT / "engine"
    field = ROOT / "field"
    linkage = ROOT / "reapplication"
    recourse = ROOT / "recourse"
    school = pub / "research/observed/school"
    institution = pub / "two-studies/institutional"
    py = sys.executable
    def prefix(year):
        return (args.early_numeric if year <= 2023 else args.late_numeric).resolve()/str(year)
    def register(year):
        return inst / f"{year}-register"

    if args.stage in ("prepare", "all"):
        for year in YEARS:
            run(py, linkage/"prepare_relisted.py", "--publication", pub,
                "--archive", archives / RELEASES[year], "--numeric", prefix(year),
                "--private", work, "--register", register(year), "--year", year,
                "--out", results / f"relisted-{year}.csv")
        run(py, linkage/"verify_relisted.py", "--publication", pub, "--results", results)

    if args.stage in ("replay", "all"):
        run(py, engine/"reproduce_response.py", "--publication", pub, "--early-numeric",
            args.early_numeric, "--late-numeric", args.late_numeric, "--side", inst,
            "--registers", inst, "--family", args.family or inst,
            "--family-template", args.family_template, "--relisted", work,
            "--private", work, "--results", results)
        run(py, linkage/"loss_history.py", "--inputs", work, "--solver", work,
            "--out", results/"loss-history.csv")

    if args.stage in ("bounds", "all"):
        for source in (field/"coverage_selection.cpp",field/"selection_replay.cpp",
                       field/"participation_replay.cpp",recourse/"export_status.cpp"):
            run(os.environ.get("CXX","c++"), "-O3", "-std=c++17", f"-I{school}",
                f"-I{institution}", f"-I{engine}", source, "-o", work/source.stem)
        for year in YEARS:
            annual = (prefix(year), inst/f"{year}.side", register(year))
            run(work/"coverage_selection", *annual, year, results/f"selection-{year}.csv")
            run(work/"selection_replay", *annual, year, results/f"selection-replay-{year}.csv")
            willing = work / f"{year}-willing.u1"
            run(py, field/"participation_mask.py", "--counts", work/f"{year}-reapplication-counts.npz",
                "--out", willing)
            run(work/"participation_replay", *annual, year, willing, work/f"{year}-unoffered.u1",
                results/f"participation-{year}.csv")
            run(work/"export_status", *annual, work/str(year))
        run(py, field/"verify_selection.py","--inputs",results,"--results",results,
            "--out",results/"selection-summary.csv")
        run(py, field/"verify_participation.py","--relisted",results,"--results",results,
            "--out",results/"participation-summary.csv")
        for script, filename in (("coverage_thresholds","coverage-thresholds"),("differential_response","differential-response")):
            run(py, field/(script+".py"), "--inputs",results,"--out",results/(filename+".csv"))
        run(py, recourse/"second_chances.py","--search",work,"--status",work,"--registers",inst,
            "--out",results/"second-chances.csv")
        run(py, field/"check_suffix_example.py","--publication",pub)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ("publication","archives","early-numeric","late-numeric","institutional-private","private","results"):
        ap.add_argument("--"+flag, required=True, type=Path)
    ap.add_argument("--stage", choices=("prepare","replay","bounds","all"), default="all")
    ap.add_argument("--family", type=Path, help="Directory of course-level family lottery inputs.")
    ap.add_argument("--family-template", default="course-family-{year}.bin")
    main(ap.parse_args())
