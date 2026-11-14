"""Reproduce the national response simulations from cached institutional inputs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

def run(*args):
    subprocess.run([str(v) for v in args],check=True)

def main(a):
    here=Path(__file__).resolve().parent
    pub=a.publication.resolve()
    school=pub/"research/observed/school"
    institutional=pub/"two-studies/institutional"
    a.private.mkdir(parents=True,exist_ok=True)
    a.results.mkdir(parents=True,exist_ok=True)
    if (a.private.resolve()==a.results.resolve() or a.private.resolve() in a.results.resolve().parents
            or a.results.resolve() in a.private.resolve().parents):
        raise ValueError("Place person-level inputs and outputs in a different directory from published aggregates.")
    for unit in ("response_replay","risk_screen","check_response_order"):
        run(os.environ.get("CXX","c++"),"-O3","-std=c++17",f"-I{school}",f"-I{institutional}",
            here/(unit+".cpp"),"-o",a.private/unit)
    for year in range(2022,2026):
        numeric=a.early_numeric if year<=2023 else a.late_numeric
        prefix=numeric/str(year);side=a.side/f"{year}.side"
        register=a.registers/f"{year}-register"
        ranked=a.relisted/f"{year}-relisted.i4";unoffered=a.relisted/f"{year}-unoffered.u1"
        if not register.exists() and a.registers_2025 is not None and year==2025:
            register=a.registers_2025/f"{year}-register"
        run(a.private/"check_response_order",prefix,side,register,year,ranked,unoffered,a.results/f"reference-{year}.csv")
        for panel in ("endpoints","field","multiple","reapplication"):
            reapp=(panel=="reapplication")
            run(a.private/"response_replay",prefix,side,register,year,panel,a.results/f"{panel}-{year}.csv",
                a.private/f"{year}-lost.u1" if panel=="endpoints" else "-",
                ranked if reapp else "-",unoffered if reapp else "-")
        if year<2025:
            risk=a.private/f"{year}-response-risk.u1"
            family=a.family/a.family_template.format(year=year)
            run(a.private/"risk_screen",prefix,side,family,year,risk,a.results/f"risk-cohort-{year}.csv")
            run(a.private/"response_replay",prefix,side,register,year,"risk",a.results/f"risk-{year}.csv","-","-",risk)
    run(sys.executable,here/"summarize_response.py","--results",a.results,"--institutional",institutional/"results","--out",a.results)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("publication","early-numeric","late-numeric","side","registers","family","relisted","private","results"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--family-template",default="course-family-{year}.bin")
    p.add_argument("--registers-2025",type=Path)
    main(p.parse_args())
