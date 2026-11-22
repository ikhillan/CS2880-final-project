"""Run the archived 2022/23 school scoring procedure on 2024/25 and compare whole binaries.

The earlier script has its years hardcoded. Substitute only those two literals and input
archive metadata. The output binary files live in an ephemeral private directory. The
distributed receipt contains hashes and aggregate lengths.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    publication = args.repo / "publication"
    school = publication/"research/observed/school"
    old = (publication/"research/policy/equity/prepare_later.py").read_text()
    assert old.count("for yr in [2020,2021,2022,2023]:") == 1
    code = old.replace("for yr in [2020,2021,2022,2023]:", "for yr in [2024,2025]:")
    ns = {"__name__":"archived_scoring_check"}
    exec(compile(code, "prepare_later.py", "exec"), ns)
    releases = json.loads((publication/"two-studies/robustness/sources.json").read_text())
    ns["FILES"] = {int(r["filename"][4:8]):r["filename"] for r in releases}
    ns["URLS"] = {int(r["filename"][4:8]):r["url"] for r in releases}
    with TemporaryDirectory() as tmp:
        private = Path(tmp)
        ns["run"](SimpleNamespace(school=school,old=args.old,new=args.archives,
                                  numeric=private,record=private/"receipt.json"))
        check = private/"export"
        subprocess.run(["c++","-std=c++17","-O3","-I"+str(school),
                str(publication/"research/policy/equity/export_menus.cpp"),"-o",str(check)],check=True)
        counts=[]
        for year in (2024,2025):
            subprocess.run([check,private/f"{year}.bin",str(year),private],check=True)
            files=[f"{year}.bin"]+[f"{year}-{p}.{suffix}" for p in ["full","blind","rank"] for suffix in ["mask","partners"]]
            for fname in files:
                got=sha(private/fname)
                assert got==sha(args.numeric/fname), fname
                counts.append(dict(filename=fname,bytes=(private/fname).stat().st_size,sha256=got))
    args.out.write_text(json.dumps({"same_input_and_model":True, "verified_files":counts},indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ["repo","old","archives","numeric","out"]:
        parser.add_argument("--"+arg, type=Path, required=True)
    main(parser.parse_args())
