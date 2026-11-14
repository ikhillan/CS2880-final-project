"""Rebuild masks and audit incumbent retention. Applicant-level scratch stays local."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess

YEARS = (2022, 2023, 2024)
HERE = Path(__file__).resolve().parent


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def chunks(draws, size):
    for start in range(1, draws + 1, size):
        yield start, min(start+size-1, draws)


def merge(paths, target):
    with target.open("w", newline="") as out:
        writer = None
        for path in paths:
            with path.open(newline="") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator="\n")
                    writer.writeheader()
                writer.writerows(reader)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    pub=args.publication.resolve()
    binary=args.private / "retention"
    inputs=json.loads(args.config.read_text())
    root=args.private; output=args.results
    root.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    inst=pub/"two-studies/institutional"
    includes=(pub/"research/observed/school",pub/"two-studies/audit/chile",inst)

    def paths(year):
        return {key:Path(value).resolve() for key,value in inputs[str(year)].items()}

    if args.workers < 1:
        raise ValueError("A positive number of workers is required.")
    def verify_inputs(year, keys):
        expected = json.loads((HERE/"input-sha256.json").read_text())[str(year)]
        current = paths(year)
        prefix = str(current["prefix"])
        current.update(market=Path(prefix+".bin"), blind_mask=Path(prefix+"-blind.mask"),
                       rank_mask=Path(prefix+"-rank.mask"))
        for key in keys:
            if digest(current[key]) != expected[key]:
                raise ValueError(f"Input digest differs for {year} {key}")
    def parallel(tasks):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda cmd:run(*cmd), tasks))

    if args.phase in ("all","design"):
        import numpy as np
        run(os.environ.get("CXX","c++"),"-O3","-std=c++17",*[f"-I{d}" for d in includes],HERE/"retention.cpp","-o",binary)
        common=("market","blind_mask","rank_mask","side")
        verify_inputs(2021, (*common, "register"))
        for y in YEARS:
            verify_inputs(y, (*common, "tiers"))
        train=paths(2021)
        run(binary,"fit",train["prefix"],train["side"],train["register"],output/"calibration-2021.csv")
        tasks=[]
        for y in YEARS:
            p=paths(y)
            for first,last in chunks(1024,32):
                dest=root/f"flags-{y}-{first}"
                tasks.append([binary,"screen",p["prefix"],p["side"],p["tiers"],y,first,last,dest])
        parallel(tasks)
        for y in YEARS:
            p=paths(y)
            combined=None
            for first,last in chunks(1024,32):
                flags=np.fromfile(root/f"flags-{y}-{first}",dtype="u1")
                if combined is None: combined=flags
                else: np.bitwise_or(combined,flags,out=combined)
            combined.tofile(root/f"{y}.flags")
            run(binary,"choose",p["prefix"],p["side"],root/f"{y}.flags",output/"calibration-2021.csv",root/f"{y}.mask")
        manifest={str(y):digest(root/f"{y}.mask") for y in YEARS}
        for f in ("retention.cpp","PLAN.md"):
            manifest[f]=digest(HERE/f)
        manifest["calibration-2021.csv"]=digest(output/"calibration-2021.csv")
        (output/"policy-freeze.json").write_text(json.dumps(manifest,indent=2)+"\n")
        print("All policy masks frozen",manifest,flush=True)
    frozen=json.loads((output/"policy-freeze.json").read_text())
    for y in YEARS:
        assert frozen[str(y)]==digest(root/f"{y}.mask")
    assert frozen["retention.cpp"]==digest(HERE/"retention.cpp")
    if args.phase=="design": return
    for y in YEARS:
        verify_inputs(y, ("market","blind_mask","rank_mask","side","tiers","families","register"))
    splits=[("design_audit",513,1024,False,False),("holdout_course",8615729,128,False,True),
            ("holdout_family",8615731,32,True,False)]
    for split,batch,draws,fam,rank in splits:
        tasks=[];shards=[]
        for y in YEARS:
            p=paths(y)
            for first,last in chunks(draws,16):
                dest=root/f"{split}-{y}-{first}.csv";shards.append(dest)
                tasks.append([binary,"audit",p["prefix"],p["side"],p["tiers"],y,batch,p["families"] if fam else "-",first,last,root/f"{y}.mask",dest,int(rank)])
        parallel(tasks)
        merge(shards,output/f"{split}.csv")
        print("Finished",split,flush=True)
    files=[]
    for y in YEARS:
        p=paths(y);dest=root/f"enrolled-{y}.csv"
        run(binary,"score",p["prefix"],p["side"],root/f"{y}.mask",p["register"],y,dest)
        files.append(dest)
    merge(files,output/"coverage.csv")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("publication","config","private","results"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=5)
    parser.add_argument("--phase",choices=("all","design","test"),default="all")
    main(parser.parse_args())
