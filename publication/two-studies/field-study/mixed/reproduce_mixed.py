"""Rebuild and verify the national mixed-policy simulations on the machine holding the microdata."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
HASH = {
    "frozen": "b376961275925006c11ef1800d14f017e71e05a8a8fb75612d6ec9547478a2e7",
    "blind": "6026f7fe02a6921381dd805c725b91b8eace01d9145887e20d132b3108e3c478",
    "rank": "3d9085e7f6a797408d435c44ad828a8a65ad7f33f6781003be967ad28c94c114",
    "side": "553a486868de31afb99d2da5be4c6c84607fa70ba8e21f43fbfaf771f4a481c7",
    "register": "5118c481bbb96c58b1cba4d8ea1c245f3d520b7288a09f0ca2827ca7c0dd53c2",
}
FILES = ("mixed-endpoints.csv", "mixed-draws.csv", "mixed-summary.csv")


def run(*cmd):
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE":"1"})


def main(args):
    prefix=args.prefix.resolve()
    inputs = {"frozen":Path(str(prefix)+".bin"),"blind":Path(str(prefix)+"-blind.mask"),
              "rank":Path(str(prefix)+"-rank.mask"),"side":args.side,"register":args.register}
    for key,path in inputs.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=HASH[key]:
            raise ValueError(f"Input digest differs for {key}.")
    if not 1<=args.workers<=64:
        raise ValueError("--workers must be between 1 and 64.")
    args.results.mkdir(parents=True,exist_ok=True)
    if args.results.resolve()==HERE:
        raise ValueError("Provide a result directory separate from the release.")
    pub=args.publication.resolve()
    run(sys.executable,HERE/"check_estimator.py")
    with tempfile.TemporaryDirectory() as tmp:
        stage=Path(tmp)
        binary=stage/"mixed"
        run(os.environ.get("CXX","c++"),"-O3","-std=c++17",f"-I{pub}/research/observed/school",
            f"-I{pub}/two-studies/institutional",HERE/"mixed_assign.cpp","-o",binary)

        def partition(j):
            lo=1+64*j//args.workers
            hi=64*(j+1)//args.workers
            endpoint,raw=stage/f"end-{j}.csv",stage/f"draws-{j}.csv"
            run(binary,prefix,args.side,args.register,str(lo),str(hi),endpoint,raw,"47319")
            if endpoint.read_bytes()!=(HERE/FILES[0]).read_bytes():
                raise ValueError("Full rollout endpoints differ from the release.")
            return raw

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            shards=list(pool.map(partition,range(args.workers)))
        (stage/FILES[0]).write_bytes((stage/"end-0.csv").read_bytes())
        extra=["--cohort",args.cohort] if args.cohort else []
        run(sys.executable,HERE/"summarize_mixed.py","--directory",stage,*extra,"--shards",*shards)
        for name in FILES:
            content=(stage/name).read_bytes()
            if content!=(HERE/name).read_bytes():
                raise ValueError(f"Rebuilt {name} differs from release.")
        for name in FILES:
            (args.results/name).write_bytes((stage/name).read_bytes())
    print("All three aggregate outputs match the release byte for byte.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ("publication","prefix","side","register","results"):
        parser.add_argument("--"+field,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--cohort",type=Path,help="Independent entry-2021.csv for end-point concordance.")
    main(parser.parse_args())
