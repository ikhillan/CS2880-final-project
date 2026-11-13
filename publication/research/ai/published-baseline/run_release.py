"""Rerun the authors' size-20 benchmark using their pinned checkout.

Only the undefined variable used in the *last file serialization* is replaced.
No numerical code in either original script changes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


OPT = "market_match_sw_optim.py"
MC = "market_match_mc.py"
HASHES = {
    OPT: "9e534482c5a16ce714c9c004594995b34b60c969777d72f414ab75668f6bbef0",
    MC: "e3b843b69a579369ad2c2d6b7cece09de045e0574e29bc7d3ba113f31bb3ed57",
}


def check_source(repo):
    for file, sha in HASHES.items():
        if hashlib.sha256((repo / file).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Unexpected upstream source checksum for {file}")


def run_one(repo, work, seed, runs):
    work.mkdir(exist_ok=True, parents=True)
    patched = (repo / OPT).read_text().replace("cand_exprank_rel.numpy()", "cand_rel.numpy()")
    (work / OPT).write_text(patched)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MPLCONFIGDIR=str(work / "mpl-cache"))
    config = ["--job_num=20", "--cand_num=30", f"--seed={seed}",
              "--lambda_value=0.5", "--epoch_num=50"]
    with (work / "optim.log").open("w") as log:
        subprocess.run([sys.executable, OPT, *config], cwd=work, env=env,
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    (policy_file,) = list(work.glob("LIjob_20*.json"))
    with (work / "mc.log").open("w") as log:
        subprocess.run([sys.executable, str(repo / MC), f"--json_path={policy_file.name}",
                        f"--seed={seed}", f"--runs={runs}", "--output_path=mc.json"],
                       cwd=work, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    raw = json.loads((work / "mc.json").read_text())
    rows = []
    for policy, label in [("Greedy", "naive"), ("Reciprocal", "reciprocal"),
                          ("Ours", "social_welfare")]:
        r = raw[policy]
        rows.append(dict(seed=seed, runs=runs, jobs=20, applicants=30, crowding=0.5,
                         policy=label, interviews=r.get("Expectation", r.get("Expecation")),
                         simulation_se=r["SE"]))
    print(f"Completed seed {seed}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10000)
    args = parser.parse_args()
    repo = args.repo.resolve()
    work = args.work.resolve()
    check_source(repo)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_one, repo, work / str(s), s, args.runs) for s in range(621, 631)]
        rows = [row for job in jobs for row in job.result()]
    with args.output.open("w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
