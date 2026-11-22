"""Run independent seed shards inside one Slurm allocation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shards", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def one(shard):
        with (args.out / f"worker-{shard:02d}.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "experiments_jobs.anchor_access", "evaluate",
                            "--packet", str(args.packet), "--out", str(args.out),
                            "--shard", str(shard), "--shards", str(args.shards), "--ties", "20"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        return shard

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, shard) for shard in range(args.shards)]
        for future in as_completed(futures):
            print(f"Completed shard {future.result()}", flush=True)


if __name__ == "__main__":
    main()
