"""Build the order-independent menus, fix their digests, and replay the reserved draws."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
YEARS = (2022, 2023, 2024)
MODES = ("prospective", "oracle")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.output.resolve() == HERE:
        raise ValueError("Use a separate output directory to compare with the released results.")
    public = args.publication
    compile_cmd = [os.environ.get("CXX", "c++"), "-O3", "-std=c++17",
                   "-I" + str(public / "research/observed/school"),
                   "-I" + str(public / "two-studies/institutional"),
                   "-I" + str(public / "two-studies/audit/chile")]
    for name in ("structural", "check_policy"):
        subprocess.run([*compile_cmd, HERE / (name + ".cpp"), "-o", args.scratch/name], check=True)
    prefix = {y: (args.numeric_late if y == 2024 else args.numeric_early)/str(y) for y in YEARS}
    expected = list(csv.DictReader((HERE / "frozen-masks.csv").open()))
    masks = {}
    for year in YEARS:
        for mode in MODES:
            mask = args.scratch / f"{year}-{mode}.mask"
            masks[year, mode] = mask
            cmd = [args.scratch / "structural", prefix[year], args.sides/f"{year}.side",
                   mask, args.output/f"{year}-{mode}.csv", str(year), mode]
            if mode == "oracle":
                cmd.append(args.features / f"{year}-register")
            subprocess.run(cmd, check=True)
    for row in expected:
        if digest(masks[int(row["year"]), row["mode"]]) != row["sha256"]:
            raise ValueError("Constructed mask differs from frozen mask")
    for year in YEARS:
        subprocess.run([args.scratch/"check_policy", prefix[year], str(year),
                        args.features/f"{year}-tiers", args.features/f"{year}-register",
                        args.sides/f"{year}.side", masks[year, "prospective"], masks[year, "oracle"],
                        args.output/f"{year}-validation.csv","32"],check=True)
        for suffix in (*MODES, "validation"):
            name = f"{year}-{suffix}.csv"
            if (args.output/name).read_bytes() != (HERE/name).read_bytes():
                raise ValueError("Aggregate differs from reference: " + name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("publication", "scratch", "output", "numeric-early", "numeric-late", "sides", "features"):
        parser.add_argument("--" + name, required=True, type=Path)
    main(parser.parse_args())
