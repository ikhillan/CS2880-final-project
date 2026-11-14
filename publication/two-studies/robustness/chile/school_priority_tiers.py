"""Stream checked school priority tiers to stdout for the lottery sensitivity.

Used as a pipe: private edge information is never written to the aggregate result directory.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main(args):
    sys.path.insert(0, str(args.school.resolve()))
    import prepare

    prepare.FILES.update({2022: "SAE_2022.rar", 2023: "SAE-2023.rar"})
    prepare.URLS.update({year: name for year, name in prepare.FILES.items()})
    _, _, _, _, data, info = prepare.prepare(args.archives, args.year)
    with (args.numeric / f"{args.year}.bin").open("rb") as f:
        n, m, ne = np.fromfile(f, "<i4", 3)
        offset = np.fromfile(f, "<i4", n + 1)
        for key in ("j", "priority"):
            assert np.array_equal(np.fromfile(f, "<i4", ne), data[key]), key
    assert np.array_equal(data["i"], np.repeat(np.arange(n), np.diff(offset)))
    assert (n, m, ne) == (data["n"], data["m"], len(data["tier"]))
    if args.cohorts:
        hashes = json.loads(args.cohorts.read_text())["sources"]
        archive = next(item for item in hashes if item["year"] == args.year)
        assert archive["source_sha256"] == info["source_sha256"]
    tiers = np.asarray(data["tier"], dtype=np.uint8)
    assert np.array_equal(tiers, data["tier"])
    sys.stdout.buffer.write(tiers.tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument("--school", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path)
    parser.add_argument("--year", type=int, choices=[2022, 2023], required=True)
    main(parser.parse_args())
