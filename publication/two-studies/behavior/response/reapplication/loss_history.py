"""Describe historically observed reapplication for the binding rule's gross exits."""

import argparse
import csv
from pathlib import Path
import numpy as np


def main(a):
    all_rows = []
    for year in range(2022, 2026):
        hist = np.load(a.inputs / f"{year}-reapplication-counts.npz")
        lost = np.fromfile(a.solver / f"{year}-lost.u1", dtype=np.uint8)
        assert lost.shape == hist["never"].shape and np.all(lost <= 1)
        mask = lost.astype(bool)
        for group, keep in [
            ("all_gross_exits", mask),
            ("exits_never_offered", mask & hist["never"]),
        ]:
            row = dict(year=year, group=group, applicants=int(keep.sum()))
            for col in hist.files:
                if col not in ("never", "no_initial"):
                    row[col + "_pupils"] = int(np.count_nonzero(hist[col][keep]))
            all_rows.append(row)
    assert (
        sum(r["applicants"] for r in all_rows if r["group"] == "all_gross_exits")
        == 2635
    )
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, all_rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(all_rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("inputs", "solver", "out"):
        p.add_argument("--" + k, required=True, type=Path)
    main(p.parse_args())
