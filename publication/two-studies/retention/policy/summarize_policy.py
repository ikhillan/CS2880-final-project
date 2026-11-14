"""Tabulate paired lottery audits and exact one-sided upper confidence limits."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from scipy.stats import beta


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize(results):
    output=[]
    for split in ("design_audit","holdout_course","holdout_family"):
        groups=defaultdict(list)
        for r in read(results/(split+".csv")):
            groups[(int(r["year"]),r["policy"])].append(r)
        for (year,policy), rows in groups.items():
            exits=[int(r["exits"]) for r in rows]; n=len(rows); k=sum(v>0 for v in exits)
            def total(key):
                return sum(int(r[key]) for r in rows)
            output.append(dict(split=split,year=year,policy=policy,draws=n,any_exit_draws=k,
                               upper95_any_exit=("" if split=="design_audit" else f"{1. if k==n else beta.ppf(.95,k+1,n-k):.8f}"),
                               incumbent_exits=sum(exits),worst_single_draw=max(exits),new_recipients=total("entries"),
                               retained_switching_schools=total("school_switch"),retained_switching_courses=total("course_switch"),
                               retained_better_report_position=total("retained_better"),
                               retained_worse_report_position=total("retained_worse")))
    with (results/"policy-summary.csv").open("w",newline="") as out:
        w=csv.DictWriter(out,fieldnames=output[0],lineterminator="\n");w.writeheader();w.writerows(output)
    hits=read(results/"coverage.csv")
    for r in hits:
        assert int(r["selected_hits"])-int(r["baseline_hits"])==int(r["new_hits"])-int(r["lost_hits"])
        assert int(r["linked"])<=int(r["pupils"])
    print(*(row for row in output if row["split"]!="design_audit"),sep="\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("results",type=Path)
    summarize(ap.parse_args().results)
