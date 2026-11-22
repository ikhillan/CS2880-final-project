"""Post-freeze descriptive equity breakdowns, with independent input and total checks."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

def data(prefix):
    with Path(str(prefix)+".bin").open("rb") as stream:
        n,m,ne=np.fromfile(stream,"<i4",3)
        at=np.fromfile(stream,"<i4",n+1); prog=np.fromfile(stream,"<i4",ne)
        stream.seek(4*(ne+m),1)
        official=np.fromfile(stream,"<i4",n); priority=np.fromfile(stream,"<i4",n)
        return n,m,ne,at,prog,official,priority

def run(configs, structural_base, learned_base, structural_freeze, learned_freeze, output):
    cfg=json.loads(configs.read_text()); trained=json.loads(learned_freeze.read_text())
    structural={(int(r["year"]),r["mode"]):r["sha256"] for r in csv.DictReader(structural_freeze.open())}
    rows=[]
    for y in (2022,2023,2024):
        p=cfg[str(y)];pre=Path(p["prefix"]);n,m,ne,at,prog,offer,priority=data(pre)
        with Path(p["register"]).open("rb") as stream:
            schools=np.fromfile(stream,"<i4",m)
            attended=np.fromfile(stream,"<i4",n)
            if stream.read(1):raise ValueError("extra register data")
        owners=np.repeat(np.arange(n),np.diff(at))
        def read(path, expected=None):
            raw=path.read_bytes()
            if expected and hashlib.sha256(raw).hexdigest()!=expected:raise ValueError(f"mask changed: {path}")
            if len(raw)!=ne:raise ValueError("mask length")
            return np.frombuffer(raw,dtype="u1").astype(bool)
        masks={"baseline":read(Path(str(pre)+"-blind.mask")),
               "structural":read(structural_base/f"{y}-prospective.mask",structural[y,"prospective"]),
               "learned":read(learned_base/f"{y}.mask",trained[str(y)])}
        target=(attended[owners]>=0)&(schools[prog]==attended[owners])
        def hits(mask):
            out=np.zeros(n,bool);out[owners[target & mask]]=True
            return out
        old=hits(masks["baseline"])
        for rule in ("structural","learned"):
            new=hits(masks[rule])
            for label,group in [("all",np.ones(n,bool)),("disadvantaged",priority==1),
                    ("other_priority_status",priority!=1),("initially_unplaced",offer<0),("initially_placed",offer>=0)]:
                subset=group&(attended>=0)
                gains=np.count_nonzero(subset&new&~old);losses=np.count_nonzero(subset&old&~new)
                rows.append([y,rule,label,group.sum(),subset.sum(),gains,losses,gains-losses])
    for year in (2022,2023,2024):
        for rule in ("structural","learned"):
            part=[row for row in rows if row[0]==year and row[1]==rule]
            for col in range(3,8):
                assert part[0][col]==part[1][col]+part[2][col]==part[3][col]+part[4][col]
    with output.open("w",newline="") as stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(["year","rule","group","applicants","linked","gains","losses","net"])
        writer.writerows(rows)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ("config","structural","learned","structural-freeze","learned-freeze","output"):
        ap.add_argument("--"+arg,type=Path,required=True)
    a=ap.parse_args()
    run(a.config,a.structural,a.learned,a.structural_freeze,a.learned_freeze,a.output)
