"""Re-clear the all-order structural menus using both frozen learner test panels."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader();w.writerows(rows)


def verify_masks(receipt, folder):
    rows = [row for row in read(receipt) if row["mode"] == "prospective"]
    if sorted(row["year"] for row in rows) != ["2022","2023","2024"]:
        raise ValueError("expected one frozen prospective mask for each evaluation year")
    for row in rows:
        path = folder / (row["year"]+"-prospective.mask")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"wrong structural mask: {path}")


def compare(args):
    args.private.mkdir(exist_ok=True,parents=True)
    args.results.mkdir(exist_ok=True,parents=True)
    inputs=json.loads(args.config.read_text())
    verify_masks(args.freeze,args.masks)
    def run(cmd):
        subprocess.run(cmd,check=True)
    summary=[]
    for split,batch,count,fam in (("holdout_course",8615729,128,False),("holdout_family",8615731,32,True)):
        jobs=[];parts=[]
        for y in (2022,2023,2024):
            c=inputs[str(y)]
            for start in range(1,count+1,8):
                out=args.private/f"{split}-{y}-{start}.csv"
                jobs.append([args.binary,"audit",c["prefix"],c["side"],c["tiers"],str(y),str(batch),c["families"] if fam else "-",
                             str(start),str(start+7),args.masks/f"{y}-prospective.mask",out,"0"])
                parts.append(out)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run,jobs))
        target={(r["year"],r["draw"]):r for r in read(args.learned/f"{split}.csv") if r["policy"]=="learned_retention"}
        out=[]
        for f in parts:
            for r in read(f):
                key=r["year"],r["draw"]
                if key not in target:raise ValueError(f"draw {key} missing from learned panel")
                ref=target.pop(key)
                for k in ("year","draw","architecture","batch","baseline_placed"):
                    if ref[k]!=r[k]:raise ValueError(f"unpaired {k} for {key}")
                if r["policy"]!="learned_retention":raise ValueError("audit tool arm label changed")
                # The generic audit executable labels its first supplied mask 'learned_retention'.
                # Here that input was the hash-checked structural prospective mask.
                r["policy"]="structural_all_orders"
                if int(r["exits"]):raise ValueError(f"structural promise failed: {key}")
                if int(r["placed"])!=int(r["baseline_placed"])+int(r["entries"]):raise ValueError(key)
                out.append(r)
        if target or len(out)!=3*count:raise ValueError("annual draw count")
        write(args.results/(split+".csv"),out)
        for year in (2022,2023,2024):
            rows=[r for r in out if int(r["year"])==year]
            summary.append(dict(split=split,year=year,draws=len(rows),exits=sum(int(z["exits"]) for z in rows),
                                new_recipients=sum(int(z["entries"]) for z in rows),
                                course_switches=sum(int(z["course_switch"]) for z in rows),
                                school_switches=sum(int(z["school_switch"]) for z in rows)))
    write(args.results/"comparison-summary.csv",summary)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ("binary","masks","freeze","config","learned","private","results"):
        ap.add_argument("--"+key,type=Path,required=True)
    ap.add_argument("--workers",type=int,default=4)
    compare(ap.parse_args())
