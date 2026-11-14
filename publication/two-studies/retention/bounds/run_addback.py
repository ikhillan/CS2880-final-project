"""Add previously discarded hindsight gains and choose alternative drops on frozen design draws."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import copyfile
import subprocess

def run(args):
    y=str(args.year);orig=args.scratch.resolve()/y
    stage=args.scratch.resolve()/("addback-"+y);stage.mkdir(parents=True,exist_ok=True)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    prefix=str((args.late if y=="2024" else args.early).resolve()/y)
    tier=str(args.features.resolve()/(y+"-tiers"));register=str(args.features.resolve()/(y+"-register"));side=str(args.sides.resolve()/(y+".side"))
    if not (stage/"risk").is_file():
        subprocess.run([args.risk_bin,prefix,y,tier,register,side,stage/"risk","64","513"],check=True)
    for name in ("upper","required","forbidden"):copyfile(orig/name,stage/name)
    subprocess.run([args.bin,"core",prefix,y,tier,register,side,str(stage),"0","1","1","1","513"],check=True)
    orig_mask=(stage/"core").read_bytes();upper=(stage/"upper").read_bytes()
    (stage/"anchor").write_bytes(orig_mask)
    # One swap per positive hit. Alternative dropped course is chosen for candidates
    # outside the anchor by estimated baseline exposure on the first 64 design draws.
    active=upper
    (stage/"active").write_bytes(active)
    params=[prefix,y,tier,register,side,str(stage)]
    def apply(limit):
        def one(k):
            cmd=[args.bin,"prune",*params,str(k),str(args.workers),"1",str(limit),"513"]
            subprocess.run(cmd,check=True)
        with ThreadPoolExecutor(max_workers=args.workers)as pool:list(pool.map(one,range(args.workers)))
        ordered=sorted([r for k in range(args.workers) for r in csv.reader((stage/("receipt."+str(k))).open())],key=lambda r:int(r[2]))
        assert [int(r[2]) for r in ordered]==list(range(1,limit+1))
        return ordered
    pruning=[]
    for lim in [8,32,128,512,1024]:
      for it in range(12):
        rows=apply(lim)
        exits=sum(int(r[3]) for r in rows)
        if not exits:break
        masks=[(stage/("bad."+str(k))).read_bytes() for k in range(args.workers)]
        bad=[any(v) for v in zip(*masks)]
        removed=sum(a and b for a,b in zip(active,bad))
        assert not any(a and b for a,b in zip(orig_mask,bad)), "Modified anchor"
        if not removed:raise RuntimeError(f"no progress at {lim} {it}; original anchor may need certification")
        active=bytes(a and not b for a,b in zip(active,bad));(stage/"active").write_bytes(active)
        pruning.append((y,lim,it,exits,removed));print("addback",y,lim,it,exits,removed,flush=True)
      else:raise RuntimeError("Failed to converge")
    # count true gained rows against original upper via main-run summary:
    status=subprocess.check_output([args.bin,"status",*params,"0","1","1","1","513"],text=True)
    resultyear,_,upper_bound,_=map(int,status.strip().split(","))
    assert resultyear==args.year
    gains=upper_bound-sum(a and not b for a,b in zip(upper,active))
    with (out/(y+"-addback.csv")).open("w",newline="") as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(["year","draws","retention_lower","exits","entrants","course_switches"]);w.writerow([y,1024,gains,0,sum(int(r[4]) for r in rows),sum(int(r[5]) for r in rows)])
    with (out/(y+"-addback-draws.csv")).open("w",newline="") as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(["year","batch","draw","exits","entrants","switches","detailed_changes"]);w.writerows(rows)
    with (out/(y+"-addback-pruning.csv")).open("w",newline="") as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(["year","nested_draws","iteration","exits","edits_removed"]);w.writerows(pruning)
    print("verified",y,gains,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("bin","risk-bin","scratch","out","early","late","features","sides"):p.add_argument("--"+key,type=Path,required=True)
    for key in ("workers","year"):p.add_argument("--"+key,type=int,required=True)
    run(p.parse_args())
