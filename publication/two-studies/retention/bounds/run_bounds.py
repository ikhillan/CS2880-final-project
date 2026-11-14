"""Rebuild hindsight seat-retention bounds. Applicant-level scratch files stay private."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

def run_one(args, year):
    year=str(year)
    exe=args.binary.resolve()
    p=(args.numeric_2024 if year=="2024" else args.numeric).resolve()/year
    stage=args.scratch.resolve()/year
    stage.mkdir(parents=True,exist_ok=True)
    common=[str(p),year,str(args.features.resolve()/(year+"-tiers")),str(args.features.resolve()/(year+"-register")),str(args.sides.resolve()/(year+".side")),str(stage)]
    def series(phase,start=1,end=None,batch=513):
        end=args.draws if end is None else end
        def one(k):
            subprocess.run([exe,phase,*common,str(k),str(args.workers),str(start),str(end),str(batch)],check=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(one,range(args.workers)))
    def merge(name,op):
        subprocess.run([exe,"merge",stage/name,stage/name,str(args.workers),op],check=True)
    if not args.resume:
        series("train")
        for name,op in [("upper","and"),("required","or"),("forbidden","or")]:merge(name,op)
        (stage/"active").write_bytes((stage/"upper").read_bytes())
    cmd=[exe,"status",*common,"0","1","1","1","513"]
    _,candidates,upper,exact=map(int,subprocess.check_output(cmd,text=True).strip().split(","))
    print("upper",year,candidates,upper,exact,flush=True)
    progress=[]
    # Grow the certified training support, removing edits associated
    # with exits. End only after a single frozen menu passes every requested draw.
    checkpoints=[x for x in (8,32,128,512) if x<args.draws]+[args.draws]
    def audit(phase,limit):
        series(phase,end=limit)
        results=sorted((line for k in range(args.workers) for line in
                csv.reader((stage/("receipt."+str(k))).open())),key=lambda row:int(row[2]))
        assert len(results)==limit and [int(r[2]) for r in results]==list(range(1,limit+1))
        return results
    for limit in checkpoints:
        for it in range(12):
            results=audit("prune",limit)
            exits=sum(int(row[3]) for row in results)
            prev=(stage/"active").read_bytes()
            if not exits:
                print("zero",year,limit,flush=True)
                break
            merge("bad","or")
            bad=(stage/"bad").read_bytes();removed=sum(a and b for a,b in zip(prev,bad))
            if not removed:raise RuntimeError(f"Unresolved exits {year} {limit} {it}")
            progress.append([year,limit,it,exits,removed])
            (stage/"active").write_bytes(bytes(a and not b for a,b in zip(prev,bad)))
            print("prune",year,limit,it,exits,removed,flush=True)
        else:raise ValueError("No convergence")
    # Noncandidates have ones in 'upper'. Only actual removals count against positive-hit rows.
    removed=sum(a and not b for a,b in zip((stage/"upper").read_bytes(),(stage/"active").read_bytes()))
    lower=upper-removed
    assert lower>=exact and all(int(row[3])==0 for row in results)
    output=args.results.resolve();output.mkdir(parents=True,exist_ok=True)
    def save(file,header,rows):
        with (output/file).open("w",newline="") as stream:
            writer=csv.writer(stream,lineterminator="\n");writer.writerow(header.split(","));writer.writerows(rows)
    save(year+"-oracle-draws.csv","year,batch,draw,exits,entrants,switches,detailed_changes",results)
    save(year+"-oracle-pruning.csv","year,nested_draws,iteration,exits,edits_removed",progress)
    agg=[year,args.draws,candidates,upper,exact,lower,0,sum(int(r[4]) for r in results),sum(int(r[5]) for r in results)]
    save(year+"-oracle.csv","year,draws,rowwise_upper,forced_rejection_upper,full_contract_lower,retention_lower,exits,entrants,switches",[agg])
    print("final",agg,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("binary","numeric","numeric-2024","features","sides","scratch","results"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--year",required=True);p.add_argument("--workers",type=int,default=2);p.add_argument("--draws",type=int,default=1024)
    p.add_argument("--resume",action="store_true");a=p.parse_args();run_one(a,a.year)
