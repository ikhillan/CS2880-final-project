"""Intersect per-pupil, per-configuration impossibility certificates over the declared draws."""
import argparse
import csv
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

def run(args):
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pieces = []
    prior_receipts=[]
    if args.resume_prefix:
        previous=sorted(p for p in args.resume_prefix.glob(f"{args.year}-shard-*") if not p.suffix)
        if not previous:raise ValueError("No previous certificates")
        covered=set()
        for j,old in enumerate(previous):
            line=old.with_suffix(".receipt").read_text().strip()
            year,batch,begin,end,stride,number=map(int,line.split(",")[:6])
            ids=set(range(begin,end+1,stride))
            if year!=args.year or batch!=args.batch or ids&covered or len(ids)!=number:
                raise ValueError("Previous receipt mismatch")
            covered|=ids
            new=args.scratch / f"{args.year}-shard-prior-{j}"
            for suffix in ("",".sizes",".receipt"):
                shutil.copyfile(Path(str(old)+suffix),Path(str(new)+suffix))
            prior_receipts.append((new,line))
        if covered!=set(range(1,args.start)):raise ValueError("Resume coverage mismatch")
    if args.start>1 and not prior_receipts:
        raise ValueError("Need earlier certified draws for a partial start")
    for worker in range(min(args.workers,args.draws-args.start+1)):
        path = args.scratch / f"{args.year}-shard-{worker}"
        incidence = [args.exe,args.prefix,args.year,args.tiers,args.register,args.side,args.start+worker,
                     args.draws,args.workers,args.batch,path]
        if args.veto:incidence.append("veto")
        elif args.hall:incidence.append("hall")
        pieces.append((path,list(map(str,incidence))))
    def collect(piece):
        receipt=subprocess.check_output(piece[1], text=True).strip()
        piece[0].with_suffix(".receipt").write_text(receipt+"\n")
        return receipt
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        receipts = list(pool.map(collect,pieces))
    receipts.extend(line for _,line in prior_receipts)
    pieces.extend((path,[]) for path,_ in prior_receipts)
    sizes = (pieces[0][0].with_suffix(".sizes")).read_text()
    lengths = list(map(int,sizes.splitlines()))
    surviving = [1]*sum(lengths)
    for path,_ in pieces:
        assert (path.with_suffix(".sizes")).read_text() == sizes
        shard=path.read_bytes()
        assert len(shard)==len(surviving) and set(shard)<={0,1}
        for j,value in enumerate(shard):surviving[j] &= value
    pos=total=0
    can_gain=[]
    for length in lengths:
        can_gain.append(any(surviving[pos:pos+length]))
        total += can_gain[-1]
        pos+=length
    if args.intersection:
        mapping=args.scratch / "candidate-index"
        subprocess.run(list(map(str,[args.exe,args.prefix,args.year,args.tiers,args.register,args.side,
                                     0,0,1,args.batch,mapping])),check=True)
        targets=list(map(int,mapping.read_text().splitlines()))
        assert len(targets)==len(can_gain) and targets==sorted(set(targets))
        other = args.intersection.read_bytes()
        assert len(other)>max(targets) and set(other)<={0,1}
        together=sum(both and other[k] for k,both in zip(targets,can_gain))
        with (args.out/f"intersection-{args.year}.csv").open("w") as stream:
            w=csv.writer(stream,lineterminator="\n")
            w.writerow(["year","prefix_draws","candidate_rows","prefix_upper","other_upper","intersection_upper"])
            w.writerow([args.year,args.draws,len(can_gain),total,sum(other[k] for k in targets),together])
    draws = sum(int(line.split(",")[5]) for line in receipts)
    assert draws==args.draws
    with (args.out/f"prefix-upper-{args.year}.csv").open("w") as file:
        w=csv.writer(file,lineterminator="\n")
        w.writerow("year,batch,draws,universally_fixed_prefix_rows,universally_fixed_prefix_courses,unconstrained_new_hits,retention_upper,excluded_hits,positive_menu_profiles_per_row_sum,surviving_menu_profiles_per_row_sum".split(","))
        row=receipts[0].split(",")
        w.writerow([args.year,args.batch,draws,row[6],row[7],len(lengths),total,
                    len(lengths)-total,sum(lengths),sum(surviving)])
    print(args.year, draws, len(lengths), total, flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ["exe","prefix","tiers","register","side","scratch","out"]:
        ap.add_argument("--"+key, type=Path, required=True)
    ap.add_argument("--year",type=int,required=True)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--veto",action="store_true",help="Also forbid options ahead of every feasible seat that necessarily displace a forced incumbent")
    ap.add_argument("--hall",action="store_true",help="Also eliminate edges impossible in any capacitated assignment of all baseline recipients")
    ap.add_argument("--start",type=int,default=1)
    ap.add_argument("--resume-prefix",type=Path,help="Verified same-design certificate shards covering exactly all preceding draws")
    ap.add_argument("--draws",type=int,default=1024)
    ap.add_argument("--batch",type=int,default=513)
    ap.add_argument("--intersection",type=Path,help="Optional independently certified 0/1 outer bound over all changed rows")
    args=ap.parse_args()
    if not 1<=args.workers or not 1<=args.start<=args.draws:ap.error("workers and draws must be positive")
    run(args)
