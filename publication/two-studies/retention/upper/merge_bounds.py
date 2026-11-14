"""Intersect profile-level prefix/capacity bounds and any applicant-level outer bound.

All parts must be computed for the same year, source data, and fixed row enumeration.
Only the aggregate summary is released.
"""
import argparse
import csv
from pathlib import Path
import subprocess


def merge(args):
    counts = []
    widths = None
    family_rows = []
    for directory in args.certificates:
        files = sorted(p for p in directory.glob(f"{args.year}-shard-*") if not p.suffix)
        if not files:
            raise ValueError(f"No certificate shards in {directory}")
        masks = []
        for file in files:
            lens = list(map(int,file.with_suffix(".sizes").read_text().splitlines()))
            if widths is None:
                widths=lens
            if widths != lens:
                raise ValueError("Enumeration differs between shards")
            bits=file.read_bytes()
            if len(bits)!=sum(widths) or set(bits)-{0, 1}:
                raise ValueError("Invalid certificate bits")
            masks.append(bits)
        common=bytes(all(parts) for parts in zip(*masks))
        family_rows.append(common)
        start=0;count=0
        for length in widths:
            count+=any(common[start:start+length])
            start+=length
        counts.append(count)
    feasible=bytes(all(parts) for parts in zip(*family_rows))
    ref=(args.scratch / f"{args.year}-row-ids")
    subprocess.run(list(map(str,[args.exe,args.prefix,args.year,args.tiers,args.register,args.side,
                                0,0,1,513,ref])),check=True)
    ids=list(map(int,ref.read_text().splitlines()))
    if len(ids)!=len(widths) or sorted(set(ids))!=ids:
        raise ValueError("Candidate row alignment failed")
    upper=args.other.read_bytes()
    if max(ids)>=len(upper) or set(upper)-{0,1}:
        raise ValueError("Other upper bound dimensions or values are invalid")
    start=0;can=[]
    for length in widths:
        can.append(any(feasible[start:start+length]))
        start+=length
    both=sum(y and upper[i] for i,y in zip(ids,can))
    with args.output.open("w") as target:
        w=csv.writer(target,lineterminator="\n")
        w.writerow(["year","possible_gains",*[f"relaxation_{i+1}_upper" for i in range(len(counts))],
                    "profile_intersection_upper","first_proposal_upper","joint_upper"])
        w.writerow([args.year,len(can),*counts,sum(can),sum(upper[i] for i in ids),both])
    print(f"Certified joint upper {args.year}: {both}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["exe","prefix","tiers","register","side","scratch","other","output"]:
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--certificates",type=Path,nargs="+",required=True)
    parser.add_argument("--year",type=int,required=True)
    merge(parser.parse_args())
