"""Materialize a shortlist after the original search, audit every row, and digest the mask."""
import argparse
import csv
from pathlib import Path
import hashlib
import subprocess

def main(args):
    aggregate=[]
    for year in (2022,2023,2024):
        y=str(year);prefix=(args.late if year==2024 else args.early)/y
        for name,stage,ref in [("oracle",args.scratch/y,args.first),("addback",args.scratch/("addback-"+y),args.second)]:
            with (ref/(y+"-"+name+".csv")).open(newline="") as f:
                (expected,)=csv.DictReader(f)
            cmd=[args.binary,"score",prefix,y,args.features/(y+"-tiers"),args.features/(y+"-register"),
                 args.sides/(y+".side"),stage,"0","1","1","1","513"]
            fields=list(map(int,subprocess.check_output(cmd,text=True).strip().split(",")))
            rowyear,n,edges,changed,added,dropped,base,final,gains,losses=fields
            assert rowyear==year and final-base==gains-losses and gains==int(expected["retention_lower"])
            assert changed==added==dropped==gains and losses==0
            actual=(stage/"final.mask").read_bytes()
            assert len(actual)==edges
            aggregate.append([y,name,n,edges,changed,added,dropped,base,final,gains,losses,hashlib.sha256(actual).hexdigest()])
    with args.out.open("w",newline="") as stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(["year","method","pupils_checked","candidate_records","changed_rows","added_edges","dropped_edges",
                         "baseline_hits","final_hits","gains","losses","final_mask_sha256"])
        writer.writerows(aggregate)
    print([(r[0],r[1],r[9]) for r in aggregate])
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("binary","scratch","out","early","late","features","sides","first","second"):
        p.add_argument("--"+key,type=Path,required=True)
    main(p.parse_args())
