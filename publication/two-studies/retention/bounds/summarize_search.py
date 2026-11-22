"""Combine independently verified hindsight lower bounds; check every design-draw record."""
import argparse
import csv
from pathlib import Path

def table(path):
    with path.open(newline="") as f:return list(csv.DictReader(f))

def main(p):
    out=[]
    for year in (2022,2023,2024):
        y=str(year)
        result=[]
        for mode,folder in [("oracle",p.first),("addback",p.second)]:
            (row,)=table(folder/(y+"-"+mode+".csv"))
            series=table(folder/(y+"-"+mode+"-draws.csv"))
            assert len(series)==1024 and set(int(r["draw"]) for r in series)==set(range(1,1025))
            assert all(r["year"]==y and int(r["batch"])==513 and int(r["exits"])==0 for r in series)
            assert all(0<=int(r["entrants"])<=int(r["detailed_changes"]) for r in series)
            result.append((int(row["retention_lower"]),mode,sum(int(r["entrants"]) for r in series),
                           sum(int(r["switches"]) for r in series),
                           sum(int(r["switches"])!=0 for r in series)))
        best=max(result,key=lambda z:z[0])
        out.append([y,result[0][0],result[1][0],best[0],best[1],0,*best[2:]])
    with p.out.open("w",newline="") as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(["year","first_lower","addback_lower","retention_lower","selected","exits","entrants","course_switches","draws_with_switch"]);w.writerows(out)
    print("Verified designs:",out,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("first","second","out"):ap.add_argument("--"+k,type=Path,required=True)
    main(ap.parse_args())
