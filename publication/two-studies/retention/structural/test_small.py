"""Stress the order-independent rule using independent synchronous, typed-seat DA."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import numpy as np


def check(args):
    public = args.publication.resolve()
    school = public / "research/observed/school"
    institution = public / "two-studies/institutional"
    sys.path.insert(0, str(institution))
    from test_institution import fixture, reference
    rng = random.Random(658017)
    totals = dict(markets=0, lotteries=0, changed_rows=0, switches=0, baseline_recipient_draws=0, exits=0)
    with tempfile.TemporaryDirectory() as folder:
        scratch = Path(folder)
        exe = scratch/"choose"
        subprocess.run(["c++","-O2","-std=c++17",f"-I{institution}",f"-I{school}",
                        Path(__file__).with_name("structural.cpp"),"-o",exe],check=True)
        for _ in range(args.markets):
            p=scratch/"fixture"
            fixture(rng,p)
            with p.with_suffix(".bin").open("rb") as f:
                def ints(n):
                    return np.fromfile(f,"<i4",int(n)).tolist()
                n,m,ne=ints(3)
                offsets=ints(n+1)
                course=ints(ne)
                ints(ne);ints(m);ints(n)
                disadvantaged,grades,current=ints(n),ints(n),ints(n)
            with p.with_suffix(".side").open("rb") as f:
                n_,m_,ne_,nlinks=ints(4)
                assert (n,m,ne) == (n_,m_,ne_)
                school_ids,course_grades=ints(m),ints(m)
                raw=np.array(ints(5*m)).reshape(m,5).tolist()
                caps=[q[:3]+[q[3]+q[4]] for q in raw]
                trans=[q[3]>0 for q in raw]
                honors=np.fromfile(f,"u1",n)
                flags=np.fromfile(f,"u1",ne)
                ints(ne)
                pie,exam=ints(ne),ints(ne)
                links=np.array(ints(4*nlinks)).reshape(nlinks,4).tolist()
            rows=[list(range(offsets[i],offsets[i+1])) for i in range(n)]
            owner=[i for i,row in enumerate(rows) for _ in row]
            masks=[np.zeros(ne,"u1"),np.zeros(ne,"u1")]
            for row in rows:
                fixed=[e for e in row if flags[e]&1]
                optional=[e for e in row if e not in fixed]
                k=rng.randrange(len(optional)+1)
                for mask in masks:
                    mask[fixed+rng.sample(optional,k)]=1
            for mask,label in zip(masks,["blind","rank"]):
                mask.tofile(scratch/f"fixture-{label}.mask")
            subprocess.run([exe,p,p.with_suffix(".side"),scratch/"selected",scratch/"stats",
                            "0","prospective"],check=True)
            chosen=np.fromfile(scratch/"selected","u1")
            totals["changed_rows"]+=sum(any(chosen[e]!=masks[0][e] for e in row) for row in rows)
            totals["markets"]+=1
            for turn in range(6):
                order=rng.sample(range(ne),ne)
                result=[reference(n,course,owner,rows,mask,flags,current,grades,links,caps,
                                  order,disadvantaged,honors,pie,exam,trans) for mask in [masks[0],chosen]]
                for old,new in zip(*result):
                    totals["baseline_recipient_draws"]+=old is not None
                    totals["exits"]+=old is not None and new is None
                    totals["switches"]+=old is not None and new is not None and school_ids[old[0]]!=school_ids[new[0]]
                totals["lotteries"]+=1
    assert totals["exits"]==0 and totals["switches"]>0
    print(json.dumps(totals,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--publication",type=Path,required=True)
    ap.add_argument("--markets",type=int,default=1000)
    check(ap.parse_args())
