"""Independent integer assignment checks for the vacancy construction on finite instances."""
import argparse
from itertools import product
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import numpy as np
from repair import residual, match_vacancies, replace

def verify(root):
    sys.path.insert(0,str(root.resolve()))
    from run import Kernel
    rng=np.random.default_rng(886311);checks=optouts=0
    with tempfile.TemporaryDirectory() as d:
        solver=Kernel(d)
        # Enumerate capacities, applicant/receiver orders, and first-stage row masks
        # in the two-by-two model. Then include genuinely many-to-one random cases.
        for n,m in [(2,2),(5,3)]:
            cases=product(range(4),range(4),range(4),range(4)) if n==2 else range(15000)
            for c in cases:
                if n==2:
                    aa,pp,mask_code,cap_code=c
                    lists=np.array([[0,1] if (aa>>i)&1 else [1,0] for i in range(2)])
                    pri=np.array([[0,1] if (pp>>i)&1 else [1,0] for i in range(2)]).T
                    vis=np.zeros((2,2),bool);vis[np.arange(2),[mask_code&1,mask_code>>1]]=True
                    cap=(np.array([cap_code&1,cap_code>>1])+1).astype(np.int32)
                    full=np.ones((n,m),bool)
                else:
                    full=rng.random((n,m))<.68
                    lists=np.argsort(rng.random((n,m)),axis=1)
                    pri=np.argsort(rng.random((m,n)),axis=1).T
                    vis=np.zeros((n,m),bool)
                    for i in range(n):
                        opts=np.flatnonzero(full[i])
                        if len(opts):vis[i,rng.choice(opts)]=True
                    cap=rng.integers(1,3,size=m,dtype=np.int32)
                school=lists[np.take_along_axis(full,lists,axis=1)].astype(np.int32)
                deg=full.sum(1).astype(np.int32);ids=np.repeat(np.arange(n),deg).astype(np.int32)
                x=SimpleNamespace(n=n,m=m,degree=deg,offset=np.r_[0,deg.cumsum()].astype(np.int32),a=ids,school=school,
                                  priority=pri[ids,school].astype(np.int32))
                show=vis[ids,school];start=solver.solve(x,cap,(~show).astype(np.uint8),1)
                slack,es=residual(x,start,cap,show)
                selected,_=match_vacancies(x,slack,es,42)
                if len(es):
                    # All possible assignments extending original contracts by one for each originally free row.
                    options=[[-1]+es[ids[es]==i].tolist() for i in np.unique(ids[es])]
                    choices=product(*options);opt=0
                    for seq in choices:
                        sel=np.array([j for j in seq if j>=0],int)
                        if (np.bincount(school[sel],minlength=m)<=slack).all():opt=max(opt,len(sel))
                    assert len(selected)==opt
                for subset in product([False,True],repeat=len(selected)):
                    accepted=selected[np.array(subset,bool)]
                    if len(accepted):optouts+=1
                    vis2=replace(x,show,accepted,1)
                    after=solver.solve(x,cap,(~vis2).astype(np.uint8),1)
                    wit=start.copy();wit[ids[accepted]]=accepted
                    assert np.array_equal(wit>=0,after>=0) and (after[wit>=0]<=wit[wit>=0]).all()
                    checks+=1
    return dict(two_by_two_cases=256,random_many_to_one_cases=15000,subset_runs=checks,nonempty_subset_runs=optouts)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('university',type=Path);a=p.parse_args()
    print(json.dumps(verify(a.university),indent=2))
