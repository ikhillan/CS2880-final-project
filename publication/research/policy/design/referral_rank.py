"""Rank-capped and rank-minimizing optimal vacancy referrals on one test cohort."""
import argparse
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from repair import learned,clear,residual,match_vacancies,replace

def best_rank(m,edges,seats,v):
    if not v:return edges[:0]
    applicants,ids=np.unique(m.a[edges],return_inverse=True);prog,pr=np.unique(m.school[edges],return_inverse=True)
    col=np.tile(np.arange(len(edges)),2);row=np.r_[ids,len(applicants)+pr]
    A=csr_matrix((np.ones(len(col)),(row,col)),shape=(len(applicants)+len(prog),len(edges)))
    b=np.r_[np.ones(len(applicants)),seats[prog]];cost=m.r[edges]
    lp=linprog(cost,A_ub=A,b_ub=b,A_eq=np.ones((1,len(edges))),b_eq=[v],bounds=(0,1),method='highs')
    assert lp.success and np.allclose(lp.x,np.rint(lp.x),atol=1e-8)
    selected=edges[lp.x>.5];assert len(selected)==v and m.r[selected].sum()==round(lp.fun)
    return selected

def main(a):
    sys.path.insert(0,str(a.university.resolve()))
    from run import Market,Kernel
    m=Market(a.inputs,2026,positive_scores=True);show=learned(m,a.features,a.fit,'informed')<5;rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        solver=Kernel(tmp);start=clear(m,solver,m.cap,show);seats,edges=residual(m,start,m.cap,show)
        for lim in range(1,21):
            options=edges[m.r[edges]<=lim];chosen,_=match_vacancies(m,seats,options,2880781)
            optim=best_rank(m,options,seats,len(chosen));r=m.r[optim]
            final=clear(m,solver,m.cap,replace(m,show,optim,5))
            wit=start.copy();wit[m.a[optim]]=optim
            assert np.array_equal(wit>=0,final>=0) and (final[wit>=0]<=wit[wit>=0]).all()
            observed=m.r[final[(start<0)&(final>=0)]]
            rows.append(dict(max_rank=lim,applicants_with_choice=len(np.unique(m.a[options])),maximum=len(r),
                    minimum_total_rank=int(r.sum()),mean_rank=float(np.mean(r)) if len(r) else None,
                    median_rank=float(np.median(r)) if len(r) else None,
                    actual_mean=float(np.mean(observed)) if len(r) else None,
                    actual_median=float(np.median(observed)) if len(r) else None))
    pd.DataFrame(rows).to_csv(a.out/'referral-ranks.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['university','inputs','features','fit','out']:p.add_argument(n,type=Path)
    main(p.parse_args())
