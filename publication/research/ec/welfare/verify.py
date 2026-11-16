"""Independent threshold LP check of the stable burden interval and feasibility of all stored witnesses."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, identity
from check import extract, inverse, low_burdens


def stable_lp(m):
    """Sparse Rothblum LP using inclusive prefix-sum auxiliary variables on both sides."""
    inst=m.inst
    b,s=np.where(inst.E)
    n=len(b)
    edges=np.full(inst.E.shape,-1,int)
    edges[b,s]=np.arange(n)
    prev_b=np.full(n,-1,int);prev_s=prev_b.copy()
    for owner in np.unique(b):
        js=np.flatnonzero(inst.E[owner]); js=js[np.argsort(inst.rank_b[owner,js])]
        e=edges[owner,js];prev_b[e[1:]]=e[:-1]
    for owner in np.unique(s):
        bs=np.flatnonzero(inst.E[:,owner]);bs=bs[np.argsort(inst.rank_s[bs,owner])]
        e=edges[bs,owner];prev_s[e[1:]]=e[:-1]
    # x, inclusive renter prefix p, inclusive listing prefix q. Prefix bounds
    # p,q<=1 are precisely degree upper bounds when x>=0.
    ids=np.arange(n);rr=[];cc=[];vals=[]
    def add(row,col,entry):
        rr.extend(row);cc.extend(col);vals.extend([entry]*len(row))
    for ix,prev in enumerate((prev_b,prev_s)):
        row=ids+ix*n
        add(row,ids,-1)
        add(row,ids+(1+ix)*n,1)
        active=prev>=0
        add(row[active],prev[active]+(1+ix)*n,-1)
    eq=coo_matrix((vals,(rr,cc)),shape=(2*n,3*n)).tocsr()
    # p_e+q_e-x_e >= 1 for each feasible pair, including unmatched endpoints.
    I=identity(n,format='csr')
    ub=hstack([I,-I,-I],format='csr')
    return b,s,ub,eq


def optimize(m,lp,t,mode):
    b,s,ub,eq=lp
    n=len(b)
    if not n:return 0
    cheap=m.low[b]&(m.burden[b,s]<=t)
    if not cheap.any():return 0
    obj=np.zeros(3*n);sgn=(-1 if mode=='max' else 1);obj[:n]=sgn*cheap.astype(float)
    res=linprog(obj,A_ub=ub,b_ub=-np.ones(n),A_eq=eq,b_eq=np.zeros(2*n),bounds=(0.,1.),method='highs-ds')
    assert res.success, res.message
    x=np.rint(res.x)
    assert np.abs(x-res.x).max()<1e-6
    assert np.max(ub@x+1.)<=1e-9 and np.max(np.abs(eq@x))<1e-9
    # Check the numerical LP dual certificate for this integer-valued optimum.
    y,zlo,zhi=res.ineqlin.marginals,res.lower.marginals,res.upper.marginals
    assert max(y)<=1e-8 and min(zlo)>=-1e-8 and max(zhi)<=1e-8
    dual_check=ub.T@y+eq.T@res.eqlin.marginals+zlo+zhi
    assert np.max(np.abs(dual_check-obj))<1e-7
    assert abs(-np.sum(y)+np.sum(zhi)-res.fun)<1e-7
    return int(round(sgn*res.fun))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    markets=extract(a.repo)
    from hiddenoptions.matching import blocking_mask
    results=json.loads((a.evidence/'results.json').read_text())
    with gzip.open(a.evidence/'witnesses.json.gz','rt') as f: rec=json.load(f)
    assert rec['market_order']==[list(m.key) for m in markets]
    scales=results['scopes']
    checks=dict(witnesses=0,market_assignments=0,which_solver='HiGHS dual simplex; stable matching LP with prefix variables',stable_thresholds={})
    for scope,assignments in rec['scopes'].items():
        lower=[];higher=0
        for m,out in zip(markets,assignments,strict=True):
            aa=np.asarray(out);bb=np.flatnonzero(aa>=0);assert m.inst.E[bb,aa[bb]].all()
            held=inverse(aa,m.inst.n_s)
            if scope in ('stable_min','stable_max','both_sides_ids','both_renter_ids'):
                assert np.array_equal(aa>=0,m.da>=0)
            if scope in ('stable_min','stable_max','both_sides_ids'):
                assert np.array_equal(held>=0,m.held>=0)
            if scope.startswith('stable'):
                assert not blocking_mask(m.inst,aa,held).any()
            if scope=='both_group_market_counts':
                for g in [m.low,~m.low]:assert np.sum(g&(aa>=0))==np.sum(g&(m.da>=0))
            lower.extend(low_burdens(m,aa));higher+=int(np.sum(~m.low&(aa>=0)))
            checks['market_assignments']+=1
        assert len(lower)==217 and higher==7755
        assert abs(np.quantile(lower,.25)-scales[scope]['q25'])<1e-14
        checks['witnesses']+=1
    thresholds=[(scope,mode,t) for scope,mode in [('stable_min','max'),('stable_max','min')]
                for t in [scales[scope]['q25'],np.nextafter(scales[scope]['q25'],-np.inf)]]
    values=np.zeros(len(thresholds),int)
    nontrivial=0; easy=0
    for m in markets:
        lp=stable_lp(m)
        for i,(scope,mode,t) in enumerate(thresholds):
            num=optimize(m,lp,t,mode)
            values[i]+=num
            direct=[int((low_burdens(m,a)<=t).sum()) for a in m.stable]
            assert num==(max(direct) if mode=='max' else min(direct))
            if (m.low[:,None]&(m.burden<=t)).any():nontrivial+=1
            else:easy+=1
    for i,(name,mode,t) in enumerate(thresholds):
        tag='at' if i%2==0 else 'below'
        checks['stable_thresholds'][name+'_'+tag]=int(values[i])
        assert values[i]==(55 if tag=='at' else 54)
    checks['nonzero_objective_market_LPs']=nontrivial
    checks['zero_objective_markets_at_thresholds']=easy
    a.out.write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
