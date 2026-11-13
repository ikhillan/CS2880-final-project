"""Independent batch-round DA and analytic expected uniform coverage for both held-out cohorts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def read_numeric(path):
    with path.open('rb') as f:
        def vec(n,t='<i4'):return np.fromfile(f,dtype=t,count=n)
        n,m,l=vec(3)
        n,m,l=map(int,(n,m,l))
        arrays=[vec(length) for length in [n+1,l,l,m,*([n]*6)]]
        at,j,prior,cap,off,sen,grade,ret,app,deg=arrays
        blind,rank=vec(l,'<f8'),vec(l,'<f8')
        assert not f.read(1)
    return at,j,prior,cap,off,ret,deg,blind,rank
def round_da(at,j,prior,cap,shown):
    n=len(at)-1
    own=np.repeat(np.arange(n),np.diff(at));visible=np.flatnonzero(shown)
    sizes=np.bincount(own[visible],minlength=n);nextp=np.r_[0,np.cumsum(sizes)[:-1]]
    stop=nextp+sizes;held=np.array([],dtype=int);match=np.full(n,-1,int)
    for r in range(len(j)+1):
        active=np.flatnonzero((match<0)&(nextp<stop))
        if not len(active): return match,r
        candidates=np.r_[held,visible[nextp[active]]];nextp[active]+=1
        order=candidates[np.lexsort((prior[candidates],j[candidates]))]
        schools,start,size=np.unique(j[order],return_index=True,return_counts=True)
        within=np.arange(len(order))-np.repeat(start,size)
        held=order[within<cap[j[order]]]
        match.fill(-1);match[own[held]]=j[held]
    raise RuntimeError('round bound')
def lower_upper(at,j,ret,deg,score,k):
    n=len(at)-1;own=np.repeat(np.arange(n),np.diff(at));eligible=ret[own]!=j
    ind=np.flatnonzero(eligible)
    # Classify ties, then upper/lower bounds on coverage (placements require exact tie resolution).
    order=ind[np.lexsort((ind,-score[ind],own[ind]))]
    orderpos=np.arange(len(order))-np.repeat(np.r_[0,np.cumsum(deg)[:-1]],deg)
    minscore=np.full(n,-np.inf);rowcut=orderpos==(k-1)
    minscore[own[order[rowcut]]]=score[order[rowcut]]
    up=(~eligible)|(score>=minscore[own])|(deg[own]<=k)
    lo=(~eligible)|(score>minscore[own])|(deg[own]<=k)
    no_ties=np.bincount(own[up],minlength=n)==np.minimum(deg,k)+(ret>=0)
    lo[no_ties[own]]=up[no_ties[own]]
    return lo,up
def main(a):
    df=pd.read_csv(a.counts);records=[]
    for year in (2020,2021):
        at,j,prior,cap,off,ret,deg,blind,rank=read_numeric(a.scratch/f'{year}.bin')
        n=len(at)-1;own=np.repeat(np.arange(n),np.diff(at))
        full,r=round_da(at,j,prior,cap,np.ones(len(j),bool))
        rec=dict(year=year,rounds_full=int(r),full_agreement=int((full==off).sum()),analytic_uniform={},cutoff_tie_checks={})
        f=df[(df.year==year)&(df.group=='all')]
        assert rec['full_agreement']==f.query('policy=="full"').identical_official.item()
        assert (full>=0).sum()==f.query('policy=="full"').placed.item()
        for k in (1,2,3,5):
            p=np.where(j==ret[own],1,np.minimum(1,k/np.maximum(1,deg[own])))
            c=dict()
            for key,target in [('full_covered',full),('official_covered',off)]:
                expected=p[target[own]==j].sum()
                obs=f[(f.k==k)&(f.policy=='uniform')][key]
                z=(expected-obs.mean())/(obs.std()/np.sqrt(len(obs)))
                assert abs(z)<5
                c[key]=dict(expected_count=float(expected),mean_count=float(obs.mean()),mc_z=float(z))
            rec['analytic_uniform'][k]=c
        # C++ replays random deterministic tie breaks. Every independently computed cutoff
        # permits some ambiguity. Form lower/upper coverage bounds and check observed counts.
        for name,s in [('blind',blind),('rank',rank)]:
            lo,up=lower_upper(at,j,ret,deg,s,5)
            rec['cutoff_tie_checks'][name]=int(np.unique(own[lo!=up]).size)
            d=f[(f.k==5)&(f.policy==name)].iloc[0]
            for key,target in [('full_covered',full),('official_covered',off)]:
                target_edges=target[own]==j
                assert (target_edges&lo).sum()<=d[key]<=(target_edges&up).sum()
        first=np.zeros(len(j),bool);inrow=np.arange(len(j))-at[own]
        first[(inrow<5)|(j==ret[own])]=True
        out,r=round_da(at,j,prior,cap,first);rec['rounds_first5']=r
        assert (out>=0).sum()==f.query('policy=="first" and k==5').placed.item()
        assert (out==off).sum()==f.query('policy=="first" and k==5').identical_official.item()
        if a.audit_dir:
            rec['all_partners_identical']={}
            for name in ('full','first','blind','rank'):
                mask=np.fromfile(a.audit_dir/f'{year}-{name}.mask',dtype=np.uint8).astype(bool)
                counterpart=np.fromfile(a.audit_dir/f'{year}-{name}.partners',dtype='<i4')
                destination,r=round_da(at,j,prior,cap,mask)
                np.testing.assert_array_equal(destination,counterpart)
                rec['all_partners_identical'][name]=int(len(destination))
                if name=='full':
                    rec['official_agreement_matched_only']=float(np.mean(destination[off>=0]==off[off>=0]))
                    rec['official_agreement_outside_only']=float(np.mean(destination[off<0]==off[off<0]))
        records.append(rec)
    a.out.write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit-dir',type=Path);p.add_argument('--scratch',type=Path,required=True);p.add_argument('--counts',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
