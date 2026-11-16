"""Vacancy referrals for the archived calibrated rental panels."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def residual_assignment(edges,low,rng,priority):
    n=edges.shape[0];r=rng.permutation(n);j=rng.permutation(n)
    shuffled=edges[np.ix_(r,j)]
    # One more matched renter always dominates any possible change in lower-income count.
    w=shuffled*((n+1)+priority*low[r,None])
    i,t=linear_sum_assignment(w,maximize=True)
    use=shuffled[i,t];return r[i[use]],j[t[use]]

def panel(split,data,learned,methods):
    with np.load(data/f'{split}.npz') as arr:e,rank,priority,U,low,full=(arr[k] for k in ('e','pref','seller','uniform','lower','y'))
    source=data/f'{split}-scores.npz'
    if not source.exists():source=learned/'predictions'/source.name
    with np.load(source) as arr:score=arr['context']
    meta=json.loads((data/f'{split}.json').read_text());keys=meta['keys'];n=e.shape[1]
    masks={'uniform':U<5,'context':np.stack([methods.make_mask(s,edge,key,False) for s,edge,key in zip(score,e,keys)])}
    rows=[]
    for name,D in masks.items():
        start=methods.da(rank,priority,D); base=start<n
        vacancies=np.ones_like(start,bool);m,t=np.where(base);vacancies[m,start[m,t]]=False
        graph=(~base[:,:,None])&vacancies[:,None,:]&e
        assert not np.any(graph&D)
        residual_da=methods.da(rank,priority,graph)
        for arm in ['private','neutral','low_priority','residual_DA']:
            G=D.copy();pick=np.zeros_like(base);q=[]
            for s,g in enumerate(graph):
                rng=methods.keyed_rng([*keys[s],'vacancy-referrals'])
                if arm=='private':
                    i=np.flatnonzero(g.any(1));j=np.argmin(np.where(g[i],rank[s,i],255),axis=1)
                elif arm=='residual_DA':
                    i=np.flatnonzero(residual_da[s]<n);j=residual_da[s,i]
                else:
                    i,j=residual_assignment(g,low[s],rng,arm=='low_priority')
                    maximum=(maximum_bipartite_matching(csr_matrix(g),perm_type='column')>=0).sum()
                    assert len(i)==maximum
                pick[s,i]=True
                if len(i):
                    discard=np.argmax(np.where(D[s,i],rank[s,i].astype(int),-1),axis=1);G[s,i,discard]=False;G[s,i,j]=True
                    assert np.array_equal(G[s].sum(1),D[s].sum(1))
                q.append(len(i))
            new=methods.da(rank,priority,G);after=new<n
            loss=base&~after;gain=~base&after
            rank0=np.take_along_axis(rank,np.minimum(start,n-1)[:,:,None],axis=2)[:,:,0]
            rank1=np.take_along_axis(rank,np.minimum(new,n-1)[:,:,None],axis=2)[:,:,0]
            worse=base&(~after|(rank1>rank0))
            if arm!='private':assert not worse.any() and np.array_equal(gain,pick)
            for s,(seed,design,month,borough) in enumerate(keys):
                lo=low[s];rows.append(dict(split=split,design=design,month=month,borough=borough,seed=seed,baseline=name,method=arm,
                           renters=n,low_total=int(lo.sum()),initial=int(base[s].sum()),low_initial=int((base[s]&lo).sum()),
                           vacant=int(vacancies[s].sum()),eligible=int(graph[s].any(1).sum()),eligible_low=int((graph[s].any(1)&lo).sum()),selected=q[s],
                           gained=int(gain[s].sum()),lost=int(loss[s].sum()),worse=int(worse[s].sum()),gained_low=int((gain[s]&lo).sum()),
                           lost_low=int((loss[s]&lo).sum()),gain_fullunmatched=int((gain[s]&(full[s]>=n)).sum())))
    result=pd.DataFrame(rows);agg=[]
    for (design,base,arm),group in result.groupby(['design','baseline','method']):
        cell=group.groupby(['month','borough']).sum(numeric_only=True)
        w=np.array([meta['weights'][b] for _,b in cell.index]);w/=w.sum()
        agg.append(dict(split=split,design=design,baseline=base,method=arm,
            renters=int(group.renters.sum()),gain=int(group.gained.sum()),loss=int(group.lost.sum()),gained_low=int(group.gained_low.sum()),lost_low=int(group.lost_low.sum()),
            gain_fullunmatched=int(group.gain_fullunmatched.sum()),low_total=int(group.low_total.sum()),
            total_pp=float(100*np.sum(w*(cell.gained-cell.lost)/cell.renters)),low_pp=float(100*np.sum(w*(cell.gained_low-cell.lost_low)/cell.low_total)),
            eligible_low=int(group.eligible_low.sum()),eligible=int(group.eligible.sum())))
    return result,agg

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['data','learned','out']:p.add_argument(n,type=Path)
    a=p.parse_args();s=importlib.util.spec_from_file_location('rental_methods',a.learned/'evaluate.py')
    mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod)
    rows=[];summary=[]
    for split in ['summer','winter']:
        r,x=panel(split,a.data,a.learned,mod);rows.append(r);summary.extend(x)
    pd.concat(rows).to_csv(a.out/'rental.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(summary).to_csv(a.out/'rental-summary.csv',index=False)
