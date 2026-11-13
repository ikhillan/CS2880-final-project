"""Replay quotas, DA allocations and lower-income displacement for frozen policies."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def keyed_rng(key):
    h=hashlib.sha256(('cs2880-anchor-v1|'+json.dumps(list(key),separators=(',',':'))).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:16],'little'))


def anchor_mask(score,e,quota=5):
    """Input coordinates should already be randomly permuted by a market-specific key."""
    n,m=e.shape
    objectives=np.zeros((n,m+n),float);objectives[:,:m]=-1e4
    if e.any():
        z=score[e].astype(float);span=z.max()-z.min(); unit=(z-z.min())/span if span>0 else np.zeros_like(z)
        objectives[:,:m][e]=(min(n,m)+1)+unit
    rows,cols=linear_sum_assignment(objectives,maximize=True)
    anchor=np.zeros_like(e);real=cols<m;anchor[rows[real],cols[real]]=True
    assert np.all(~anchor|e)
    rank=np.argsort(np.argsort(-np.where(e&~anchor,score,-np.inf),axis=1,kind='stable'),axis=1)
    return anchor | (e & (rank<np.maximum(0,quota-anchor.sum(1)[:,None])))


def make_mask(score,e,key,anchored,quota=5):
    rng=keyed_rng(key); rows=rng.permutation(e.shape[0]);cols=rng.permutation(e.shape[1]);index=np.ix_(rows,cols)
    x=score[index];allowed=e[index]
    if anchored:
        mask=anchor_mask(x,allowed,quota)
    else:
        order=np.argsort(np.argsort(-np.where(allowed,x,-np.inf),axis=1,kind='stable'),axis=1)
        mask=(order<quota)&allowed
    result=np.zeros_like(e);result[index]=mask
    assert np.array_equal(result.sum(1),np.minimum(e.sum(1),quota))
    return result


def da(pref,seller,graph):
    """Parallel rounds of renter proposals. Returns [market,renter] partner, or n_listing."""
    batch,n,m=graph.shape
    degree=graph.sum(2);limit=int(degree.max(initial=0));prefs=np.argsort(np.where(graph,pref,255),axis=2)[:,:,:limit]
    block,row=np.indices((batch,n));ptr=np.zeros((batch,n),int);match=np.full((batch,n),m,int)
    held=np.full((batch,m),n,int);best=np.full((batch,m),128,int)
    while True:
        k,r=np.nonzero((match==m)&(ptr<degree))
        if not len(k):break
        j=prefs[k,r,ptr[k,r]];ptr[k,r]+=1
        ranks=seller[k,r,j];np.minimum.at(best,(k,j),ranks)
        win=ranks==best[k,j];k,r,j=k[win],r[win],j[win];old=held[k,j];displaced=old<n
        match[k[displaced],old[displaced]]=m;match[k,r]=j;held[k,j]=r
    # Independent vector condition for feasible/one-to-one, no blocking pair on the shown graph.
    safe=np.minimum(match,m-1); assert np.all((match==m)|graph[block,row,safe]); assert (np.sum(held<n,1)==np.sum(match<m,1)).all()
    current_r=np.where(match<m,pref[block,row,safe],128)
    assert not np.any(graph & (pref<current_r[:,:,None]) & (seller<best[:,None,:]))
    return match


def evaluate(split,base,dest):
    with np.load(base/f'{split}.npz') as a: e,pref,seller,u,ur,tr,low,y=(a[k] for k in ('e','pref','seller','u','uniform','transparent','lower','y'))
    meta=json.loads((base/f'{split}.json').read_text()); keys=meta['keys'];rows=[]
    with np.load(base/f'{split}-scores.npz') as a:
        scores={k:a[k] for k in ('context','pointwise')}; outside={k:a[k+'_outside'] for k in scores}
    with np.load(base/f'{split}-su.npz') as a: su=a['mask']
    U=ur<5; N=len(U); groups=np.arange(N)[:,None];indices=np.arange(64)[None,:]
    baseline=da(pref,seller,U);assert np.array_equal(da(pref,seller,e),y)
    B=baseline<64;valid=y<64;safe=np.minimum(y,63);nll={}
    for name,score in scores.items():
        nll[name]=-np.where(valid,score[groups,indices,safe],outside[name]); assert np.isfinite(nll[name]).all()
    policies=('uniform','transparent','transparent_anchor','pointwise','pointwise_anchor','context','context_anchor','su')
    for policy in policies:
        name=policy.split('_')[0];anchored=policy.endswith('_anchor')
        if policy=='uniform': mask=U
        elif policy=='transparent': mask=tr<5
        elif policy=='su': mask=su
        else:
            s=u if name=='transparent' else scores[name]
            mask=np.stack([make_mask(x,ei,key,anchored) for x,ei,key in zip(s,e,keys)])
        assert not np.any(mask&~e) and np.array_equal(mask.sum(2),np.minimum(e.sum(2),5))
        for rollout in ('all','higher'):
            graph=mask if rollout=='all' else np.where(low[:,:,None],U,mask)
            if rollout=='higher': assert np.array_equal(graph[low],U[low])
            matches=baseline if policy=='uniform' else da(pref,seller,graph);M=matches<64
            target_in=valid & graph[groups,indices,safe];lost=low&B&~M;gained=low&~B&M
            # Every retained full-market partner is one certified unit of total DA cardinality.
            assert np.all(M.sum(1)>=target_in.sum(1))
            for t,(seed,design,month,borough) in enumerate(keys):
                rows.append(dict(split=split,seed=seed,design=design,month=month,borough=borough,policy=policy,rollout=rollout,
                    lower_n=int(low[t].sum()),higher_n=int((~low[t]).sum()),lower_zero=int((low[t]&~e[t].any(1)).sum()),
                    matched_total=int(M[t].sum()),matched_lower=int((M[t]&low[t]).sum()),matched_higher=int((M[t]&~low[t]).sum()),
                    baseline_matched_lower=int((B[t]&low[t]).sum()),baseline_matched_total=int(B[t].sum()),
                    joint_loss=int(lost[t].sum()),joint_gain=int(gained[t].sum()),full_matched=int(valid[t].sum()),
                    full_partner_in_menu=int(target_in[t].sum()),target_nll_sum=float(nll[name][t].sum()) if name in nll else None))
    dest.mkdir(exist_ok=True,parents=True)
    with gzip.open(dest/f'{split}-counts.csv.gz','wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(split,len(rows),'validated allocations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',nargs='+',default=['cal','summer','winter']);a=p.parse_args()
    for split in a.split:evaluate(split,a.data,a.out)
