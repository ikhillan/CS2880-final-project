"""Evaluate the capacity-recovery frontier on the archived learned NYC panels."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frontier import capacity,solve

FRACTIONS=[0,25,50,75,100]


def run(split,base,learned,ev):
    with np.load(base/f'{split}.npz') as data:
        e,pref,seller,unif,low,y=[data[k] for k in ('e','pref','seller','uniform','lower','y')]
    src=base/f'{split}-scores.npz'
    if not src.exists():src=learned/'predictions'/src.name
    with np.load(src) as model:
        lp=model['context'];p=np.exp(lp.astype(float));outside=np.exp(model['context_outside'].astype(float))
    assert np.allclose(p.sum(2)+outside,1,atol=1e-5)
    meta=json.loads((base/f'{split}.json').read_text());keys=meta['keys'];n=len(keys)
    graphs={'uniform':unif<5,'old_anchor':np.stack([ev.make_mask(lp[t],e[t],keys[t],True) for t in range(n)])}
    graphs.update({str(fr):np.zeros_like(e) for fr in FRACTIONS})
    v0v1=[];target={fr:[] for fr in FRACTIONS}
    for t,key in enumerate(keys):
        top=ev.make_mask(lp[t],e[t],key,False)
        lo=capacity(top);hi=capacity(e[t]);v0v1.append([lo,hi])
        rng=ev.keyed_rng(key);ix=np.ix_(rng.permutation(64),rng.permutation(64))
        base_top=top[ix];prob=p[t][ix];feasible=e[t][ix]
        for fr in FRACTIONS:
            r=lo+(fr*(hi-lo)+99)//100;target[fr].append(r)
            g=base_top if r==lo else solve(prob,feasible,base_top,r)[0]
            graphs[str(fr)][t][ix]=g
    b=ev.da(pref,seller,graphs['0']);M=64;valid=y<M;records=[]
    for name,g in graphs.items():
        matches=ev.da(pref,seller,g);got=matches<M
        hits=valid & np.take_along_axis(g,np.minimum(y,M-1)[:,:,None],2)[:,:,0]
        mass=(p*g).sum((1,2));actual=got.sum(1)
        for t,(seed,design,month,borough) in enumerate(keys):
            cap=capacity(g[t]);lo,hi=v0v1[t]
            req=target[int(name)][t] if name.isdigit() else -1
            assert cap>=req and actual[t]<=cap<=hi and actual[t]>=hits[t].sum()
            if name=='old_anchor': assert hi==cap
            records.append(dict(split=split,seed=seed,design=design,month=month,borough=borough,policy=name,
                capacity=cap,requested=req,full_capacity=hi,n=M,low_n=int(low[t].sum()),placement=int(actual[t]),
                full=int(valid[t].sum()),hit=int(hits[t].sum()),predicted_mass=float(mass[t]),
                low_placement=int((got[t]&low[t]).sum()),lower_loss=int((~got[t]&low[t]&(b[t]<M)).sum()),
                lower_gain=int((got[t]&low[t]&(b[t]==M)).sum()),new=int((got[t]&~valid[t]).sum()),
                lost=int((~got[t]&valid[t]).sum())))
    return pd.DataFrame(records),meta


def summarize(rows,meta):
    result=[];pairs=dict(placement='n',capacity='n',hit='full',predicted_mass='n',low_placement='low_n',new='n',lost='n',lower_loss='low_n',lower_gain='low_n')
    for design,r in rows.groupby('design'):
        keys=sorted(set(zip(r.month,r.borough)))
        w=np.array([meta['weights'][b] for _,b in keys]);w/=w.sum()
        seeds=sorted(r.seed.unique());n=len(seeds)
        bootstrap=np.random.default_rng(441903).multinomial(n,np.repeat(1/n,n),size=5000)
        f=np.vstack([np.ones(n),bootstrap])
        mi=pd.MultiIndex.from_tuples([(s,m,b) for s in seeds for m,b in keys],names=['seed','month','borough'])
        arms={}
        for arm,df in r.groupby('policy'):
            frame=df.set_index(['seed','month','borough']).reindex(mi)
            names=list(dict.fromkeys(list(pairs)+list(pairs.values())))
            count=frame[names].to_numpy().reshape(n,len(keys),len(names))
            sums=np.einsum('bs,scm->bcm',f,count,optimize=True)
            vals=np.stack([100*((sums[:,:,names.index(numer)]/sums[:,:,names.index(denom)])@w) for numer,denom in pairs.items()],1)
            arms[arm]=vals
        for arm,a in arms.items():
            comparisons={'level':a,'minus_0':a-arms['0'],'minus_old_anchor':a-arms['old_anchor'],'minus_uniform':a-arms['uniform']}
            for contrast,s in comparisons.items():
                for j,metric in enumerate(pairs):
                    q=np.quantile(s[1:,j],[.025,.975])
                    result.append(dict(split=r.split.iloc[0],design=design,policy=arm,contrast=contrast,metric=metric,
                                       value=s[0,j],lo=q[0],hi=q[1]))
    return pd.DataFrame(result)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',required=True,type=Path);a.add_argument('--learned',required=True,type=Path)
    a.add_argument('--out',required=True,type=Path);args=a.parse_args()
    spec=importlib.util.spec_from_file_location('frozen_evaluator',args.learned/'evaluate.py')
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    summaries=[];counts=[]
    for split in ['summer','winter']:
        r,m=run(split,args.data,args.learned,ev);counts.append(r);summaries.append(summarize(r,m))
    args.out.mkdir(parents=True,exist_ok=True)
    pd.concat(counts).to_csv(args.out/'rental-counts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.concat(summaries).to_csv(args.out/'rental-summary.csv',index=False)
