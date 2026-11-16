"""Minimum probability-mass loss for a target capacitated assignment witness on PAES."""
import argparse
from pathlib import Path
import sys
import tempfile
import json
import time
import numpy as np
import pandas as pd
from ortools.graph.python.min_cost_flow import SimpleMinCostFlow

SCALE=10**9

def model_menu(m,features,fit,name):
    with np.load(fit/f'{name}-ensemble.npz') as z:
        prob=z['p'][m.a,np.load(features/'test-col.npy')]
    rng=np.random.default_rng(118844)
    order=np.lexsort((rng.permutation(len(prob)),-prob,m.a))
    ranks=np.empty(len(prob),np.uint8);ranks[order]=(np.arange(len(prob))-m.offset[m.a[order]]).astype(np.uint8)
    top=ranks<5
    worst=np.full(m.n,-1,np.int32)
    final=np.flatnonzero(ranks==np.maximum(0,np.minimum(m.degree,5)-1)[m.a])
    worst[m.a[final]]=final
    assert len(final)==sum(m.degree>0)
    return prob,top,worst

def witness(m,cost,target,permutation):
    n,mcol=m.n,m.m
    s=n+mcol;t=s+1
    f=SimpleMinCostFlow()
    f.add_arcs_with_capacity_and_unit_cost(np.full(n,s,np.int32),np.arange(n,dtype=np.int32),np.ones(n,np.int32),np.zeros(n,np.int64))
    idx=np.arange(len(cost),dtype=np.int32)
    if permutation:idx=np.random.default_rng(48124+permutation).permutation(idx)
    edge_arcs=f.add_arcs_with_capacity_and_unit_cost(m.a[idx],m.school[idx]+n,np.ones(len(idx),np.int32),cost[idx])
    f.add_arcs_with_capacity_and_unit_cost(n+np.arange(mcol,dtype=np.int32),np.full(mcol,t,np.int32),m.cap,np.zeros(mcol,np.int64))
    f.set_node_supply(s,target);f.set_node_supply(t,-target)
    status=f.solve()
    if status!=f.OPTIMAL:raise RuntimeError(f'flow failed: {status}')
    selected=idx[f.flows(edge_arcs)>0];assert len(selected)==target
    assert len(np.unique(m.a[selected]))==target
    assert (np.bincount(m.school[selected],minlength=mcol)<=m.cap).all()
    assert int(cost[selected].sum())==f.optimal_cost()
    return selected,int(f.optimal_cost())

def main(args):
    sys.path.insert(0,str(args.previous.resolve()))
    from run import Market,Kernel
    from capacity import visible_capacity
    m=Market(args.inputs,2026,positive_scores=True);both=np.ones(len(m.a),bool)
    maxcap=visible_capacity(m,m.cap,both)
    rows=[];sensitivity=[]
    with tempfile.TemporaryDirectory() as temp:
        solver=Kernel(temp)
        full=solver.solve(m,m.cap,np.zeros(len(m.a),np.uint8),1)
        for name in ('informed','blind'):
            prob,top,worst=model_menu(m,args.features,args.fit,name)
            original=solver.solve(m,m.cap,(~top).astype(np.uint8),1)
            start=visible_capacity(m,m.cap,top)
            cut=prob[worst[m.a]]
            penalty=np.maximum(0.,cut-prob)
            cost=np.rint(penalty*SCALE).astype(np.int64)
            qr=np.minimum(5,m.degree)
            for fraction,perm in [(f,0) for f in (0,.25,.5,.75,1)]+[(1,p) for p in (1,2)]:
                target=start+int(np.ceil(fraction*(maxcap-start)))
                t0=time.perf_counter()
                mask=top.copy();price=0
                if fraction:
                    sel,price=witness(m,cost,target,perm);added=sel[~top[sel]]
                    mask[worst[m.a[added]]]=False;mask[added]=True
                    assert mask[sel].all()
                assert np.array_equal(np.bincount(m.a[mask],minlength=m.n),qr)
                actual=visible_capacity(m,m.cap,mask)
                assert actual>=target
                if fraction==1:assert actual==maxcap
                loss=prob[top].sum()-prob[mask].sum()
                assert abs(loss-price/SCALE)<(target+1)/(2*SCALE)+2e-9
                rank=(~mask).astype(np.uint8);y=solver.solve(m,m.cap,rank,1);m.verify(m.cap,rank,1,y)
                a=(full>=0);b=(original>=0);c=(y>=0)
                restored=int((a&c).sum());entered=int((~a&c).sum());assert restored+entered==c.sum()
                row=dict(model=name,fraction=fraction,permutation=perm,target=target,actual_capacity=actual,full_capacity=maxcap,
                         unedited_capacity=start,cost_integer=price,pred_mass=prob[mask].sum(),mass_loss=loss,
                         edits=int((mask&~top).sum()),DA=int(c.sum()),full_matched=int(a.sum()),
                         reference_recovered=int(mask[full[a]].sum()),historical_covered=int(mask[m.recorded[m.recorded>=0]].sum()),
                         retained=restored,entrants=entered,exits=int((a&~c).sum()),
                         lost_vs_original=int((b&~c).sum()),gained_vs_original=int((~b&c).sum()),
                         worse_vs_original=int((b&((y<0)|(y>original))).sum()),
                         better_vs_original=int((c&((original<0)|(y<original))).sum()))
                rows.append(row);print(name,fraction,perm,actual,row['DA'],round(loss,3),round(time.perf_counter()-t0,1),flush=True)
                if name=='informed' and fraction in [0,1] and not perm:
                    for rate in (0,.25,.5,.75,1):
                        cap=m.base+np.floor(rate*(m.cap-m.base)).astype(np.int32)
                        v=solver.solve(m,cap,rank,1);m.verify(cap,rank,1,v)
                        sensitivity.append(dict(full_frontier=int(fraction),overbook_fraction=rate,seats=int(cap.sum()),placed=int((v>=0).sum())))
    args.out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(rows).to_csv(args.out/'frontier.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(args.out/'capacity-sensitivity.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('previous','inputs','features','fit','out'):p.add_argument(field,type=Path)
    main(p.parse_args())
