"""Coordinated referrals to residual seats under a hard display quota."""
from pathlib import Path
import argparse
import copy
import json
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

SEED=2880781


def residual(m,baseline,cap,visible):
    placed=baseline[baseline>=0]
    slack=cap-np.bincount(m.school[placed],minlength=m.m).astype(np.int32)
    assert (slack>=0).all()
    edges=np.flatnonzero((baseline[m.a]<0)&(slack[m.school]>0))
    assert not np.any(visible[edges]), 'baseline blocking pair'
    return slack,edges


def match_vacancies(m,slack,edges,seed):
    applicants=np.unique(m.a[edges]); programs=np.unique(m.school[edges])
    if len(edges)==0:
        return edges,dict(eligible=0,reachable_seats=0,candidate_edges=0,cut=0,hall_left=0,hall_neighbor_seats=0)
    rng=np.random.default_rng(seed)
    applicants=applicants[rng.permutation(len(applicants))];programs=programs[rng.permutation(len(programs))]
    aindex=np.full(m.n,-1,int);aindex[applicants]=np.arange(len(applicants))+1
    pindex=np.full(m.m,-1,int);pindex[programs]=np.arange(len(programs))+1+len(applicants)
    l=aindex[m.a[edges]];r=pindex[m.school[edges]]
    sink=len(applicants)+len(programs)+1
    rows=np.r_[np.zeros(len(applicants),int),l,pindex[programs]]
    cols=np.r_[aindex[applicants],r,np.full(len(programs),sink)]
    # Unit source -> applicant, large internal capacities express Hall's witness.
    caps=np.r_[np.ones(len(applicants),int),np.full(len(edges),len(applicants)+1,int),slack[programs]]
    graph=csr_matrix((caps,(rows,cols)),shape=(sink+1,sink+1))
    result=maximum_flow(graph,0,sink,method='dinic')
    take=np.asarray(result.flow[l,r]).ravel()
    assert np.isin(take,[0,1]).all()
    selected=edges[take==1]
    cut_res=(graph-result.flow).tocsr();cut_res.eliminate_zeros()
    reach=np.zeros(sink+1,bool)
    reach[breadth_first_order(cut_res,0,directed=True,return_predecessors=False)]=True
    cut=int(caps[reach[rows]&~reach[cols]].sum())
    left=int(reach[aindex[applicants]].sum());right=slack[programs][reach[pindex[programs]]].sum()
    assert not reach[sink] and cut==result.flow_value==len(selected)==len(applicants)-left+right
    return selected,dict(eligible=len(applicants),reachable_seats=int(slack[programs].sum()),candidate_edges=len(edges),
                         cut=cut,hall_left=left,hall_neighbor_seats=int(right))


def uncoordinated(m,edges):
    _,first=np.unique(m.a[edges],return_index=True)
    return edges[first]


def serial(m,slack,edges):
    if not len(edges):return edges
    people,start=np.unique(m.a[edges],return_index=True)
    rng=np.random.default_rng(SEED);order=rng.permutation(len(people));end=np.r_[start[1:],len(edges)]
    seats=slack.copy();matched=[]
    for t in order:
        for e in edges[start[t]:end[t]]:
            j=m.school[e]
            if seats[j]>0:
                seats[j]-=1;matched.append(e);break
    return np.asarray(matched,dtype=np.int32)


def replace(m,visible,selected,k):
    revised=visible.copy()
    edited=m.a[selected]
    assert len(np.unique(edited))==len(edited) and (m.degree[edited]>k).all() and not revised[selected].any()
    last=np.full(m.n,-1,int);bounds=np.flatnonzero(visible)
    np.maximum.at(last,m.a[bounds],bounds)
    assert (last[edited]>=0).all()
    revised[last[edited]]=False;revised[selected]=True
    assert np.array_equal(np.bincount(m.a[revised],minlength=m.n),np.minimum(m.degree,k))
    return revised


def clear(m,kernel,cap,visible):
    order=np.ascontiguousarray(~visible,dtype=np.uint8)
    result=kernel.solve(m,cap,order,1);m.verify(cap,order,1,result)
    return result


def independent(m,visible,cap,out):
    e=np.flatnonzero(visible);sub=copy.copy(m)
    sub.a=m.a[e];sub.school=m.school[e];sub.priority=m.priority[e]
    sub.degree=np.bincount(sub.a,minlength=m.n)
    sub.offset=np.r_[0,sub.degree.cumsum()]
    other=sub.batch(cap);placed=other>=0;other[placed]=e[other[placed]]
    assert np.array_equal(other,out)


def learned(m,features,fit,name):
    with np.load(fit/f'{name}-ensemble.npz') as f:prob=f['p']
    cols=np.load(features/'test-col.npy');p=prob[m.a,cols]
    tie=np.random.default_rng(118844).permutation(len(cols));ix=np.lexsort((tie,-p,m.a))
    order=np.empty(len(cols),np.uint8)
    order[ix]=(np.arange(len(ix))-m.offset[m.a[ix]]).astype(np.uint8)
    return order


def metrics(m,reference,original,after,new,selected):
    was=original>=0;now=after>=0;oldfull=reference>=0;recorded=m.recorded>=0
    improved=was&now&(after<original);worse=was&(~now|(after>original))
    entrants=~was&now
    return dict(places=int(now.sum()),gross_loss=int((was&~now).sum()),gross_gain=int(entrants.sum()),
        selected=len(selected),old_improve=int(improved.sum()),
        old_worse=int(worse.sum()),old_same=int((was&(original==after)).sum()),
        gains_fullmatched=int((entrants&oldfull).sum()),gains_fullunmatched=int((entrants&~oldfull).sum()),
        gains_recorded=int((entrants&recorded).sum()),gains_neither=int((entrants&~recorded&~oldfull).sum()),
        full_retained=int((now&oldfull).sum()),exact_recorded=int((recorded&(after==m.recorded)).sum()),
        covered_recorded=int(new[m.recorded[recorded]].sum()),covered_full=int(new[reference[oldfull]].sum()),
        added_sum_rank=int(m.r[after[entrants]].sum()),added_median_rank=float(np.median(m.r[after[entrants]])) if entrants.any() else None)


def evaluate_panel(year,name,capname,m,cap,rank,k,kernel,full,results,diagnostics):
    show=rank<k;start=clear(m,kernel,cap,show);slack,edges=residual(m,start,cap,show)
    key=dict(year=year,baseline=name,capacity=capname,k=k)
    opt,cert=match_vacancies(m,slack,edges,SEED)
    graphs=[('unchanged',np.empty(0,np.int32)),('private_favorite',uncoordinated(m,edges)),
            ('serial',serial(m,slack,edges)),('maximum',opt)]
    only_residual=np.zeros(len(m.a),bool)
    only_residual[edges]=True
    residual_da=clear(m,kernel,slack,only_residual)
    graphs.append(('residual_DA',residual_da[residual_da>=0]))
    main=(year,name,capname,k)==(2026,'informed','offers',5)
    for method,selected in graphs:
        changed=replace(m,show,selected,k);after=clear(m,kernel,cap,changed)
        if main:independent(m,changed,cap,after)
        rec=metrics(m,full,start,after,changed,selected)
        if method in ('maximum','serial','residual_DA'):
            witness=start.copy();witness[m.a[selected]]=selected
            m.verify(cap,(~changed).astype(np.uint8),1,witness)
            assert (after[witness>=0]>=0).all() and (after[witness>=0]<=witness[witness>=0]).all()
            assert np.array_equal(after>=0,witness>=0)
            assert rec['gross_loss']==rec['old_worse']==0 and rec['gross_gain']==len(selected)
        results.append(dict(**key,method=method,**rec))
    diagnostics.append(dict(**key,applicants=m.n,base_placements=int((start>=0).sum()),unmatched=int((start<0).sum()),
        unfilled=int(slack.sum()),degree_zero=int((m.degree==0).sum()),full_placements=int((full>=0).sum()),
        recorded_offers=int((m.recorded>=0).sum()),**cert))
    if main:
        optout=[]
        for t in [.25,.5,.75]:
            lottery=np.random.default_rng(SEED+10).random(m.n)<t
            selected=opt[lottery[m.a[opt]]];changed=replace(m,show,selected,k);after=clear(m,kernel,cap,changed)
            rec=metrics(m,full,start,after,changed,selected)
            assert rec['gross_loss']==rec['old_worse']==0 and rec['gross_gain']==len(selected)
            optout.append(dict(takeup=t,**rec))
        variants=[]
        times=np.zeros(m.n,int)
        for j in range(17):
            chosen,_=match_vacancies(m,slack,edges,SEED+j);times[m.a[chosen]]+=1
            variants.append(dict(seed=SEED+j,chosen=len(chosen),gains_neither=int(((full[m.a[chosen]]<0)&(m.recorded[m.a[chosen]]<0)).sum()),
                          median_rank=float(np.median(m.r[chosen])) if len(chosen) else None))
        freq=np.bincount(times[np.unique(m.a[edges])],minlength=18)
        forecast=[]
        for noise in [0,10,30]:
            alt=copy.copy(m);rng=np.random.default_rng(28802416)
            tie=rng.permutation(len(m.a));junk=rng.standard_normal(len(m.a))
            order=np.lexsort((tie,-(m.score+noise*junk),m.school))
            alt.priority=np.empty(len(order),np.int32);alt.priority[order]=np.arange(len(order))
            pre=clear(alt,kernel,cap,show);sl,e=residual(alt,pre,cap,show)
            chosen,_=match_vacancies(alt,sl,e,SEED);changed=replace(m,show,chosen,k);after=clear(m,kernel,cap,changed)
            forecast.append(dict(score_noise=noise,baseline_partner_agree=int((pre==start).sum()),
                primary_unmatched_claimed=int((start[m.a[chosen]]<0).sum()),**metrics(m,full,start,after,changed,chosen)))
        return dict(takeup=optout,lotteries=variants,eligible_selection_frequency=freq.tolist(),forecast=forecast)
    return {}


def main(a):
    sys.path.insert(0,str(a.university.resolve()))
    from run import Market,Kernel,features,history,ordering
    results=[];diag=[];extra={};prev=None
    with tempfile.TemporaryDirectory() as tmp:
        kernel=Kernel(tmp)
        for year in (2024,2025,2026):
            m=Market(a.inputs,year,positive_scores=True)
            rank={'first':m.pos.astype(np.uint8),
                  'uniform0':ordering(m,np.random.default_rng(14012026).random(len(m.a)))}
            if prev is not None:
                _,x,_=features(m,prev);rank['cutoff']=ordering(m,40.*(x>=0)-m.r)
            prev=history(m)
            if year==2026:
                for name in ['informed','blind']:rank[name]=learned(m,a.features,a.fit,name)
            for capname,cap in [('offers',m.cap),('base',m.base)]:
                if capname=='base' and year!=2026:continue
                full=clear(m,kernel,cap,np.ones(len(m.a),bool))
                for name,r in rank.items():
                    if capname=='base' and name!='informed':continue
                    for k in ([1,5] if name=='first' else [5]):
                        panel=evaluate_panel(year,name,capname,m,cap,r,k,kernel,full,results,diag)
                        if panel:extra=panel
                        print('cleared',year,name,capname,k,flush=True)
    a.out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(results).to_csv(a.out/'panel.csv',index=False);pd.DataFrame(diag).to_csv(a.out/'reachable.csv',index=False)
    (a.out/'sensitivity.json').write_text(json.dumps(extra,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('university','inputs','features','fit','out'):p.add_argument(name,type=Path)
    main(p.parse_args())
