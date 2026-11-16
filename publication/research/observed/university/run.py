"""Observed application lists, capacity-aware DA, and held-out menu predictions."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

HERE=Path(__file__).resolve().parent
QUOTAS=[1,3,5,10]
DRAWS=64
i32=np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='C_CONTIGUOUS')
u8=np.ctypeslib.ndpointer(dtype=np.uint8,ndim=1,flags='C_CONTIGUOUS')

class Kernel:
    def __init__(self,tmp):
        ext='dylib' if sys.platform=='darwin' else 'so'
        dest=Path(tmp)/('da.'+ext)
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(HERE/'da.cpp'),'-o',str(dest)],check=True)
        self.lib=ct.CDLL(str(dest));self.lib.solve.argtypes=[ct.c_int,ct.c_int,i32,i32,i32,i32,u8,ct.c_uint,i32]
        self.lib.solve.restype=ct.c_int
    def solve(self,m,cap,rank,k):
        out=np.empty(m.n,dtype=np.int32)
        nprop=self.lib.solve(m.n,m.m,m.offset,m.school,m.priority,cap,rank,k,out)
        assert 0<=nprop<=len(rank)
        return out

def codes(series):
    return pd.to_numeric(series,errors='raise').to_numpy(dtype=np.int32)
class Market:
    def __init__(self,root,year,*,positive_scores=False):
        folder=root/str(year)
        data=pd.read_csv(next(folder.glob('C_POST*.csv')),sep=';',encoding='utf-8-sig',low_memory=False,
                         usecols=['MRUN','ORDEN_PREF','COD_CARRERA_PREF','PTJE_PREF','TIPO_PREF','ESTADO_PREF'])
        offers=pd.read_csv(next(folder.glob('OFERTA*.csv')),sep=';' if year==2024 else ',',encoding='utf-8-sig')
        data=data[data.TIPO_PREF.eq('REGULAR')].copy()
        self.n=data.MRUN.nunique()
        data['student']=pd.factorize(data.MRUN,sort=True)[0]
        self.program=codes(offers.COD_CARRERA);self.m=len(offers)
        if positive_scores:
            numeric=pd.to_numeric(data.PTJE_PREF.astype(str).str.replace(',','.'),errors='coerce')
            eligible=(numeric>0)&data.COD_CARRERA_PREF.isin(self.program)&data.ORDEN_PREF.between(1,20)
        else:
            eligible=data.ESTADO_PREF.isin([24,25,26])
        edges=data[eligible].sort_values(['student','ORDEN_PREF']).copy()
        duplicates=int(edges.duplicated(['student','COD_CARRERA_PREF']).sum())
        if positive_scores: edges=edges.drop_duplicates(['student','COD_CARRERA_PREF'],keep='first')
        self.school=pd.Index(self.program).get_indexer(edges.COD_CARRERA_PREF).astype(np.int32)
        assert (self.school>=0).all() and offers.COD_CARRERA.is_unique
        assert not edges.duplicated(['student','COD_CARRERA_PREF']).any() and not edges.duplicated(['student','ORDEN_PREF']).any()
        self.r=codes(edges.ORDEN_PREF);self.a=codes(edges.student)
        self.degree=np.bincount(self.a,minlength=self.n).astype(np.int32)
        self.offset=np.r_[0,np.cumsum(self.degree)].astype(np.int32)
        self.score=pd.to_numeric(edges.PTJE_PREF.astype(str).str.replace(',','.'),errors='raise').to_numpy(dtype=float)
        assert np.isfinite(self.score).all() and self.r.min()>=1 and self.r.max()<=20
        self.pos=(np.arange(len(edges),dtype=np.int32)-self.offset[self.a])
        cap=offers[['VAC_1ER','VAC_2DO','SOBRECUPOS_1ER','SOBRECUPOS_2DO']].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy()
        assert np.all(cap==cap.astype(np.int32)) and np.min(cap)>=0
        self.cap=cap.sum(1).astype(np.int32);self.base=cap[:,:2].sum(1).astype(np.int32)
        rng=np.random.default_rng(288013+year)
        order=np.lexsort((rng.permutation(len(edges)), -self.score, self.school))
        self.priority=np.empty(len(edges),np.int32);self.priority[order]=np.arange(len(edges))
        picked=edges.ESTADO_PREF.eq(24).to_numpy()
        assert not pd.Series(self.a[picked]).duplicated().any()
        self.recorded=np.full(self.n,-1,np.int32);self.recorded[self.a[picked]]=np.flatnonzero(picked)
        self.flow=dict(regular_rows=len(data),students=self.n,programs=self.m,edges=len(edges),
                       excluded_rows=int((~eligible).sum()),duplicate_rows=duplicates,no_edges=int((self.degree==0).sum()),
                       more_than_five=int((self.degree>5).sum()),regular_seats=int(self.base.sum()),
                       extra_seats=int(self.cap.sum()-self.base.sum()),recorded_offers=int(picked.sum()))

    def verify(self,cap,rank,k,out):
        placed=np.flatnonzero(out>=0);edges=out[placed];dest=self.school[edges]
        occupancy=np.bincount(dest,minlength=self.m)
        assert (occupancy<=cap).all() and np.array_equal(self.a[edges],placed)
        assert np.all(rank[edges]<k)
        worst=np.full(self.m,-1,np.int32);np.maximum.at(worst,dest,self.priority[edges])
        last=np.where(out>=0,out,self.offset[1:]);better=np.arange(len(rank))<last[self.a]
        block=better&(rank<k)&((occupancy<cap)[self.school]|(self.priority<worst[self.school]))
        assert not np.any(block)

    def batch(self,cap):
        # Simultaneous resident proposals and a fresh program-wide ordering each round.
        held=np.empty(0,dtype=np.int32);ptr=self.offset[:-1].copy();free=np.flatnonzero(self.degree)
        while len(free):
            proposals=ptr[free];ptr[free]+=1
            pool=np.r_[held,proposals]
            order=np.lexsort((self.priority[pool],self.school[pool]));sorted_pool=pool[order]
            js=self.school[sorted_pool]
            start=np.maximum.accumulate(np.where(np.r_[True,js[1:]!=js[:-1]],np.arange(len(js)),0))
            keep=np.arange(len(js))-start < cap[js]
            held=sorted_pool[keep];rejected=self.a[sorted_pool[~keep]]
            free=rejected[ptr[rejected]<self.offset[rejected+1]]
        out=np.full(self.n,-1,np.int32);out[self.a[held]]=held
        return out

def history(m):
    idx=m.recorded[m.recorded>=0]; cut=np.full(m.m,np.inf);np.minimum.at(cut,m.school[idx],m.score[idx])
    counts=np.bincount(m.school,minlength=m.m);global_cut=float(np.median(cut[np.isfinite(cut)]))
    return pd.DataFrame({'cut':cut,'pressure':np.log1p(counts/np.maximum(1,m.cap))},index=m.program),global_cut

def features(m,h):
    table,global_cut=h
    htab=table.reindex(m.program)
    missing=~np.isfinite(htab.cut.to_numpy())
    cut=np.where(missing,global_cut,htab.cut.to_numpy());pressure=htab.pressure.fillna(table.pressure.median()).to_numpy()
    x=np.clip((m.score-cut[m.school])/50,-5,5);r=np.log(m.r);d=np.log1p(m.degree[m.a])
    # Saturating and hinge effects let the fit distinguish safe options and near misses.
    features=np.column_stack((r,r*r,d,(m.score-650)/100,x,np.maximum(x,0),np.minimum(x+1,0),
                              r*x,r*np.maximum(x,0),r*d,pressure[m.school],np.log1p(m.cap[m.school]),
                              missing[m.school].astype(float)))
    return features,x,int(missing[m.school].sum())

def fit(X,y):
    mu=X.mean(0);sd=X.std(0).clip(.01);Z=(X-mu)/sd
    def fn(w):
        v=Z@w[:-1]+w[-1];res=expit(v)-y
        val=np.mean(np.logaddexp(0,v)-y*v)+.0005*np.sum(w[:-1]**2)
        grad=np.r_[Z.T@res/len(y)+.001*w[:-1],res.mean()]
        return val,grad
    result=minimize(fn,np.zeros(X.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':240,'ftol':1e-12})
    assert result.success,result.message
    return mu,sd,result.x

def ordering(m,value):
    # Larger policy scores win. Ties fall to earlier submitted choices.
    ix=np.lexsort((m.r,-value,m.a));order=np.empty(len(ix),np.uint8)
    order[ix]=(np.arange(len(ix))-m.offset[m.a[ix]]).astype(np.uint8)
    return order

def measures(m,rank,k,y,full):
    offered=full>=0;record=m.recorded[m.recorded>=0]
    return dict(placements=int((y>=0).sum()),recovery=int((rank[full[offered]]<k).sum()),
                observed_recovery=int((rank[record]<k).sum()),exact_observed=int(np.sum((m.recorded>=0)&(m.recorded==y))),
                matched_loss=int(np.sum(offered&(y<0))),matched_gain=int(np.sum(~offered&(y>=0))),
                matched_worse=int(np.sum(offered&((y<0)|(y>full)))),
                unedited_lost=int(np.sum(offered&(y<0)&(m.degree<=k))))

def execute(root,out,tmp):
    kernel=Kernel(tmp);years={};meta={}
    for year in [2024,2025,2026]:
        m=Market(root,year);zeros=np.zeros(len(m.a),np.uint8)
        y=kernel.solve(m,m.cap,zeros,1);m.verify(m.cap,zeros,1,y)
        assert np.array_equal(y,m.batch(m.cap))
        known=m.recorded>=0; hit=known&(y==m.recorded)
        meta[year]=dict(**m.flow,full_placements=int((y>=0).sum()),baseline_observed_exact=int(hit.sum()),
                        match_status_agreement=int(((y>=0)==known).sum()),full_assignment_agreement=int((y==m.recorded).sum()))
        years[year]=(m,y)
    train,ytrain=years[2025];X,_,_=features(train,history(years[2024][0]))
    target=np.zeros(len(X));target[ytrain[ytrain>=0]]=1
    mu,sd,w=fit(X,target);del X
    m,reference=years[2026];X,x,mis=features(m,history(train))
    learned=(X-mu)@(w[:-1]/sd)+w[-1]
    # Cutoff control: rank applicants' predicted admissible options first, then in their own order.
    cutoff=(x>=0).astype(float)*40-m.r
    ranks={'trained':ordering(m,learned),'own_rank':m.pos.astype(np.uint8),'historical_cutoff':ordering(m,cutoff)}
    out.mkdir(parents=True,exist_ok=True)
    records=[];analytic=[]
    for capname,cap in [('published_plus_overbook',m.cap),('published_only',m.base)]:
        full=kernel.solve(m,cap,np.zeros(len(m.a),np.uint8),1)
        assert np.array_equal(full,m.batch(cap))
        if capname.endswith('only'):meta['base_capacity_2026']=dict(full_placements=int((full>=0).sum()),baseline_observed_exact=int(((full>=0)&(full==m.recorded)).sum()))
        def run(p,k,rank,draw):
            assigned=kernel.solve(m,cap,rank,k);m.verify(cap,rank,k,assigned)
            records.append(dict(capacity=capname,policy=p,k=k,draw=draw,**measures(m,rank,k,assigned,full)))
        quotas=QUOTAS if capname.endswith('overbook') else [5]
        for k in quotas:
            for p,o in ranks.items():run(p,k,o,-1)
            analytic.append(dict(capacity=capname,k=k,full_count=int((full>=0).sum()),
                       expected_uniform_recovery=float(np.minimum(1,k/m.degree[full>=0]).sum()),
                       expected_uniform_observed=float(np.minimum(1,k/m.degree[m.recorded>=0]).sum())))
        for draw in range(DRAWS):
            rand=np.random.default_rng(14012026+draw)
            o=ordering(m,rand.random(len(m.a)))
            for k in quotas: run('uniform',k,o,draw)
        print(capname,'finished',flush=True)
    pd.DataFrame(records).to_csv(out/'counts.csv',index=False)
    pd.DataFrame(analytic).to_csv(out/'analytic.csv',index=False)
    meta['fitted']={'feature_mean':mu.tolist(),'feature_sd':sd.tolist(),'weights_intercept_last':w.tolist(),'fallback_edges_2026':mis}
    (out/'metrics.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:execute(a.inputs,a.out,tmp)
