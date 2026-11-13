"""Chronologically held-out bilateral preference benchmark. All published rows are aggregates."""
import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import logsumexp

TRAIN = (1, 2, 3, 4, 5, 7, 8, 9)
TEST = (10, 11, 13, 14, 15, 17, 19, 21)
DAY = {10:'2003-09-24',11:'2003-09-24',13:'2003-10-08',14:'2003-10-08',
       15:'2004-02-24',17:'2004-02-25',19:'2004-04-06',21:'2004-04-07'}
INTERESTS = 'sports tvsports exercise dining museums art hiking gaming clubbing reading tv theater movies concerts music shopping yoga'.split()
TRAITS = 'attr sinc intel fun amb'.split()
DIRECT = ['age', 'date', 'go_out']+[x+'3_1' for x in TRAITS]
WEIGHTS = [x+'1_1' for x in TRAITS]+['shar1_1']
PROFILE = DIRECT+INTERESTS+WEIGHTS+['field_cd','goal']
HASH = {'data.tab':'8616bd64ade62c07d8668229c926761ed7a2f59a3f97b8064bd69efc3e07f81c',
        'profiles.csv':'2e518c3315c4ae360d4c4b143b64b49dc3efc21b93d51007703ec59f97524309'}

@dataclass
class Market:
    wave: int
    side: int
    observed: np.ndarray
    mutual: np.ndarray
    acceptable: np.ndarray
    row_liking: np.ndarray
    column_liking: np.ndarray
    x: np.ndarray
    similarity: np.ndarray

def read_data(directory):
    for name,digest in HASH.items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
    original=pd.read_csv(directory/'data.tab',sep='\t')
    keys=['wave','iid','pid']
    ext=pd.read_csv(directory/'profiles.csv',encoding='latin1',usecols=keys+['dec','gender','like']+PROFILE)
    merged=original.merge(ext.dropna(subset=['pid']),on=keys,how='left',validate='one_to_one',
                          suffixes=('_dep',''),indicator=True)
    assert (merged['_merge']=='both').all() and (merged.dec==merged.dec_dep).all()
    assert (merged.gender_dep==merged.gender).all() and set(merged.wave)==set(TRAIN+TEST)
    assert not (set(merged.loc[merged.wave.isin(TRAIN),'iid'])&set(merged.loc[merged.wave.isin(TEST),'iid']))
    # The pre-event dating frequency comes from the same questionnaire in both data files.
    merged['date']=merged['date'].where(merged['date'].notna(),merged['date_dep'])
    assert not merged.groupby('iid')[PROFILE].nunique(dropna=True).gt(1).any().any()
    return merged

def person_features(a,b):
    na,nb=len(a),len(b)
    x=[]
    for name in DIRECT:
        aa=a[name].to_numpy(float)[:,None]; bb=b[name].to_numpy(float)[None,:]
        x.extend([np.broadcast_to(aa,(na,nb)),np.broadcast_to(bb,(na,nb))])
    age=np.abs(a.age.to_numpy(float)[:,None]-b.age.to_numpy(float)[None,:]); x.append(age)
    for key in ('goal','field_cd'):
        aa=a[key].to_numpy(float)[:,None];bb=b[key].to_numpy(float)[None,:]
        x.append(np.where(np.isfinite(aa)&np.isfinite(bb),aa==bb,np.nan))
    aa=a[INTERESTS].to_numpy(float)[:,None,:];bb=b[INTERESTS].to_numpy(float)[None,:,:]
    valid=np.isfinite(aa)&np.isfinite(bb);count=valid.sum(2).clip(min=1)
    am=np.where(valid,aa,0).sum(2)/count;bm=np.where(valid,bb,0).sum(2)/count
    at=np.where(valid,aa-am[:,:,None],0);bt=np.where(valid,bb-bm[:,:,None],0)
    den=np.sqrt((at*at).sum(2)*(bt*bt).sum(2))
    corr=np.divide((at*bt).sum(2),den,out=np.full_like(den,np.nan),where=(den>0)&(count>=2));x.append(corr)
    sim=np.nan_to_num(corr,nan=0.)-np.nan_to_num(age,nan=8.)/10
    aw=a[WEIGHTS].to_numpy(float); bw=b[WEIGHTS].to_numpy(float)
    aw=aw/np.maximum(aw.sum(1)[:,None],1);bw=bw/np.maximum(bw.sum(1)[:,None],1)
    for i,trait in enumerate(TRAITS):
        x.extend([aw[:,i,None]*b[trait+'3_1'].to_numpy()[None,:],a[trait+'3_1'].to_numpy()[:,None]*bw[None,:,i]])
    return np.stack(x,axis=-1),sim

def build_markets(df,side):
    result=[]
    for wave in TRAIN+TEST:
        d=df[df.wave==wave]; person=d.groupby(['gender','iid'])[PROFILE].first()
        a=person.loc[side]; b=person.loc[1-side]
        def values(whose,column):
            sub=d[d.gender==whose]
            frame=sub.pivot(index='iid',columns='pid',values=column)
            return (frame if whose==side else frame.T).reindex(index=a.index,columns=b.index).to_numpy()
        left=values(side,'dec');right=values(1-side,'dec'); observed=np.isfinite(left)&np.isfinite(right)
        mutual=(left==1)&(right==1)
        la,lb=values(side,'like'),values(1-side,'like')
        acceptable=mutual&(la>=1)&(la<=10)&(lb>=1)&(lb<=10)
        feat,sim=person_features(a,b)
        result.append(Market(wave,side,observed,mutual,acceptable,la,lb,feat,sim))
    return result

def refine(market,draw):
    rng=np.random.default_rng([27128,market.wave,market.side,draw])
    a=np.where(market.acceptable,market.row_liking,-999.)
    b=np.where(market.acceptable,market.column_liking,-999.)
    row=np.lexsort((rng.random(a.shape),-a),axis=1)
    priority=np.argsort(np.lexsort((rng.random(b.shape),-b),axis=0),axis=0)
    return row,priority

def allocate(order,priority,graph):
    na,nb=graph.shape
    partner=np.full(na,-1,int); holder=np.full(nb,-1,int)
    ptr=np.zeros(na,int)
    for a0 in range(na):
        a=a0
        while a!=-1:
            while ptr[a]<nb:
                b=order[a,ptr[a]];ptr[a]+=1
                if not graph[a,b]:continue
                old=holder[b]
                if old>=0 and priority[old,b]<priority[a,b]:continue
                partner[a]=b;holder[b]=a;a=old
                if old>=0:partner[old]=-1
                break
            else:a=-1
    # Verify stability and individual rationality directly using complete strict rankings.
    rr=np.argsort(order,axis=1)
    current=np.where(partner>=0,rr[np.arange(na),partner],nb)
    against=np.where(holder>=0,priority[holder,np.arange(nb)],na)
    assert not np.any(graph&(rr<current[:,None])&(priority<against[None,:]))
    assert all(graph[a,b] for a,b in enumerate(partner) if b>=0)
    return partner

def matrices(markets,known):
    pools={m.wave:(m.acceptable if known else m.observed) for m in markets}
    features={}
    for m in markets:
        pool=pools[m.wave]; shape=pool.shape
        features[m.wave]=np.concatenate([m.x,np.log1p(np.broadcast_to(pool.sum(1)[:,None],shape))[:,:,None],
                                     np.log1p(np.broadcast_to(pool.sum(0)[None,:],shape))[:,:,None]],axis=2)
    train=np.concatenate([features[w][pools[w]] for w in TRAIN])
    mean=np.nanmean(train,axis=0);dev=np.nanstd(train,axis=0)
    for m in markets:
        a=features[m.wave];scaled=np.nan_to_num((a-mean)/np.maximum(dev,1e-8))
        features[m.wave]=np.concatenate([np.ones((*a.shape[:2],1)),scaled],axis=2)
    return pools,features

def fit(markets,known):
    pools,x=matrices(markets,known)
    obs=[]
    for m in markets:
        if m.wave not in TRAIN:continue
        y=np.zeros(m.observed.shape)
        for seed in range(32):
            partner=allocate(*refine(m,seed+10000),m.acceptable)
            good=np.flatnonzero(partner>=0);y[good,partner[good]]+=1/32
        obs.append((x[m.wave],pools[m.wave],y))
    n=sum(len(y) for _,_,y in obs)
    def loss(w):
        v=0.;g=np.zeros_like(w)
        for X,pool,y in obs:
            s=np.where(pool,X@w,-np.inf)
            z=np.logaddexp(0,logsumexp(s,axis=1))
            v+=z.sum()-(y*(X@w)).sum()
            err=np.where(pool,np.exp(s-z[:,None]),0)-y
            g+=np.einsum('ij,ijk->k',err,X)
        # Intercept alone sets the outside mass; only nonconstant coefficients penalized.
        g=g/n;g[1:]+=.1*w[1:]
        return v/n+.05*(w[1:]@w[1:]),g
    ans=minimize(loss,np.zeros(obs[0][0].shape[2]),jac=True,method='L-BFGS-B',options={'gtol':1e-9,'ftol':1e-12})
    assert ans.success
    return pools,{k:X@ans.x for k,X in x.items()}

def slates(score,pool,k,anchor=False):
    ranks=np.argsort(-np.where(pool,score,-np.inf),axis=1,kind='stable')
    result=np.zeros_like(pool)
    if anchor:
        na,nb=pool.shape
        scaled=(score-np.min(score))/max(np.ptp(score),1e-9)
        value=np.where(pool,2*min(na,nb)+scaled,-1e7)
        row,col=linear_sum_assignment(np.column_stack([value,np.zeros((na,na))]),maximize=True)
        real=col<nb;result[row[real],col[real]]=True
    for i,order in enumerate(ranks):
        selected=order[pool[i,order]&~result[i,order]][:max(0,k-int(result[i].sum()))]
        result[i,selected]=True
    assert np.array_equal(result.sum(1),np.minimum(k,pool.sum(1)))
    return result

def run(markets):
    records=[]
    for known in (False,True):
        pools,scores=fit(markets,known)
        for m in markets:
            if m.wave not in TEST:continue
            pool=pools[m.wave]; score=scores[m.wave]
            for seed in range(256):
                order,priority=refine(m,seed)
                full=allocate(order,priority,m.acceptable)
                good=np.flatnonzero(full>=0); labels=full[good]
                random=np.random.default_rng([8671,m.wave,m.side,seed]).random(pool.shape)
                for k in (1,3,5):
                    for name,sc,anchor in (('uniform',random,False),('similarity',m.similarity,False),
                                           ('learned',score,False),('anchor',score,True)):
                        shown=slates(sc,pool,k,anchor)
                        matched=allocate(order,priority,shown&m.acceptable)
                        records.append((m.side,int(known),m.wave,DAY[m.wave],k,name,seed,len(full),len(good),
                                        int(shown[good,labels].sum()),int((matched>=0).sum()),int((shown&m.mutual).sum()),int(m.mutual.sum()),
                                        int(shown.any(0).sum()),pool.shape[1]))
    names=['side','known','wave','day','k','policy','seed','n','full','hit','placed','observed_hits','mutual','visible','n_other']
    return pd.DataFrame(records,columns=names)

def summarize(data,out):
    data=data.copy()
    for name,n,d in [('recovery','hit','full'),('placement','placed','n'),('mutual_exposed','observed_hits','mutual'),
                    ('participant_coverage','visible','n_other')]:
        data[name]=100*data[n]/data[d]
    keys=['side','known','k','policy','wave','day']
    metrics=['recovery','placement','mutual_exposed','participant_coverage']
    frame=data.groupby(keys)[metrics+['n','full','mutual']].mean().reset_index()
    frame.to_csv(out/'wave-results.csv',index=False)
    days=sorted(set(DAY.values()));dayvec=np.array([days.index(DAY[w]) for w in TEST])
    resamp=np.random.default_rng(2811).multinomial(len(days),np.full(len(days),1/len(days)),20000)
    boot=resamp[:,dayvec];boot=boot/boot.sum(1,keepdims=True)
    results=[]
    for (side,known,k),g in frame.groupby(['side','known','k']):
        base=g[g.policy=='uniform'].set_index('wave').loc[list(TEST),metrics].to_numpy()
        for name,group in g.groupby('policy'):
            vals=group.set_index('wave').loc[list(TEST),metrics].to_numpy()
            diff=vals-base;bootdiff=boot@diff
            for j,key in enumerate(metrics):
                lo,hi=np.quantile(bootdiff[:,j],[.025,.975])
                results.append([side,known,k,name,key,vals[:,j].mean(),diff[:,j].mean(),lo,hi])
    return pd.DataFrame(results,columns=['side','known','k','policy','metric','rate','change','low','high'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--side',type=int,choices=[0,1]);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    df=read_data(a.data);summary={'directed_decisions':len(df),'identified_respondents':int(df.iid.nunique()),'source_waves':sorted(map(int,df.wave.unique()))}
    records=[]
    for side in ([0,1] if a.side is None else [a.side]):
        markets=build_markets(df,side)
        if side==0:
            summary['valid_observed_pairs']=sum(int(m.observed.sum()) for m in markets)
            summary['mutual_yes_pairs']=sum(int(m.mutual.sum()) for m in markets)
            summary['mutual_rated_pairs']=sum(int(m.acceptable.sum()) for m in markets)
            summary['train_subjects']=sum(sum(m.observed.shape) for m in markets if m.wave in TRAIN)
            summary['test_subjects']=sum(sum(m.observed.shape) for m in markets if m.wave in TEST)
            summary['test_pairs']=sum(int(m.observed.sum()) for m in markets if m.wave in TEST)
            summary['test_mutual']=sum(int(m.mutual.sum()) for m in markets if m.wave in TEST)
            summary['test_mutual_rated']=sum(int(m.acceptable.sum()) for m in markets if m.wave in TEST)
        records.append(run(markets))
    result=summarize(pd.concat(records),a.out)
    result.to_csv(a.out/'summary.csv',index=False)
    (a.out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary));print(result.query("side==0 and k==5 and policy=='learned'").round(3).to_string(index=False))

if __name__=='__main__':main()
