"""Finite-sample simultaneous market rollout risk bounds in the conditional simulator."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import xlogy, xlog1py

def upper(x,n,delta):
    if x==1: return 1.
    if x==0: return -np.expm1(np.log(delta)/n)
    target=-np.log(delta)/n
    def f(p):
        return xlogy(x,x/p) + xlogy(1-x,1-x) - xlog1py(1-x,-p) - target
    end=np.nextafter(1.,0.)
    return brentq(f,x,end) if f(end)>=0 else 1.

def compute(rows,weights):
    policy_names=['transparent','pointwise','context']
    policy_names=[p+d for p in policy_names for d in ('_top','_anchor')]
    designs=['uniform','enriched']
    weights={k:v/sum(weights.values()) for k,v in weights.items()}
    output=[]
    for design in designs:
        for policy in policy_names:
            stats={}
            for part,n,start in [('cal',32,2026120900),('summer',48,2026120500),('winter',48,2026120700)]:
                g=rows[(rows['split']==part)&(rows['design']==design)&(rows.policy==policy)].copy()
                assert len(g)==12*n and set(g.seed)==set(range(start,start+n)),(design,policy,part,len(g))
                assert not g.duplicated(['seed','month','borough']).any()
                assert (g.groupby('seed').size()==12).all() and len(g.month.unique())==3 and set(g.borough)==set(weights)
                g['w']=g.borough.map(weights)/3
                g['bad']=g.w*(g.loss>0)
                stats[part]=dict(markets=len(g),risk=float(g.groupby('seed').bad.sum().mean()))
                by_cell=g.groupby(['month','borough']).agg({'gain':'sum','loss':'sum','lower_n':'sum','w':'first'})
                stats[part]['net_pp']=float((100*by_cell.w*(by_cell.gain-by_cell.loss)/by_cell.lower_n).sum())
            u=upper(stats['cal']['risk'],32,.05/12)
            for target in [.01,.05,.1]:
                q=min(1.,target/u)
                output.append(dict(design=design,policy=policy,target=target,cal_risk=stats['cal']['risk'],
                                    upper=u,uptake=q,summer_unrestricted=stats['summer']['risk'],
                                    summer_risk=q*stats['summer']['risk'],summer_gain_pp=q*stats['summer']['net_pp'],
                                    winter_unrestricted=stats['winter']['risk'],winter_risk=q*stats['winter']['risk'],
                                    winter_gain_pp=q*stats['winter']['net_pp']))
    return pd.DataFrame(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('normalized_counts',type=Path)
    p.add_argument('metadata',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    rows=pd.read_csv(a.normalized_counts)
    compute(rows,json.loads(a.metadata.read_text())['weights']).to_csv(a.output,index=False,float_format='%.12g')
