"""Distribution, common-composition estimates and order-averaged editing-group contributions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def totals(v):
    return v.drop(columns=['priority','grade','entitled','length'],errors='ignore').sum(numeric_only=True)

def expanded(frame):
    rows=[]
    for (year,mechanism,policy),v in frame.groupby(['year','mechanism','policy']):
        for priority in ('all','designated','other'):
            a=v if priority=='all' else v[v.priority==int(priority=='designated')]
            for length in ('all','short','long'):
                b=a if length=='all' else a[(a.length<=1)==(length=='short')]
                for entitlement in ('all','yes','no'):
                    g=b if entitlement=='all' else b[b.entitled==int(entitlement=='yes')]
                    t=totals(g)
                    rows.append(dict(year=year,mechanism=mechanism,policy=policy,priority=priority,length=length,
                                     entitlement=entitlement,**{c:int(z) for c,z in t.items() if c!='year'}))
    return pd.DataFrame(rows)

def standardize(frame):
    index=['grade','length','entitled'];out=[]
    for (year,mechanism),v in frame[frame.policy=='rank'].groupby(['year','mechanism']):
        strata=v.groupby(index).n.sum()
        counts=v.pivot(index=index,columns='priority',values='n').fillna(0)
        common=counts.index[counts.min(axis=1)>0]
        w=strata.reindex(common);w=w/w.sum()
        for pr,g in v.groupby('priority'):
            sub=g.set_index(index).reindex(common)
            both=(g.gain-g.loss).sum()/g.n.sum()
            probs={'gain':sub.gain/sub.n,'loss':sub.loss/sub.n,'net':(sub.gain-sub.loss)/sub.n,
                   'new_nonrecipient':sub.new_nonrecipient/sub.n,'worse_held':sub.worse_held/sub.n}
            out.append(dict(year=year,mechanism=mechanism,priority=pr,common_n=int(sub.n.sum()),
                            total_n=int(g.n.sum()),common_strata=len(common),crude_net_pp=100*both,
                            **{label+'_pp':100*(p*w).sum() for label,p in probs.items()}))
    return pd.DataFrame(out)

def group_effects(allcells):
    a=allcells[(allcells.length=='all')&(allcells.entitlement=='all')]
    out=[]
    for (y,mech,recipient),q in a.groupby(['year','mechanism','priority']):
        s=q.set_index('policy');both=s.loc['rank','placed']-s.loc['blind','placed']
        for editor,other in [('only_designated','only_other'),('only_other','only_designated')]:
            first=s.loc[editor,'placed']-s.loc['blind','placed'];last=s.loc['rank','placed']-s.loc[other,'placed']
            out.append(dict(year=y,mechanism=mech,recipient_group=recipient,editor_group=editor,
                          first=first,last=last,order_average=(first+last)/2,joint=both))
    o=pd.DataFrame(out)
    for _,g in o.groupby(['year','mechanism','recipient_group']):assert g.order_average.sum()==g.joint.iloc[0]
    return o

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('counts',type=Path);args=p.parse_args()
    frames=pd.concat([pd.read_csv(args.counts/f'{year}-cells.csv.gz') for year in range(2020,2024)],ignore_index=True)
    cells=expanded(frames)
    cells.to_csv(args.counts/'transitions.csv.gz',index=False)
    standardize(frames).to_csv(args.counts/'standardized.csv',index=False)
    group_effects(cells).to_csv(args.counts/'edit-decomposition.csv',index=False)
