"""Paired cluster-bootstrap intervals for cell-standardized match and placement comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATS=['matched_total','matched_lower','matched_higher','lower_n','higher_n','joint_loss','joint_gain','full_matched','full_partner_in_menu','target_nll_sum','lower_zero']


def describe(data,meta):
    cells=sorted({(k[2],k[3]) for k in meta['keys']});seeds=sorted(data.seed.unique());
    weights=np.array([meta['weights'][b] for m,b in cells]);weights/=weights.sum();n=len(seeds)
    table_index=pd.MultiIndex.from_tuples([(s,m,b) for s in seeds for m,b in cells],names=['seed','month','borough'])
    rng=np.random.default_rng(441903);boot=rng.multinomial(n,np.full(n,1/n),size=5000);freq=np.vstack([np.ones(n),boot]);result=[]
    for design,frame in data.groupby('design'):
        values={}
        for (policy,roll),df in frame.groupby(['policy','rollout']):
            table=df.set_index(['seed','month','borough']).reindex(table_index);assert table.matched_total.notna().all()
            arr=table[STATS].fillna(0).to_numpy().reshape(n,len(cells),len(STATS));totals=np.einsum('bs,scf->bcf',freq,arr,optimize=True)
            def col(key): return totals[:,:,STATS.index(key)]
            def weighted(numerator,denominator):return (numerator/denominator)@weights
            all_n=col('lower_n')+col('higher_n'); metrics={'total_rate':100*weighted(col('matched_total'),all_n)}
            for s in ('lower','higher'): metrics[s+'_rate']=100*weighted(col('matched_'+s),col(s+'_n'))
            for s in ('joint_loss','joint_gain','lower_zero'): metrics[s]=100*weighted(col(s),col('lower_n'))
            metrics['oracle_hit']=100*weighted(col('full_partner_in_menu'),col('full_matched'))
            if df.target_nll_sum.notna().any(): metrics['nll']=weighted(col('target_nll_sum'),all_n)
            values[policy,roll]=metrics
        for (policy,roll),metrics in values.items():
            for name,vals in metrics.items():
                if name=='lower_zero' and (policy!='uniform' or roll!='all'):continue
                if name=='nll' and (policy.endswith('_anchor') or roll!='all'):continue
                comps={'level':vals}
                if name in ('total_rate','lower_rate','higher_rate','oracle_hit'):
                    for ref in ('uniform','transparent','transparent_anchor','pointwise','pointwise_anchor','su'):
                        if policy==ref:continue
                        comps['minus_'+ref]=vals-values[ref,roll][name]
                if name=='nll' and policy=='context':comps['minus_pointwise']=vals-values['pointwise',roll][name]
                for what,v in comps.items():
                    lo,hi=np.quantile(v[1:],[.025,.975]);result.append(dict(design=design,policy=policy,rollout=roll,metric=name,contrast=what,estimate=v[0],lo=lo,hi=hi))
    return pd.DataFrame(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--counts',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    frames=[]
    for split in ['summer','winter']:
        df=pd.read_csv(a.counts/f'{split}-counts.csv.gz');summary=describe(df,json.loads((a.data/f'{split}.json').read_text()));frames.append(summary.assign(split=split))
    pd.concat(frames,ignore_index=True).to_csv(a.counts/'contrasts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
