"""Standardized levels, paired uncertainty and a joint 36-effect band from archived counts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ranks-meta',type=Path,required=True)
    p.add_argument('--counts',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    meta=json.loads(a.ranks_meta.read_text()); counts=pd.read_csv(a.counts)
    seeds=meta['blocks']; cells=[(m,b) for m in meta['months'] for b in meta['boroughs']]
    crossed='policy_w_p' in counts
    settings=['design','w_p','policy_w_p' if crossed else 'sigma_r','budget']
    stats=['lower_n','higher_n','fully_visible_lower_n','UU_matched_lower','UT_matched_lower','TT_matched_lower',
           'UU_matched_higher','UT_matched_higher','TT_matched_higher','fully_visible_lower_lost',
           'fully_visible_lower_gained','UU_blocked_lower','UT_blocked_lower','TT_blocked_lower']
    weights=np.array([meta['raw_borough_weights'][b]/len(meta['months']) for m,b in cells]);weights/=weights.sum()
    rng=np.random.default_rng(130155)
    boot=rng.multinomial(seeds,np.ones(seeds)/seeds,size=10000)
    freq=np.concatenate([np.ones((1,seeds)),boot])
    ix=pd.MultiIndex.from_tuples([(s,m,b) for s in range(seeds) for m,b in cells],names=['block','month','borough'])
    est=[]; samples=[]; names=[]
    for keys,d in counts.groupby(settings,sort=True):
        table=d.set_index(ix.names).reindex(ix)[stats]
        assert len(table)==len(ix) and not table.isna().any().any()
        data=table.to_numpy().reshape(seeds,len(cells),len(stats))
        pts=np.einsum('bs,sck->bck',freq,data,optimize=True)
        scalar=lambda field: pts[:,:,stats.index(field)]
        def ratio(num,den):
            assert (scalar(den)>0).all()
            return 100*np.einsum('bc,c->b',scalar(num)/scalar(den),weights)
        def reported(v):
            return dict(estimate=float(v[0]),pointwise95=np.quantile(v[1:],[.025,.975]).tolist())
        row=dict(zip(settings,keys))
        for arm in ('UU','UT','TT'):
            for group in ('lower','higher'):
                row[arm+'_matched_'+group]=reported(ratio(arm+'_matched_'+group,group+'_n'))
            row[arm+'_blocked_lower']=reported(ratio(arm+'_blocked_lower','lower_n'))
        for group in ('lower','higher'):
            contrast=ratio('UT_matched_'+group,group+'_n')-ratio('UU_matched_'+group,group+'_n')
            row['UT_minus_UU_'+group]=reported(contrast)
            if group=='lower': samples.append(contrast)
        row['fully_visible_lower_gained_contribution']=reported(ratio('fully_visible_lower_gained','lower_n'))
        row['fully_visible_lower_lost_contribution']=reported(ratio('fully_visible_lower_lost','lower_n'))
        row['fully_visible_lower_net_contribution']=reported(ratio('fully_visible_lower_gained','lower_n')-
                                                           ratio('fully_visible_lower_lost','lower_n'))
        sums=d[stats].sum().to_dict()
        row['counts']={key:int(sums[key]) for key in ('lower_n','higher_n','fully_visible_lower_n','fully_visible_lower_lost','fully_visible_lower_gained')}
        est.append(row); names.append(keys)
    assert len(est)==36
    v=np.array(samples).T; sd=v[1:].std(0,ddof=1)
    assert (sd>0).all()
    critical=float(np.quantile(np.abs((v[1:]-v[0])/sd).max(axis=1),.95))
    for j,row in enumerate(est):
        row['UT_minus_UU_lower']['simultaneous95']=(v[0,j]+np.array([-1,1])*critical*sd[j]).tolist()
    sensitivity={}
    for design in meta['designs']:
        for quota in ('k5','q40'):
            sub=[r for r in est if r['design']==design and r['budget']==quota]
            effects=[r['UT_minus_UU_lower'] for r in sub]
            sensitivity[design+'_'+quota]=dict(minimum=min(e['estimate'] for e in effects),
                maximum=max(e['estimate'] for e in effects),positive=sum(e['estimate']>0 for e in effects),
                negative=sum(e['estimate']<0 for e in effects),joint_above_zero=sum(e['simultaneous95'][0]>0 for e in effects),
                joint_below_zero=sum(e['simultaneous95'][1]<0 for e in effects))
    result=dict(blocks=seeds,market_draws=seeds*len(cells)*2,conditional_markets=seeds*len(cells)*2*9,
                da_runs=seeds*len(cells)*2*9*2*3,critical_value=critical,
                bootstrap=dict(reps=len(boot),seed=130155,unit='paired entire seed blocks retaining all designs, cells, parameters and interventions'),
                panel_ranges=sensitivity,estimates=est)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(sensitivity,indent=2))
    print('Baseline:')
    for row in est:
        if row['w_p']==4.0 and (row.get('sigma_r')==.5 or row.get('policy_w_p')==4.0):
            print(row['design'],row['budget'],row['UT_minus_UU_lower'],row['UU_matched_lower']['estimate'],row['UT_matched_lower']['estimate'])


if __name__=='__main__':
    main()
