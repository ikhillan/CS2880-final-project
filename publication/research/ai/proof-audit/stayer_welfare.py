"""Rank and rent changes among original lower-income recipients, using the frozen scorer."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np
from evaluate_guardrail import da, rank_slates
from guardrail import repair_with_freezes


def standardize(rows, weights):
    seeds=sorted({r['seed'] for r in rows});cells=sorted({(r['month'],r['borough']) for r in rows})
    w=np.array([weights[b] for _,b in cells]);w/=w.sum()
    fields=['original','better','equal','worse','changed','transfer','lost','stayed','burden_change',
            'burden_increased','burden_decreased','common_stayed','common_burden_change','common_burden_increased','common_burden_decreased']
    table={(r['seed'],r['month'],r['borough']):r for r in rows}
    x=np.array([[[table[s,m,b][k] for k in fields] for m,b in cells] for s in seeds])
    rng=np.random.default_rng(441903);n=len(seeds)
    freq=np.vstack([np.ones(n),rng.multinomial(n,np.full(n,1/n),size=5000)])
    total=np.einsum('bs,scf->bcf',freq,x,optimize=True)
    summary={'raw':dict(zip(fields,x.sum((0,1)).tolist())),'standardized':{}}
    for numerator in ('better','equal','worse','changed','transfer','lost','burden_change','burden_increased','burden_decreased',
                      'common_burden_change','common_burden_increased','common_burden_decreased'):
        den='original'
        if numerator.startswith('burden_'):den='stayed'
        if numerator.startswith('common_burden_'):den='common_stayed'
        bot=total[:,:,fields.index(den)];ok=(bot>0).all(1)
        if not ok[0]:
            summary['standardized'][numerator]=dict(value=None,empty_cells=[cells[j] for j in range(len(cells)) if bot[0,j]==0])
            continue
        est=100*(total[ok,:,fields.index(numerator)]/bot[ok])@w
        summary['standardized'][numerator]=dict(value=float(est[0]),paired_95ci=np.quantile(est[1:],[.025,.975]).tolist(),dropped_bootstraps=int((~ok).sum()))
    return summary


def run(base,split,out):
    meta=json.loads((base/f'{split}.json').read_text())
    with np.load(base/f'{split}.npz') as data:
        e,pref,seller,U,low,x=(data[k] for k in ('e','pref','seller','uniform','lower','x'))
        assert meta['feature_order'][0]=='rent_over_cap'
        burden=x[:,:,:,0].astype(float)*.4
        U=U<5
    with np.load(base/f'{split}-scores.npz') as data:scores=data['context']
    n,m=e.shape[1:];original=da(pref,seller,U)
    score,direct=rank_slates(e,meta['keys'],scores,None)
    direct=np.where(low[:,:,None],U,direct);guard=np.empty_like(e)
    for t in range(len(e)):
        guard[t],_,_=repair_with_freezes(e[t],score[t],U[t],low[t],np.where(original[t]<m,original[t],-1))
    mu_direct=da(pref,seller,direct);mu_guard=da(pref,seller,guard)
    ids=np.indices(original.shape)
    def costs(arr,mu):
        return arr[ids[0],ids[1],np.minimum(mu,m-1)]
    oldrank=costs(pref,original);oldprice=costs(burden,original)
    who=low & (original<m);common=who & (mu_direct<m) & (mu_guard<m)
    rows=[]
    for mode,mu in [('direct',mu_direct),('blind',mu_guard)]:
        stay=who & (mu<m);ranks=np.where(mu<m,costs(pref,mu),128);prices=costs(burden,mu)
        better=who & (ranks<oldrank);same=who & (ranks==oldrank);worse=who & (ranks>oldrank)
        assert np.array_equal(same,who & (original==mu))
        assert np.array_equal(same|worse|better,who)
        loss=who & (mu==m)
        assert mode=='direct' or not loss.any()
        for t,(seed,design,month,borough) in enumerate(meta['keys']):
            row=dict(split=split,mode=mode,seed=seed,design=design,month=month,borough=borough,
                     original=int(who[t].sum()),better=int(better[t].sum()),equal=int(same[t].sum()),worse=int(worse[t].sum()),
                     changed=int((who[t]&(original[t]!=mu[t])).sum()),transfer=int((stay[t]&(original[t]!=mu[t])).sum()),
                     lost=int(loss[t].sum()))
            for prefix,sel in [('',stay[t]),('common_',common[t])]:
                price_delta=prices[t][sel]-oldprice[t][sel]
                row[prefix+'stayed']=len(price_delta)
                row[prefix+'burden_change']=float(price_delta.sum())
                row[prefix+'burden_increased']=int((price_delta>0).sum())
                row[prefix+'burden_decreased']=int((price_delta<0).sum())
            rows.append(row)
    with (out/f'{split}-recipient-welfare.csv.gz').open('wb') as dest, gzip.GzipFile(fileobj=dest,mode='wb',mtime=0) as zipped:
        with io.TextIOWrapper(zipped,newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary={}
    for design,mode in sorted({(r['design'],r['mode']) for r in rows}):
        subset=[r for r in rows if r['design']==design and r['mode']==mode]
        summary[f'{design}:{mode}']=standardize(subset,meta['weights'])
    (out/f'{split}-recipient-welfare.json').write_text(json.dumps(summary,indent=2)+'\n')
    for key,v in summary.items():
        print(split,key,'raw worse/equal/better',*[int(v['raw'][f]) for f in ('worse','equal','better')],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--split',nargs='+',default=['summer','winter']);a=p.parse_args()
    for split in a.split:run(a.base,split,a.out)
