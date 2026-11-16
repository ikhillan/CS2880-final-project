"""Replay optional renter veto at old offered asking rent before applicant-proposing DA."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from evaluate_guardrail import da, rank_slates
from guardrail import repair_with_freezes


def summarize(rows,weights):
    metrics={'low_rate':('matched_lower','n_lower'), 'high_rate':('matched_higher','n_higher'), 'total_rate':('matched','n')}
    result={}
    for design in sorted({r['design'] for r in rows}):
        part=[r for r in rows if r['design']==design]
        seeds=sorted({r['seed'] for r in part}); cells=sorted({(r['month'],r['borough']) for r in part})
        w=np.array([weights[b] for _,b in cells]);w/=w.sum()
        rng=np.random.default_rng(441903);n=len(seeds)
        freq=np.vstack((np.ones(n),rng.multinomial(n,np.full(n,1/n),size=5000)))
        out={};paired={}
        for mode in ('guard','veto'):
            group=[r for r in part if r['mode']==mode]
            ix={(r['seed'],r['month'],r['borough']):r for r in group}
            fields=[k for k in group[0] if k not in ('seed','design','month','borough','mode')]
            data=np.array([[[ix[s,m,b][f] for f in fields] for m,b in cells] for s in seeds])
            samples=np.einsum('bs,scf->bcf',freq,data,optimize=True)
            vals={}
            for name,(num,den) in metrics.items():
                top=samples[:,:,fields.index(num)];bottom=samples[:,:,fields.index(den)]
                assert (bottom>0).all()
                vals[name]=100*(top/bottom)@w
            out[mode]=dict(raw={f:int(v) for f,v in zip(fields,data.sum((0,1)))},
                           std_percent={k:float(v[0]) for k,v in vals.items()})
            paired[mode]=vals
        shifts={}
        for name in metrics:
            diff=paired['veto'][name]-paired['guard'][name]
            shifts[name]=dict(pp=float(diff[0]),paired_95ci=np.quantile(diff[1:],[.025,.975]).tolist())
        out['veto_minus_guard']=shifts;result[design]=out
    return result


def run(base,out,split):
    metadata=json.loads((base/f'{split}.json').read_text());keys=metadata['keys']
    with np.load(base/f'{split}.npz') as data:
        e,pref,seller,U,low=(data[k] for k in ('e','pref','seller','uniform','lower'));U=U<5
    with np.load(base/f'{split}-scores.npz') as data:score=data['context']
    money=base/f'{split}-exact-rents.npz'
    if not money.exists():
        money=Path(__file__).resolve().parent/'exact-prices'/money.name
    with np.load(money) as data:rent=data['rent']
    n,m=e.shape[1:];original=da(pref,seller,U);ix,row=np.indices(original.shape)
    rank_old=pref[ix,row,np.minimum(original,m-1)]
    price_old=rent[ix,np.minimum(original,m-1)]
    initial=low & (original<m)
    removed=initial[:,:,None] & U & (rent[:,None,:]>price_old[:,:,None])
    retained=~removed
    assert not (removed[ix,row,np.minimum(original,m-1)]&initial).any()
    scores,_=rank_slates(e,keys,score,None);graph=np.empty_like(e)
    for t in range(len(e)):
        pre=np.where(original[t]<m,original[t],-1)
        graph[t],_,_=repair_with_freezes(e[t],scores[t],U[t],low[t],pre)
        # Closure decisions, stable original offers and every editable higher-income list
        # are identical when the lower rows voluntarily remove more costly choices.
        rerun,_,_=repair_with_freezes(e[t],scores[t],U[t]&retained[t],low[t],pre)
        assert np.array_equal(rerun,graph[t]&retained[t])
    assert np.array_equal(graph.sum(2),U.sum(2)) and np.array_equal(graph[low],U[low])
    # Original held edges survive all deletions; no new blocks can be introduced by deletions.
    # Verify original stability on the restricted original graph directly.
    held_rank=np.full((len(e),m),128,int)
    b,r=np.nonzero(original<m);held_rank[b,original[b,r]]=seller[b,r,original[b,r]]
    owned_rank=np.where(original<m,rank_old,128)
    assert not np.any((U & retained) & (pref<owned_rank[:,:,None]) & (seller<held_rank[:,None,:]))
    with_veto=graph & retained
    assert np.array_equal(graph[~low],with_veto[~low])
    mu_guard,mu_veto=(da(pref,seller,vis) for vis in [graph,with_veto])
    def prices(mu):return rent[ix,np.minimum(mu,m-1)]
    def ranks(mu):return np.where(mu<m,pref[ix,row,np.minimum(mu,m-1)],128)
    rank1,rank2=ranks(mu_guard),ranks(mu_veto)
    improved1=initial & (rank1<rank_old);improved2=initial & (rank2<rank_old)
    assert not (initial & ((mu_veto==m)|(rank2>rank_old)|(prices(mu_veto)>price_old))).any()
    assert not (initial & ((mu_guard==m)|(rank1>rank_old))).any()
    assert not (improved1&~improved2&(mu_veto!=original)).any()
    shared={'removed_recipients':removed.any(2).sum(1),'removed_choices':removed.sum((1,2)),
            'prior_improvements_forgone':(improved1 & ~improved2).sum(1),
            'newly_improving':(~improved1 & improved2).sum(1),
            'improved_both_different_unit':(improved1 & improved2 & (mu_veto!=mu_guard)).sum(1),
            'improved_both_same_unit':(improved1 & improved2 & (mu_veto==mu_guard)).sum(1),
            'old_matched_switch_guard_to_veto_worse':(initial&(rank2>rank1)).sum(1),
            'old_matched_switch_guard_to_veto_better':(initial&(rank2<rank1)).sum(1)}
    rows=[]
    for mode,mu in [('guard',mu_guard),('veto',mu_veto)]:
        matched=mu<m
        rank=ranks(mu);price=prices(mu)
        for t,(seed,design,month,borough) in enumerate(keys):
            new=low[t]&~initial[t]&matched[t]
            rec=dict(seed=seed,design=design,month=month,borough=borough,mode=mode,
                     n=n,n_lower=int(low[t].sum()),n_higher=int((~low[t]).sum()),
                     matched=int(matched[t].sum()),matched_lower=int((matched[t]&low[t]).sum()),
                     matched_higher=int((matched[t]&~low[t]).sum()),original_lower=int(initial[t].sum()),
                     original_lost=int((initial[t]&~matched[t]).sum()),
                     worse_rank=int((initial[t]&(rank[t]>rank_old[t])).sum()),
                     higher_rent=int((initial[t]&(price[t]>price_old[t])).sum()),
                     strict_improvements=int((initial[t]&(rank[t]<rank_old[t])).sum()),
                     newly_placed_lower=int(new.sum()))
            rec.update({k:int(v[t]) for k,v in shared.items()})
            rows.append(rec)
    output=out/f'{split}-rent-veto-counts.csv.gz'
    with output.open('wb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as zipped:
        with io.TextIOWrapper(zipped,newline='') as fp:
            wr=csv.DictWriter(fp,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
    summary=summarize(rows,metadata['weights'])
    checks={'exact_rent_file_sha256':hashlib.sha256(money.read_bytes()).hexdigest(),
            'markets':len(keys),'summary':summary}
    (out/f'{split}-rent-veto.json').write_text(json.dumps(checks,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--split',nargs='+',default=['summer','winter']);a=p.parse_args()
    for split in a.split:
        vals=run(a.base,a.out,split)
        for design,z in vals.items():print(split,design,z['veto']['raw'],z['veto_minus_guard'],flush=True)
