"""Paired identity-level replay of protected, quota-preserving learned shortlists."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from guardrail import repair_with_freezes, threat_mask, certificate

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'learned'))
from evaluate import da, keyed_rng, make_mask


def summarize(rows, weights):
    metrics = {'low_match_rate':'lower_n','high_match_rate':'higher_n','total_match_rate':'n',
               'low_loss_rate':'lower_n', 'low_gain_rate':'lower_n','frozen_high_rate':'higher_n',
               'high_switch_rate':'higher_n','eligible_baseline_low_retained_rate':'baseline_low_matched'}
    numer = dict(low_match_rate='low_matched',high_match_rate='high_matched',total_match_rate='matched',
                 low_loss_rate='low_loss',low_gain_rate='low_gain',frozen_high_rate='frozen_high',
                 high_switch_rate='high_switch',eligible_baseline_low_retained_rate='retained')
    slices = {}
    for (design,mode) in sorted({(r['design'],r['mode']) for r in rows}):
        subset=[r for r in rows if r['design']==design and r['mode']==mode]
        other=[r for r in rows if r['design']==design and r['mode']=='direct']
        assert [(r['seed'],r['month'],r['borough']) for r in subset] == [(r['seed'],r['month'],r['borough']) for r in other]
        ss = {k:sum(row[k] for row in subset) for k in ('n','lower_n','higher_n','baseline_low_matched','matched','low_matched',
                   'high_matched','low_loss','low_gain','frozen_high','high_switch','unguarded_bound')}
        ss.update(markets=len(subset), needs_freezes=sum(row['frozen_high']>0 for row in subset),
                  complete_freeze=sum(row['frozen_high']==row['higher_n'] and row['higher_n']>0 for row in subset),
                  cascading_freezes=sum(row['rounds']>1 for row in subset),max_rounds=max(row['rounds'] for row in subset))
        blocks=sorted({r['seed'] for r in subset})
        cells=sorted({(r['month'],r['borough']) for r in subset})
        w=np.array([weights[b] for _,b in cells]);w/=w.sum()
        rng=np.random.default_rng(441903)
        boot=rng.multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=5000)
        freq=np.vstack((np.ones(len(blocks)),boot))
        def standardized(group):
            index={(r['seed'],r['month'],r['borough']):r for r in group}
            fields=sorted(set(metrics.values()) | set(numer.values()))
            arr=np.array([[[index[seed,m,b][k] for k in fields] for m,b in cells] for seed in blocks])
            totals=np.einsum('bs,scf->bcf',freq,arr,optimize=True)
            vals={}
            for name,den in metrics.items():
                base=totals[:,:,fields.index(den)]
                if (base==0).any():
                    continue
                vals[name]=100*((totals[:,:,fields.index(numer[name])]/base)@w)
            return vals
        curr=standardized(subset);base=standardized(other)
        ss['standardized']={}
        for name in sorted(curr.keys() & base.keys()):
            vals=curr[name];delta=vals-base[name]
            ss['standardized'][name]=dict(rate=float(vals[0]),change_pp=float(delta[0]),
                                         paired_95ci_pp=np.quantile(delta[1:],[.025,.975]).tolist())
        slices[f'{design}:{mode}']=ss
    return slices


def rank_slates(e,keys,score,transparent):
    tr=transparent
    tie_scores=np.zeros_like(e,int)
    direct=np.zeros_like(e)
    n,m=e.shape[1:]
    for t,key in enumerate(keys):
        if score is None:
            tie_scores[t]=-tr[t].astype(int)
            direct[t]=tr[t]<5
            continue
        rng=keyed_rng(key);rng.permutation(n);cols=rng.permutation(m)
        order=np.argsort(np.where(e[t][:,cols],-score[t][:,cols],np.inf),axis=1,kind='stable')
        ir=np.argsort(order,axis=1)
        tie_scores[t][:,cols]=-ir
        direct[t]=e[t] & (tie_scores[t]>-5)
        assert np.array_equal(direct[t],make_mask(score[t],e[t],key,False))
    return tie_scores,direct


def run(base, dest, split, name):
    with np.load(base/f'{split}.npz') as z:
        e,pref,seller,U,low,tr=(z[k] for k in ('e','pref','seller','uniform','lower','transparent'))
        U=U<5
    keys=json.loads((base/f'{split}.json').read_text())
    if name=='transparent':
        score=None
    else:
        with np.load(base/f'{split}-scores.npz') as z:score=z[name]
    initial=da(pref,seller,U); n,m=e.shape[1:]
    tie_scores,direct=rank_slates(e,keys['keys'],score,tr)
    direct=np.where(low[:,:,None],U,direct)
    rows=[]
    for mode in ['direct','blind','known']:
        graph=direct.copy();frozen=np.zeros(low.shape,bool)
        rounds=np.zeros(len(e),int);bounds=np.zeros(len(e),int)
        for t, key in enumerate(keys['keys']):
            pre=np.where(initial[t]==m,-1,initial[t])
            pri=seller[t] if mode=='known' else None
            dangerous=threat_mask(e[t],low[t],pre,pri)
            bounds[t]=certificate(dangerous&direct[t])
            if mode!='direct':
                graph[t],fixed,rounds[t]=repair_with_freezes(e[t],tie_scores[t],U[t],low[t],pre,pri)
                frozen[t]=fixed & ~low[t]
        matched=da(pref,seller,graph) < m
        base_match=initial < m
        assert np.array_equal(graph.sum(2),U.sum(2)) and np.array_equal(graph[low],U[low])
        loss=low & base_match & ~matched;gain=low & ~base_match & matched
        assert mode=='direct' or not loss.any()
        assert np.all(loss.sum(1)<=bounds)
        for t,(seed, design, month, borough) in enumerate(keys['keys']):
            rows.append(dict(split=split,score=name,mode=mode,seed=seed,design=design,month=month,borough=borough,
                             n=n,lower_n=int(low[t].sum()),higher_n=int((~low[t]).sum()),
                             baseline_low_matched=int((base_match[t]&low[t]).sum()),
                             retained=int((matched[t]&base_match[t]&low[t]).sum()),
                             matched=int(matched[t].sum()),low_matched=int((matched[t]&low[t]).sum()),
                             high_matched=int((matched[t]&~low[t]).sum()),low_loss=int(loss[t].sum()),
                             low_gain=int(gain[t].sum()),frozen_high=int(frozen[t].sum()),
                             high_switch=int((~low[t] & (graph[t]!=U[t]).any(1)).sum()),
                             unguarded_bound=int(bounds[t]),rounds=int(rounds[t])))
    detail=dest/f'{split}-{name}.csv.gz'
    with detail.open('wb') as f, gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as gz:
        import io
        with io.TextIOWrapper(gz,newline='') as out:
            w=csv.DictWriter(out,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    out=summarize(rows,keys['weights'])
    report=dict(split=split,score=name,market_hash=keys['sha256'],summary=out,
                detail_sha256=hashlib.sha256(detail.read_bytes()).hexdigest())
    if name!='transparent':report['score_hash']=hashlib.sha256((base/f'{split}-scores.npz').read_bytes()).hexdigest()
    (dest/f'{split}-{name}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(split,name,[(k,v['low_loss'],v['low_gain'],v['frozen_high'],v['high_switch'],v['needs_freezes']) for k,v in out.items()],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--split',nargs='+',default=['summer','winter']);p.add_argument('--score',nargs='+',default=['transparent','pointwise','context']);a=p.parse_args()
    for split in a.split:
        for name in a.score:run(a.base,a.out,split,name)
