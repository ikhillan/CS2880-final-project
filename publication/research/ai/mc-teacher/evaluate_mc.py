"""Paired comparison of Monte Carlo DA posteriors, frozen network, and fixed rules."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

def masks(counts,transparent_rank,keys,feasible,mod):
    values=counts.astype(float)+(64-transparent_rank.astype(float))/65
    return np.stack([mod.make_mask(s,e,key,False) for s,e,key in zip(values,feasible,keys)])

def compare(source,path,key_path,out,precision=None):
    import evaluate as mod
    receipt=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['output_sha256']
    partners=np.load(path)['partner']; meta=json.loads(key_path.read_text())
    assert partners.shape==(384,128,64)
    result=[]
    for split in ['summer','winter']:
        keys=json.loads((source/f'{split}.json').read_text())['keys']
        selected=[(i,k) for i,k in enumerate(meta) if k[0]==split]
        ids=[keys.index(k[1:]) for i,k in selected]; subset=[keys[i] for i in ids]
        samples=partners[[i for i,k in selected]]
        with np.load(source/f'{split}.npz') as a:
            e,tr,ur,low,y,pref,prior=[a[k][ids] for k in ['e','transparent','uniform','lower','y','pref','seller']]
        with np.load(source/f'{split}-scores.npz') as a:
            policies={name:np.stack([mod.make_mask(s,em,k,False) for s,em,k in zip(a[name][ids],e,subset)])
                      for name in ['context','pointwise']}
        policies.update(uniform=ur<5,transparent=tr<5)
        with np.load(source/f'{split}-su.npz') as a:policies['su']=a['mask'][ids]
        n=len(ids); market=np.arange(n)[:,None];renter=np.arange(64)[None,:]
        assert np.array_equal(mod.da(pref,prior,e),y)
        for draws in [16,64,128]:
            counts=np.zeros((n,64,65),int)
            for draw in range(draws):
                counts[market,renter,samples[:,draw,:]]+=1
            assert np.all(counts.sum(2)==draws) and not (counts[:,:,:64][~e]>0).any()
            policies['mc'+str(draws)]=masks(counts[:,:,:64],tr,subset,e,mod)
        if precision is not None:
            info=json.loads(precision.with_suffix('.json').read_text())
            assert info['input_sha256']==receipt['input_sha256']
            assert hashlib.sha256(precision.read_bytes()).hexdigest()==info['output_sha256']
            with np.load(precision) as values:
                for name in ('mc512','mc2048'):
                    c=values[name][[i for i,k in selected]]
                    assert np.all(c.sum(2)==int(name[2:])) and not (c[:,:,:64][~e]>0).any()
                    policies[name]=masks(c[:,:,:64],tr,subset,e,mod)
        for name,graph in policies.items():
            assert not (graph&~e).any() and np.array_equal(graph.sum(2),np.minimum(e.sum(2),5))
            z=mod.da(pref,prior,graph)
            for i,k in enumerate(subset):
                M=y[i];N=z[i];inside=(M<64)&graph[i,np.arange(64),np.minimum(M,63)]
                omits=(M<64)&~inside;K=set(M[omits]);T=set(N[omits])-{64}
                r=int(omits.sum());loss=(M<64)&(N==64); gain=(M==64)&(N<64)
                bnd=r-len(K&T)
                assert max(loss.sum(),gain.sum())<=bnd
                result.append(dict(split=split,seed=k[0],design=k[1],month=k[2],borough=k[3],
                    policy=name,full=int((M<64).sum()),hits=int(inside.sum()),total=int((N<64).sum()),
                    lower_n=int(low[i].sum()),lower_total=int(((N<64)&low[i]).sum()),
                    gross_loss=int(loss.sum()),gross_gain=int(gain.sum()),refined_bound=bnd))
    pd.DataFrame(result).to_csv(out,index=False,compression='gzip')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('learned_code','data','draws','keys','out'):p.add_argument(k,type=Path)
    p.add_argument('--precision',type=Path)
    a=p.parse_args();sys.path.insert(0,str(a.learned_code.resolve()));compare(a.data,a.draws,a.keys,a.out,a.precision)
