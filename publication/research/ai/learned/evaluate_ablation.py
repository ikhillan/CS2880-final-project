"""Compare the equal-parameter networks using paired market outcomes and fixed cell weighting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import da,make_mask


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(base,out,split):
    output=[]
    meta=json.loads((base/f'{split}.json').read_text());keys=meta['keys'];spec=json.loads((base/f'{split}-matched.json').read_text())
    assert sha(base/f'{split}-matched.npz')==spec['prediction_sha'];assert sha(base/f'{split}.npz')==spec['data_sha'];assert sha(out/'ablation.py')==spec['ablation_sha']
    for f,d in spec['weights'].items():assert sha(out/'models'/f)==d
    with np.load(base/f'{split}.npz') as z:e,pref,seller,y=(z[k] for k in ['e','pref','seller','y'])
    with np.load(base/f'{split}-matched.npz') as z:ms,mo=z['matched'],z['matched_outside']
    with np.load(base/f'{split}-scores.npz') as z:cs,co=z['context'],z['context_outside']
    valid=y<64;safe=np.minimum(y,63);g=np.arange(len(e))[:,None];r=np.arange(64)[None,:]
    for name,x,o in [('matched',ms,mo),('context',cs,co)]:
        mask=np.stack([make_mask(s,allowed,key,False) for s,allowed,key in zip(x,e,keys)]);matched=da(pref,seller,mask)<64
        hit=mask[g,r,safe]&valid;assert np.all(hit.sum(1)<=matched.sum(1));loss=-np.where(valid,x[g,r,safe],o)
        for idx,(seed,design,month,borough) in enumerate(keys):
            output.append(dict(seed=seed,design=design,month=month,borough=borough,model=name,matched=int(matched[idx].sum()),hits=int(hit[idx].sum()),oracle=int(valid[idx].sum()),nll=float(loss[idx].sum())))
    df=pd.DataFrame(output);est=[];rng=np.random.default_rng(291303);nseed=meta['blocks'];weights=meta['weights'];boot=rng.multinomial(nseed,np.full(nseed,1/nseed),size=5000);counts=np.vstack([np.ones(nseed),boot]);
    cells=sorted(set((k[2],k[3]) for k in keys));w=np.array([weights[b] for _,b in cells]);w/=w.sum();seed=sorted(df.seed.unique());ix=pd.MultiIndex.from_tuples([(s,m,b) for s in seed for m,b in cells],names=['seed','month','borough'])
    for design,frame in df.groupby('design'):
        result={}
        for model,table in frame.groupby('model'):
            z=table.set_index(list(ix.names)).reindex(ix)[['matched','hits','oracle','nll']].to_numpy().reshape(nseed,len(cells),4);t=np.einsum('as,scf->acf',counts,z,optimize=True)
            result[model]={'total':100*(t[:,:,0]/(nseed*64))@w,'hit':100*(t[:,:,1]/t[:,:,2])@w,'nll':(t[:,:,3]/(nseed*64))@w}
        for metric in ('total','hit','nll'):
            dif=result['context'][metric]-result['matched'][metric];lo,hi=np.quantile(dif[1:],[.025,.975]);est.append(dict(split=split,design=design,metric=metric,ownrow=result['matched'][metric][0],context=result['context'][metric][0],difference=dif[0],lo=lo,hi=hi))
    return df.assign(split=split),est


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();allrows=[];ests=[]
    for split in ['summer','winter']:
        rows,vals=run(a.data,a.out,split);allrows.append(rows);ests.extend(vals)
    pd.concat(allrows).to_csv(a.out/'ablation-counts.csv.gz',index=False,compression={'method':'gzip','mtime':0});pd.DataFrame(ests).to_csv(a.out/'ablation-contrasts.csv',index=False)
