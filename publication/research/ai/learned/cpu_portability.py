"""How often does changing the inference device change a frozen top-five decision?"""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate import da,make_mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--cpu',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();split='summer'
    keys=json.loads((a.data/f'{split}.json').read_text())['keys'];rows=[]
    with np.load(a.data/f'{split}.npz') as z:e,pref,seller=(z[k] for k in ('e','pref','seller'))
    with np.load(a.cpu/f'{split}-scores.npz') as cpu,np.load(a.data/f'{split}-scores.npz') as gpu:
        for name in ('pointwise','context'):
            s,t=cpu[name],gpu[name];d=np.abs(s[e]-t[e]);do=np.abs(cpu[name+'_outside']-gpu[name+'_outside'])
            for anchor in [False,True]:
                masks=[];matches=[]
                for scores in (s,t):
                    m=np.stack([make_mask(x,E,k,anchor) for x,E,k in zip(scores,e,keys)]);masks.append(m);matches.append(da(pref,seller,m))
                a0,a1=matches;rows.append(dict(policy=name,anchored=anchor,n_markets=len(s),n_renters=int(e.shape[0]*e.shape[1]),
                    differing_menus=int(np.sum(np.any(masks[0]!=masks[1],axis=2))),different_assigned_partners=int((a0!=a1).sum()),
                    different_match_status=int(((a0<64)!=(a1<64)).sum()),delta_total=int((a0<64).sum()-(a1<64).sum()),
                    max_feasible_logp_delta=float(d.max()),q99_feasible_logp_delta=float(np.quantile(d,.99)),max_outside_logp_delta=float(do.max())))
    a.out.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__': main()
