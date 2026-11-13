"""Independent paired Monte Carlo precision check, conditional on the eight measured markets."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import replay as R
p=argparse.ArgumentParser();p.add_argument('--data',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
markets=R.build_markets(R.read_data(a.data),0);pools,scores=R.fit(markets,False);rows=[]
for m in markets:
    if m.wave not in R.TEST:continue
    pool=pools[m.wave];learn=R.slates(scores[m.wave],pool,5)
    vals=[]
    for seed in range(4096):
        rng=np.random.default_rng([61943,m.wave,seed]);random=R.slates(rng.random(pool.shape),pool,5)
        order,prior=R.refine(m,seed+50000)
        full=R.allocate(order,prior,m.acceptable);i=np.flatnonzero(full>=0)
        arm=[]
        for g in [learn,random]:
            partner=R.allocate(order,prior,m.acceptable&g)
            arm.append([g[i,full[i]].mean(),np.mean(partner>=0),(m.mutual&g).sum()/m.mutual.sum()])
        vals.append(100*(np.array(arm[0])-arm[1]))
    rows.append(np.array(vals))
vals=np.array(rows).mean(axis=0)
result={'draws_per_wave':4096,'wave_count':len(rows),'delta_pp':dict(zip(['recovery','placement','mutual_exposed'],vals.mean(0))),
        'conditional_mcse_pp':dict(zip(['recovery','placement','mutual_exposed'],vals.std(axis=0,ddof=1)/64))}
(a.out/'mc-precision.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
