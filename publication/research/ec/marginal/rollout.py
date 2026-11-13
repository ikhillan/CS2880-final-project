"""Random independent rollout, stratified by exact number switched within a market."""
import argparse
import hashlib
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

cli=argparse.ArgumentParser(description=__doc__)
for flag in ('ranks','meta','menus','out'):
    cli.add_argument('--'+flag,required=True,type=Path)
a=cli.parse_args()
d=Path(__file__).resolve().parent
meta=json.loads(a.meta.read_text())
with a.ranks.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()==meta['sha256']
with tempfile.TemporaryDirectory() as td:
    td=Path(td)
    z=dict(np.load(a.ranks))
    with (td/'data.bin').open('wb') as f:
        f.write(struct.pack('=i',len(z['lower'])))
        for i in range(len(z['lower'])):
            for arr in [z['pref'][i,6],z['seller'][i],z['uniform'][i],z['transparent'][i,0],z['transparent'][i,2],z['lower'][i]]:
                f.write(np.ascontiguousarray(arr,dtype=np.uint8).tobytes())
    subprocess.run(['c++','-O3','-std=c++17',str(d/'rollout.cpp'),'-o',str(td/'rollout')],check=True)
    subprocess.run([str(td/'rollout'),str(td/'data.bin'),str(td/'rows.csv')],check=True)
    raw=pd.read_csv(td/'rows.csv')
a.out.mkdir(exist_ok=True,parents=True)
raw.to_csv(a.out/'rollout_counts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
menus=pd.read_csv(a.menus)
end=raw[raw.switched==raw.higher_n].merge(menus[['market','weight','joint']],on=['market','weight'],validate='1:1')
assert (end.perm_sum_delta==24*end.joint).all() and (raw[raw.switched==0].perm_sum_delta==0).all()
ps=[0,.01,.025,.05,.1,.2,.35,.5,.75,1.]
weights=np.tile([meta['raw_borough_weights'][b] for b in meta['boroughs']],3);weights/=weights.sum()
freq=np.vstack([np.ones(100),np.random.default_rng(79249).multinomial(100,np.full(100,.01),size=5000)])
res={}
for p in ps:
    raw['expected']=raw.perm_sum_delta/24*binom.pmf(raw.switched,raw.higher_n,p)
    totals=raw.groupby(['market','weight']).agg(num=('expected','sum'),den=('lower_n','first')).reset_index()
    for (name,w),g in totals.groupby([np.array(meta['designs'])[totals.market//12%2], 'weight']):
        a0=g.sort_values('market')
        ratio=(freq@a0.num.to_numpy().reshape(100,12))/(freq@a0.den.to_numpy().reshape(100,12))
        v=100*ratio@weights
        res.setdefault(name+'_'+str(w),[]).append({'p':p,'pp':float(v[0]),'pointwise95':np.quantile(v[1:],[.025,.975]).tolist()})
(a.out/'rollout.json').write_text(json.dumps({'permutations_per_market':24,'market_draws':int(raw.market.nunique()),'curves':res},indent=2)+'\n')
