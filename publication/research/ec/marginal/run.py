"""Evaluate individual disclosure menus and compare isolated effects with a full-group switch."""
import argparse
import hashlib
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--ranks',type=Path,required=True)
p.add_argument('--meta',type=Path,required=True)
p.add_argument('--gpu-counts',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
args=p.parse_args()
root=Path(__file__).resolve().parent
meta=json.loads(args.meta.read_text())
with args.ranks.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()==meta['sha256']
a=np.load(args.ranks)
args.out.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp)
    subprocess.run(['c++','-O3','-std=c++17',str(root/'menus.cpp'),'-o',str(tmp/'menus')],check=True)
    with (tmp/'tensors.bin').open('wb') as f:
        f.write(struct.pack('=i',len(a['lower'])))
        pref=a['pref'][:,6].copy(); seller=a['seller']; uniform=a['uniform']; score=a['transparent']; low=a['lower']
        for i in range(len(low)):
            for arr in [pref[i],seller[i],uniform[i],score[i,0],score[i,2],low[i]]:
                f.write(np.ascontiguousarray(arr,dtype=np.uint8).tobytes())
    subprocess.run([str(tmp/'menus'),str(tmp/'tensors.bin'),str(tmp/'rows.csv')],check=True)
    rows=pd.read_csv(tmp/'rows.csv')
rows['block']=rows.market//24
rows['design']=np.array(meta['designs'])[rows.market//12%2]
rows['month']=np.array(meta['months'])[rows.market//4%3]
rows['borough']=np.array(meta['boroughs'])[rows.market%4]
gpu=pd.read_csv(args.gpu_counts)
rows['isolated']=rows.isolated_gain-rows.isolated_loss
rows['joint']=rows.final-rows.initial
rows['interaction']=rows.joint-rows.isolated
rows['net_discrepancy']=rows.joint!=rows.isolated
rows.to_csv(args.out/'counts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
cl=list(meta['boroughs']); months=list(meta['months'])
cells=[(m,b) for m in months for b in cl]; weights=np.array([meta['raw_borough_weights'][b] for m,b in cells]);weights/=weights.sum()
freq=np.vstack([np.ones(100),np.random.default_rng(264892).multinomial(100,np.full(100,.01),size=10000)])
stats={}
metrics=['isolated','expected_net','joint','interaction','isolated_loss','isolated_gain','joint_loss','joint_gain','newly_high_seats','released_high_seats','forward_gain','forward_loss','reverse_gain','reverse_loss']
for (d,w),df in rows.groupby(['design','weight']):
    idx=pd.MultiIndex.from_tuples([(b,m,city) for b in range(100) for m,city in cells],names=['block','month','borough'])
    table=df.set_index(idx.names).reindex(idx)
    assert not table.isna().any().any()
    denomin=table.lower_n.to_numpy().reshape(100,12)
    denominator=freq@denomin
    out={}
    for k in metrics:
        vals=freq@table[k].to_numpy().reshape(100,12)
        v=100*(vals/denominator)@weights
        out[k]={'pp':float(v[0]),'mc95':np.quantile(v[1:],[.025,.975]).tolist()}
    out['raw_markets_with_nonzero_interaction']=int(df.net_discrepancy.sum());stats[f'{d}_{w}']=out
# Match the GPU counts by actual keys, including the separate scoring and preference coefficients.
keys=['block','design','month','borough','weight']
sub=gpu[(gpu.w_p==4)&(gpu.budget=='k5')&gpu.policy_w_p.isin([1,4])].rename(columns={'policy_w_p':'weight'})
merged=rows.merge(sub,on=keys,validate='one_to_one',suffixes=('', '_gpu'))
assert len(merged)==len(rows)
for x,y in [('final','UT_matched_lower'),('initial','UU_matched_lower'),('lower_n','lower_n_gpu')]:
    assert (merged[x]==merged[y]).all(),(x,y)
report={'design_sha256':hashlib.sha256((root/'DESIGN.md').read_bytes()).hexdigest(),
        'markets':int(rows.market.nunique()),'market_scorer_comparisons':len(rows),
        'singleton_probes':int(rows[rows.weight==1].probes.sum()),
        'independent_full_list_checks_first_block':int(rows[rows.weight==1].direct_checks.sum()),
        'units':'counts per 100 lower-income renter draws standardized by borough and month; raw discrepancy counts are unweighted',
        'panels':stats}
(args.out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
