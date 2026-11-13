"""Independent simultaneous DA for each shortlist selected by a GPU ensemble."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run import Market

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inputs','features','fit','archived'):p.add_argument(k,type=Path)
    a=p.parse_args();m=Market(a.inputs,2026,positive_scores=True)
    actual=pd.read_csv(a.archived/'offer-counts.csv').query('policy in ["blind", "informed"]').set_index(['policy','k'],verify_integrity=True)
    col=np.load(a.features/'test-col.npy');full=m.batch(m.cap);checked=0
    for name in ['blind','informed']:
        with np.load(a.fit/f'{name}-ensemble.npz') as f:prob=f['p'][m.a,col]
        tie=np.random.default_rng(118844).permutation(len(col));order=np.lexsort((tie,-prob,m.a))
        # Traverse applicants in reported order on a compressed graph containing only retained choices.
        for k in [1,3,5,10]:
            retained=order[np.arange(len(order))-m.offset[m.a[order]]<k];idx=np.sort(retained)
            sub=copy.copy(m);sub.a=m.a[idx];sub.school=m.school[idx];sub.priority=m.priority[idx]
            sub.degree=np.bincount(sub.a,minlength=sub.n);sub.offset=np.r_[0,sub.degree.cumsum()]
            sol=sub.batch(m.cap);orig=np.full(m.n,-1,np.int32);orig[sol>=0]=idx[sol[sol>=0]]
            r=actual.loc[(name,k)]
            assert r.placements==(orig>=0).sum()
            assert r.matched_loss==((full>=0)&(orig<0)).sum() and r.matched_gain==((full<0)&(orig>=0)).sum()
            assert r.exact_observed==((orig>=0)&(orig==m.recorded)).sum()
            assert r.observed_recovery==np.isin(m.recorded[m.recorded>=0],idx).sum()
            assert r.recovery==np.isin(full[full>=0],idx).sum()
            checked+=1
    receipts=[]
    for f in sorted((a.archived/'models').glob('*.json')):
        receipt=json.loads(f.read_text())
        for y in [2025,2026]:
            key=('train' if y==2025 else 'test')+'_features_sha'
            assert hashlib.sha256((a.features/f'features{y}.npz').read_bytes()).hexdigest()==receipt[key]
        assert hashlib.sha256(f.with_suffix('.pt').read_bytes()).hexdigest()==receipt['weights']
        receipts.append(receipt['job'])
    print(json.dumps(dict(checked_policy_quota_combinations=checked,verified_counts_per_combination=6,
                          verified_training_receipts=len(receipts),full_cohort=m.n,slurm_jobs=receipts),indent=2))
