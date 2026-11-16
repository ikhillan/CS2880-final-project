"""Rehash splits and checkpoints and replay archived full-role counts with original serial DA."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from evaluate import make_mask


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path);p.add_argument('--data',type=Path);a=p.parse_args();root=a.root.resolve()
    data=a.data or root/'work/ec-ai-learn';out=Path(__file__).resolve().parent;sys.path.insert(0,str((a.repo or root/'work/todd-project').resolve()))
    from hiddenoptions.matching import Instance,deferred_acceptance,NOT_EDGE
    all_seeds=set();records={};checks=0
    for split in ('train','val','cal','summer','winter'):
        meta=json.loads((data/f'{split}.json').read_text());assert digest(data/f'{split}.npz')==meta['sha256'];seeds={k[0] for k in meta['keys']}
        assert all_seeds.isdisjoint(seeds);all_seeds|=seeds
        records[split]={'n_markets':len(meta['keys']),'n_blocks':len(seeds),'unavailable':len(meta['missing']),'tensor_sha':meta['sha256']}
        if split in ('train','val'):continue
        pred=json.loads((data/f'{split}-scores.json').read_text());assert digest(data/f'{split}-scores.npz')==pred['prediction_sha']
        assert pred['data_sha']==meta['sha256'];assert pred['code_sha']==digest(out/'train.py')
        for fn,hashval in pred['weights'].items():assert digest(out/'models'/fn)==hashval
        records[split].update(prediction_sha=pred['prediction_sha'],su_sha=digest(data/f'{split}-su.npz'),counts_sha=digest(out/f'{split}-counts.csv.gz'))
        frame=pd.read_csv(out/f'{split}-counts.csv.gz').set_index(['seed','design','month','borough','rollout','policy'])
        keyseed=[min(seeds),max(seeds)];selected=[i for i,k in enumerate(meta['keys']) if k[0] in keyseed]
        with np.load(data/f'{split}.npz') as z: e,pref,seller,u,U,T,low=(z[k][selected] for k in ('e','pref','seller','u','uniform','transparent','lower'))
        with np.load(data/f'{split}-scores.npz') as z: scores={k:z[k][selected] for k in ('context','pointwise')}
        with np.load(data/f'{split}-su.npz') as z:su=z['mask'][selected]
        for i,original in enumerate(selected):
            key=meta['keys'][original];ei=e[i];un=U[i]<5;lo=low[i]
            inst=Instance(np.where(ei,pref[i].astype(int),NOT_EDGE),np.where(ei,seller[i].astype(int),NOT_EDGE),np.arange(64),np.arange(64,128),np.ones(128,int),list(map(str,range(128))))
            baseline=deferred_acceptance(inst,un)[0]>=0
            for policy in ['uniform','transparent','transparent_anchor','pointwise','pointwise_anchor','context','context_anchor','su']:
                name=policy.split('_')[0]
                if policy=='uniform':mask=un
                elif policy=='transparent':mask=T[i]<5
                elif policy=='su':mask=su[i]
                else:mask=make_mask(u[i] if name=='transparent' else scores[name][i],ei,key,policy.endswith('_anchor'))
                for rollout in ('all','higher'):
                    actual=deferred_acceptance(inst,mask if rollout=='all' else np.where(lo[:,None],un,mask))[0]>=0
                    expected=[actual.sum(),(actual&lo).sum(),(baseline&~actual&lo).sum(),(~baseline&actual&lo).sum()]
                    rec=frame.loc[(*key,rollout,policy),['matched_total','matched_lower','joint_loss','joint_gain']].to_numpy()
                    assert np.array_equal(expected,rec);checks+=1
    (out/'audit.json').write_text(json.dumps({'partitions':records,'serial_paired_outcome_checks':checks,'seeds_disjoint':True,'checkpoint_and_prediction_chain_verified':True},indent=2)+'\n')
    print(checks,'serial DA comparisons passed')


if __name__=='__main__':main()
