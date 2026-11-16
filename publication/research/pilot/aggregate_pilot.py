"""Aggregate the original seven non-LLM arms on the full three-month pilot."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

GROUPS = ('lower_income','higher_income')
POLICIES = ('uniform_k5','transparent_k5','expensive_first_k5','uniform_q40',
            'transparent_q40','expensive_first_q40','full')
FEATURES = ('n','own_blocker_rate','can_hide_rate','no_feasible_listing','match_before','match_full_disclosure')


def counts(outcomes):
    values = []
    for group in GROUPS:
        row = outcomes[group]
        n = row['n']
        values.append([n,row['blocker_count'],row['can_hide_count'],
                       *[round(n*row[k]) if n else 0 for k in FEATURES[3:]]])
    return values


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--repo',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    a = cli.parse_args()
    sys.path.insert(0,str(a.repo.resolve()))
    from hiddenoptions.data import borough_weights,load_renters,DEV_MONTHS
    weights = borough_weights(load_renters())
    folder = a.repo/'results/stage1_pilot'
    runs = [json.loads(line) for line in (folder/'runs.jsonl').read_text().splitlines()]
    markets = [json.loads(line) for line in (folder/'markets.jsonl').read_text().splitlines()]
    ids = [x['market_id'] for x in markets]
    assert len(set(ids))==len(ids)==600
    cells = sorted({(r['borough'],r['month']) for r in markets if r['status']=='ok'})
    w = np.array([weights[b]/len(DEV_MONTHS) for b,m in cells])
    coverage = float(w.sum())
    w = w/coverage
    report = dict(commit='981ed97f6b1c517b8aa9a785531c2592a700133c',
                  config_sha256=sha256((a.repo/'configs/stage1_pilot.json').read_bytes()).hexdigest(),
                  design='Stage 1 on June-August 2026; 20 shared seed blocks within each design; income groups in lower,higher order',
                  bootstrap={'samples':10000,'type':'paired full seed blocks with replacement','seed':731911},
                  status=dict(Counter(m['status'] for m in markets)),coverage=coverage,designs={})
    for variant in ('primary','enriched'):
        rows = [r for r in runs if r['variant']==variant]
        seeds = sorted({r['seed'] for r in rows})
        nums = np.full((len(seeds),len(cells),len(POLICIES),2,len(FEATURES)),-1,dtype=int)
        for r in rows:
            idx=(seeds.index(r['seed']),cells.index((r['borough'],r['month'])),POLICIES.index(r['exposure']))
            assert np.all(nums[idx]==-1)
            nums[idx] = counts(r['outcomes'])
        assert len(rows)==len(seeds)*len(cells)*len(POLICIES) and np.all(nums>=0)
        assert (nums[:,:,:,:,0]==nums[:,:,:1,:,0]).all()
        raw = nums.sum(axis=0)
        def normalize(table):
            return np.einsum('c,...cpgm->...pgm',w,table[...,1:]/table[...,:1])
        point = normalize(raw)
        totals = raw.sum(axis=0)
        pooled = totals[...,1:]/totals[...,:1]
        rates = {}
        for i,p in enumerate(POLICIES):
            rates[p] = {g:{metric:float(point[i,j,k]) for k,metric in enumerate(FEATURES[1:])}
                        for j,g in enumerate(GROUPS)}
        rng = np.random.default_rng(731911)
        draws = rng.multinomial(len(seeds), [1/len(seeds)]*len(seeds), 10000)
        # For the bootstrap only n and own-blocker count are required.
        resampled = normalize(np.einsum('rs,scpgm->rcpgm',draws,nums[...,:2]))[...,0]
        # Each compared policy uses the same market within a seed block.
        contrasts = {}
        for suffix in ('k5','q40'):
            t = POLICIES.index('transparent_'+suffix)
            u = POLICIES.index('uniform_'+suffix)
            treated = point[t,:,0]
            control = point[u,:,0]
            effect = treated-control
            samples = resampled[:,t]-resampled[:,u]
            T = float(effect[0]-effect[1])
            contrasts[suffix]=dict(delta_lower=float(effect[0]),delta_higher=float(effect[1]),T=T,
                                   paired_mc_interval_95=np.quantile(samples[:,0]-samples[:,1],[.025,.975]).tolist(),
                                   gap_control=float(control[0]-control[1]),gap_treated=float(treated[0]-treated[1]))
            # Recompute group-switch allocation of the signed T from all four arms.
            switch = np.zeros((len(cells),4,2,2),int)
            for m in markets:
                if m['variant']!=variant or m['status']!='ok':
                    continue
                cell=cells.index((m['borough'],m['month']))
                arm=m['group_switch']['transparent_vs_uniform_'+suffix]
                for k,combo in enumerate(('AA','BA','AB','BB')):
                    for j,g in enumerate(('lower','higher')):
                        switch[cell,k,j] += [arm[combo][g+'_count'],arm[combo][g+'_n']]
            groups = np.einsum('c,cpg->pg',w,switch[...,0]/switch[...,1])
            z = groups[:,0]-groups[:,1]
            sl=(z[1]-z[0]+z[3]-z[2])/2
            sh=(z[2]-z[0]+z[3]-z[1])/2
            assert abs(sl+sh-T)<1e-12
            contrasts[suffix].update(switch_lower=float(sl),switch_higher=float(sh))
        report['designs'][variant]=dict(seeds=seeds, markets=len(seeds)*len(cells),
                  group_sizes=raw.sum(axis=0)[0,:,0].tolist(),
                  pooled_uniform_k5={g:dict(zip(FEATURES[1:],pooled[0,j].tolist())) for j,g in enumerate(GROUPS)},
                  policies=rates,transparent_minus_uniform=contrasts)
    report['realized_rows']=len(runs)
    report['skipped_arms']=dict(Counter(p for m in markets for p in m.get('skipped_exposures',[])))
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
