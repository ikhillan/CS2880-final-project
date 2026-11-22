"""Verify shared aggregate outputs and the baselines from prior releases."""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path):
    with path.open() as f:
        return [{k: int(v) if v.removeprefix("-").isdigit() else v for k, v in row.items()}
                for row in csv.DictReader(f)]

def summarize(rows):
    return (len(rows),sum(r['missing_old_pairs']+r['rows_with_blockers']>0 for r in rows),
            sum(r['exits']>0 for r in rows),sum(r['exits'] for r in rows),
            sum(r['entrants'] for r in rows),max(r['exits'] for r in rows))

def check():
    fresh = defaultdict(list)
    labels, ceilings = defaultdict(int), defaultdict(int)
    old_register=read(ROOT.parent.parent/'behavior/field/offer-concordance.csv')
    for year in (2022,2023,2024):
        primary=read(ROOT/f'{year}.csv')
        nested=read(ROOT/f'nested-{year}.csv')
        test=read(ROOT/f'validation-{year}.csv')
        historical=read(ROOT/f'labels-{year}.csv')
        upper=read(ROOT/f'ceiling-{year}.csv')
        assert [len(data) for data in (primary,nested,test,historical,upper)]==[324,548,512,14,10]
        blind_headcounts={}
        for row in primary+nested+test:
            key=(row['split'],row['draw'])
            baseline=row['placed']-row['entrants']+row['exits']
            assert baseline==blind_headcounts.setdefault(key,baseline)
        for policy,budget in [('budget0',0),('budget1',1),('budget2',2)]:
            rows=[r for r in primary if r['split']=='design' and r['policy']==policy]
            assert len(rows)==16 and summarize(rows)[1]<=budget
        zero={(r['split'],r['draw']):r for r in primary if r['policy']=='budget0'}
        for r in nested:
            if r['policy']=='cert16':
                orig=zero[r['split'],r['draw']]
                assert all(r[k]==orig[k] for k in r if k!='policy')
        for r in test:fresh[r['policy']].append(r)
        for p in ('blind','rank'):
            subset=[r for r in historical if r['policy']==p]
            old=[r for r in old_register if r['year']==year and r['stratum'] in ('enrolled_at_initial_offer','enrolled_elsewhere','no_initial_offer')]
            for k,prev in [('april_schools_shown',p+'_hits'),('linked_placed',p+'_placed'),('linked','people')]:
                assert sum(r[k] for r in subset)==sum(r[prev] for r in old)
        for r in historical:
            for k in ['placed','exits','entrants','april_schools_shown','initial_offers_shown']:
                labels[r['policy'],k]+=r[k]
        for target in ('april_school','initial_offer'):
            rows=sorted((r for r in upper if r['target']==target),key=lambda r:r['support'])
            assert [r['support'] for r in rows]==[0,16,64,256,1024]
            assert all(a['additional_reachable']>=b['additional_reachable'] for a,b in zip(rows,rows[1:]))
            for r in rows:
                ceilings[r['support'],target]+=r['additional_reachable']
                if r['support']:
                    p='cert'+str(r['support'])
                    field='april_schools_shown' if target=='april_school' else 'initial_offers_shown'
                    assert sum(s[field] for s in historical if s['policy']=='blind')==r['present_at_baseline']
                    assert sum(s[field] for s in historical if s['policy']==p)<=r['present_at_baseline']+r['additional_reachable']
    assert summarize(fresh['cert1024'])==(768,187,83,87,91,2)
    assert summarize(fresh['control1024_0'])[3]==237115
    assert labels['cert1024','exits']==0 and labels['cert1024','entrants']==1
    assert labels['cert1024','april_schools_shown']-labels['blind','april_schools_shown']==-27
    assert labels['rank','april_schools_shown']-labels['blind','april_schools_shown']==3590
    assert [ceilings[n,'april_school'] for n in (0,16,64,256,1024)]==[4823,593,353,302,278]
    assert ceilings[1024,'initial_offer']==48
    print('Independent simulator validation, matched controls, historical labels and upper bounds reconcile.')
    print('Additional April schools reachable with hindsight by support size:',{n:ceilings[n,'april_school'] for n in (0,16,64,256,1024)})
if __name__=='__main__':
    check()
