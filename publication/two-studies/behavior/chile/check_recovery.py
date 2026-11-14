"""Check serial outputs by reconstructing lists and applying an independent, round-based DA."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def mix(x):
    with np.errstate(over='ignore'):
        x = np.asarray(x, dtype=np.uint64) + np.uint64(0x9e3779b97f4a7c15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return x ^ (x >> np.uint64(31))


def selected_programs(m, chosen):
    z = np.full(m.n, -1)
    z[chosen>=0] = m.j[chosen[chosen>=0]]
    return z


def signature(z):
    i = np.arange(len(z), dtype=np.uint64)
    x = (i << np.uint64(32)) | (z+1).astype(np.uint64)
    return str(int(np.bitwise_xor.reduce(mix(x))))


def policy(m, mask, eligible, rule):
    hidden=np.flatnonzero(~mask)
    # Binary prepared file is already in the full submitted order, with protected option last.
    first=np.full(m.n,m.ne,dtype=np.int64)
    np.minimum.at(first,m.i[hidden],hidden)
    eligible=eligible & (first<m.ne)
    answer=mask.copy()
    if rule in ('full','all_last','all_original'):
        answer[eligible[m.i]]=True
    elif rule!='binding':
        answer[first[eligible]]=True
    if rule not in ('last','before_right','all_last'):
        return answer,m.solve(answer)
    # Classify edges independently of the serial solver's per-row copy, then sort
    # by pupil, new pursuit group and original rank within group.
    group=np.zeros(m.ne,dtype=np.uint8)
    group[~mask]=1
    if rule=='before_right': group[m.j==m.ret[m.i]]=2
    order=np.lexsort((np.arange(m.ne),group,m.i))
    modified=copy.copy(m)
    modified.j=m.j[order]
    modified.pr=m.pr[order]
    local=modified.solve(answer[order])
    chosen=np.full(m.n,-1)
    chosen[local>=0]=order[local[local>=0]]
    return answer,chosen


def verify_year(y, market_type, numeric, outputs, evidence):
    m=market_type(numeric/f'{y}.bin')
    checks=pd.read_csv(outputs/f'school-recovery-{y}-checks.csv',dtype={'blind_hash':str,'rank_hash':str}).set_index('response')
    sweep=pd.read_csv(outputs/f'school-recovery-{y}.csv')
    tails=pd.read_csv(outputs/f'school-tail-{y}-arms.csv',dtype={'assignment_hash':str}).set_index(['response','policy'])
    ranks=pd.read_csv(evidence/'audit/chile/school-rank-thresholds.csv').query('year==@y and group=="all" and k==1').iloc[0]
    original=pd.read_csv(evidence/'audit/chile/school-capacity-bracket.csv').query('year==@y and scenario=="baseline"').iloc[0]
    baseline=sweep.query('response=="full" and recovery==0').iloc[0]
    assert (baseline.blind_first,baseline.rank_first)==(ranks.submitted_blind,ranks.submitted_rank)
    assert (baseline.rank_placed-baseline.blind_placed,baseline.rank_history-baseline.blind_history)==(original.placed_delta,original.hist_delta)
    key=mix(np.uint64(202611140314) ^ (np.uint64(y)<<np.uint64(32)))
    quarter=mix(key + np.arange(m.n,dtype=np.uint64)) >> np.uint64(62)
    participants=quarter<2
    admin=m.app
    totals=0
    for name in ('blind','rank'):
        mask=np.fromfile(numeric/f'{y}-{name}.mask',dtype=np.uint8).astype(bool)
        for rule in ('full','one'):
            shown,z=policy(m,mask,participants,rule)
            assert signature(selected_programs(m,z)) == checks.loc[rule,name+'_hash']
            row=sweep.query('response==@rule and recovery==.5 and pair==0 and half==0').iloc[0]
            assert (z>=0).sum()==row[name+'_placed']
            assigned=selected_programs(m,z)
            assert ((z>=0)&(z==m.at[:-1])&(assigned!=admin)).sum()==row[name+'_first']
            totals+=1
        for rule in ('binding','in_order','before_right','last','all_last','all_original'):
            shown,z=policy(m,mask,np.ones(m.n,dtype=bool),rule)
            row=tails.loc[(rule,name)]
            assert signature(selected_programs(m,z))==row.assignment_hash
            assert (z>=0).sum()==row.placed
            # Distinguish administratively appended from volunteered first choices.
            assert ((z>=0)&(z==m.at[:-1])&(selected_programs(m,z)!=admin)).sum()==row['first']
            assert (shown & (m.j==m.off[m.i])).sum()==row.history
            totals+=1
    return {'year':y,'policies_verified':totals,'applicant_assignments_checked_via_vectors':int(totals*m.n)}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for option in ['repo','early','late','directory']:cli.add_argument('--'+option,type=Path,required=True)
    args=cli.parse_args()
    base=args.repo/'publication'
    spec=importlib.util.spec_from_file_location('equity_transitions',base/'research/policy/equity/transitions.py')
    lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    result=[]
    for y in range(2022,2026):
        result.append(verify_year(y,lib.Market,args.early if y<2024 else args.late,args.directory,base/'two-studies'))
        print(result[-1],flush=True)
    (args.directory/'school-recovery-verification.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
