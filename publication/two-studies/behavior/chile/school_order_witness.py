"""Certify distinct realized-order reversal witnesses on two displays of the full option graph."""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from check_recovery import policy, selected_programs, signature


def links(m, old, new):
    # At a course, cancel its common holders. Pair the remaining old and new
    # holders in increasing applicant index, up to the smaller number of holders.
    before = selected_programs(m, old)
    after = selected_programs(m, new)
    departing = np.flatnonzero((before>=0) & (before!=after))
    incoming = np.flatnonzero((after>=0) & (before!=after))
    buckets = lambda students, programs: pd.DataFrame({'i':students, 'j':programs[students]}).sort_values(['j','i'])
    a = buckets(departing, before)
    b = buckets(incoming, after)
    for bucket in (a,b): bucket['k'] = bucket.groupby('j').cumcount()
    joined = a.merge(b,on=['j','k'],validate='one_to_one',suffixes=('_a','_b'))
    result=np.full(m.n,-1,dtype=np.int32)
    result[joined.i_a]=joined.i_b
    return result


def precedes(x, y, visible):
    # Negative edge index denotes an outside assignment. Visible options precede
    # the hidden suffix. Both orderings use the submitted order inside a group.
    can_compare = (x>=0) & (y>=0)
    vx,vy=visible[np.maximum(x,0)],visible[np.maximum(y,0)]
    return (x>=0) & (y<0) | (can_compare & ((vx & ~vy) | ((vx==vy)&(x<y))))


def trace(m, x, y, mask_x, mask_y, reversal):
    starts=np.flatnonzero((x>=0) & (y<0))
    successor=links(m,x,y)
    x_in_x=precedes(x,y,mask_x); x_in_y=precedes(x,y,mask_y)
    seen=set();endpoints=set(); lengths=Counter()
    for initial in starts:
        i=initial; length=0
        while True:
            assert i not in seen
            seen.add(i)
            assert x_in_x[i] and x_in_y[i]
            k=successor[i]
            assert k>=0 and m.j[y[k]]==m.j[x[i]]
            assert m.pr[y[k]]<m.pr[x[i]] and x_in_x[k]
            length+=1
            if reversal[k]:
                assert k not in endpoints and k not in seen
                endpoints.add(k);seen.add(k);lengths[length]+=1
                break
            assert x_in_y[k]; i=k
    assert len(starts)==len(endpoints)==sum(lengths.values())
    return endpoints,lengths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','early','late','directory'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('school_market',a.repo/'publication/research/policy/equity/transitions.py')
    lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    summary=[];dist=[]
    for year in range(2022,2026):
        numeric=a.early if year<2024 else a.late
        m=lib.Market(numeric/f'{year}.bin')
        masks=[np.fromfile(numeric/f'{year}-{s}.mask',np.uint8).astype(bool) for s in ('blind','rank')]
        za,zb=(policy(m,mask,np.ones(m.n,dtype=bool),'all_last')[1] for mask in masks)
        validated=pd.read_csv(a.directory/f'school-tail-{year}-arms.csv',dtype={'assignment_hash':str}).query('response=="all_last"').set_index('policy')
        for n,z in [('blind',za),('rank',zb)]: assert signature(selected_programs(m,z))==validated.loc[n,'assignment_hash']
        rev=precedes(za,zb,masks[0]) & precedes(zb,za,masks[1])
        assert np.all((za[rev]>=0)&(zb[rev]>=0))
        assert np.all(np.logical_or(masks[0][za[rev]]!=masks[1][za[rev]],masks[0][zb[rev]]!=masks[1][zb[rev]]))
        losses,lens_l=trace(m,za,zb,masks[0],masks[1],rev)
        gains,lens_g=trace(m,zb,za,masks[1],masks[0],rev)
        summary.append(dict(year=year,applicants=m.n,matched_both=int(((za>=0)&(zb>=0)).sum()),reversals=int(rev.sum()),
                            losses=len(losses),gains=len(gains),distinct_loss_witnesses=len(losses),distinct_gain_witnesses=len(gains),
                            shared_terminal_witnesses=len(losses&gains),longest_loss_path=max(lens_l),longest_gain_path=max(lens_g)))
        for n,lens in [('loss',lens_l),('gain',lens_g)]:
            dist.extend(dict(year=year,direction=n,programs=steps,applicants=count) for steps,count in sorted(lens.items()))
        print(summary[-1],flush=True)
    pd.DataFrame(summary).to_csv(a.directory/'school-order-witnesses.csv',index=False)
    pd.DataFrame(dist).to_csv(a.directory/'school-order-path-lengths.csv',index=False)


if __name__=='__main__':main()
