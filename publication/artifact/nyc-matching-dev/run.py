"""Paired development-only catalogue matching diagnostic for NYC, June-August 2026."""
import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import platform

import numpy as np

MONTHS = ('2026-06','2026-07','2026-08')
BOROUGHS = ('Bronx','Brooklyn','Manhattan','Queens')
PRIORITIES = ('common','heterogeneous')
POLICIES = ('uniform','cheapest')
GROUPS = ('lower','higher')


def sha(path):
    with path.open('rb') as fh:
        return hashlib.file_digest(fh, 'sha256').hexdigest()


def deferred(row_lists, column_ranks):
    n, m = column_ranks.shape
    proposer, holder = np.full(n,-1,int), np.full(m,-1,int)
    next_index = np.zeros(n,int)
    queue = deque(range(n))
    while queue:
        b = queue.popleft()
        if next_index[b] == len(row_lists[b]):
            continue
        j = row_lists[b][next_index[b]]
        next_index[b] += 1
        old = holder[j]
        if old == -1 or column_ranks[b,j] < column_ranks[old,j]:
            holder[j],proposer[b] = b,j
            if old != -1:
                proposer[old] = -1
                queue.append(old)
        else:
            queue.append(b)
    return proposer,holder


def lists(order, allowed):
    return [line[allowed[b,line]].tolist() for b,line in enumerate(order)]


def row_ranks(row_lists, n_columns):
    result = np.full((len(row_lists),n_columns),n_columns+1,int)
    for b,row in enumerate(row_lists):
        result[b,row] = np.arange(1,len(row)+1)
    return result


def assigned_ranks(matches, ranks, degree):
    a = np.arange(len(matches))
    return np.where(matches >= 0, ranks[a,matches], degree+1)


def block_edges(renter, listing, rr, lr, allowed, degree, listing_degrees):
    b_current = assigned_ranks(renter,rr,degree)
    j_current = assigned_ranks(listing,lr.T,listing_degrees)
    return allowed & (rr<b_current[:,None]) & (lr<j_current[None,:])


def shortlist_orders(uniform_keys, prices, eligible, tie_keys):
    n=len(prices)
    uniform=np.argsort(uniform_keys,axis=1,kind='stable')
    price=np.broadcast_to(np.lexsort((tie_keys,prices)),(n,n))
    chosen={}
    for policy,order in [('uniform',uniform),('cheapest',price)]:
        mask=np.zeros_like(eligible)
        for b,row in enumerate(lists(order,eligible)):
            mask[b,row[:5]]=True
        assert np.array_equal(mask.sum(axis=1),np.minimum(5,eligible.sum(axis=1)))
        chosen[policy]=mask
    for mask in chosen.values():
        assert not (mask & ~eligible).any()
    return chosen


def one_market(donors, pool, block, month_index, borough_index, size):
    seed=np.random.SeedSequence([20261112,block,month_index,borough_index])
    rngs=[np.random.default_rng(child) for child in seed.spawn(9)]
    profiles, weights=donors
    chosen=rngs[0].choice(len(profiles),size=size,replace=True,p=weights)
    incomes,minbeds,lower=profiles[chosen].T
    prices,beds=pool[rngs[1].choice(len(pool),size=size,replace=False)].T
    budget=np.maximum(0,.4*incomes/12)
    eligible=(prices[None,:] <= budget[:,None]) & (beds[None,:] >= minbeds[:,None])
    degrees=eligible.sum(axis=1)
    listing_degrees=eligible.sum(axis=0)
    # Renter taste and quality are independent of policy and both priority specifications.
    u=(-2*prices[None,:]/np.maximum(budget[:,None],1)
       +.5*np.minimum(2,beds[None,:]-minbeds[:,None])
       +.5*rngs[2].standard_normal(size)[None,:]
       +.75*rngs[3].standard_normal((size,size)))
    order=np.argsort(-u,axis=1,kind='stable')
    all_lists=lists(order,eligible)
    rr=row_ranks(all_lists,size)
    visible=shortlist_orders(rngs[4].random((size,size)),prices,eligible,rngs[5].random(size))
    common=np.broadcast_to(rngs[6].permutation(size)[:,None],(size,size))
    heterogeneous=np.argsort(rngs[7].random((size,size)),axis=0,kind='stable')
    reco={p: lists(order,visible[p]) for p in POLICIES}
    for priority,receiver_order in [('common',common),('heterogeneous',heterogeneous)]:
        receiver_lists=lists(receiver_order.T,eligible.T)
        lr=row_ranks(receiver_lists,size).T
        best,best_listing=deferred(all_lists,lr)
        worst_listing,worst=deferred(receiver_lists,rr.T)
        assert not block_edges(best,best_listing,rr,lr,eligible,degrees,listing_degrees).any()
        assert not block_edges(worst,worst_listing,rr,lr,eligible,degrees,listing_degrees).any()
        best_rank,worst_rank=assigned_ranks(best,rr,degrees),assigned_ranks(worst,rr,degrees)
        assert (best_rank<=worst_rank).all() and ((best>=0)==(worst>=0)).all()
        if priority=='common':
            assert np.array_equal(best,worst)
        for p in POLICIES:
            init,init_listing=deferred(reco[p],lr)
            r0=assigned_ranks(init,rr,degrees)
            blockers=block_edges(init,init_listing,rr,lr,eligible,degrees,listing_degrees)
            assert not (blockers & visible[p]).any()
            has_block=blockers.any(axis=1)
            assert not (has_block & (degrees<=5)).any()
            if not has_block.any():
                assert (best_rank<=r0).all() and (r0<=worst_rank).all()
            matched=init>=0
            score=np.where(matched,1-(r0-1)/np.maximum(degrees,1),0)
            for group,member in [('lower',lower.astype(bool)),('higher',~lower.astype(bool))]:
                target=member & matched & (best_rank<=r0)
                slack=target & (best_rank<worst_rank)
                crossing=target & (r0<worst_rank)
                record=Counter(markets=1,renters=int(sum(member)),
                    choices_gt5=int(sum(member & (degrees>5))),
                    zero_degree=int(sum(member & (degrees==0))),
                    blockers=int(sum(member & has_block)),matched=int(sum(member & matched)),
                    full_matched=int(sum(member & (best>=0))),rank_score_sum=float(score[member].sum()),
                    initial_top3=int(sum(member & matched & (r0<=3))),
                    unavoidable_loss=int(sum(member & (best_rank>r0))),
                    nonempty_group=int(member.any()),screen_pass=int(not crossing.any()),
                    screen_pass_vacuous=int(not target.any()),screen_pass_positive_threshold=int(target.any() and not crossing.any()),
                    screen_pass_variable_interval=int(slack.any() and not crossing.any()),
                    screen_positive_threshold=int(target.any()),screen_variable_interval=int(slack.any()),
                    risk_candidates=int(crossing.sum()))
                yield (priority,p,group),record


def main(data, module_path, out, blocks):
    sys.path.insert(0,str(module_path.resolve().parent))
    spec=importlib.util.spec_from_file_location('scarcity',module_path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta=json.loads((data/'acs/summary.json').read_text())
    donors=defaultdict(list)
    with (data/'acs/recent_donors.csv').open() as stream:
        for r in csv.DictReader(stream):
            bor=r['borough']
            if r['donor_split']=='calibration' and bor in BOROUGHS:
                donors[bor].append([float(r['income_july2026']),max(0,(int(r['household_size'])-1)//2),
                                    float(r['income_2024'])<=meta['median_income_2024'],float(r['weight'])])
    prepared={}
    for bor,rows in donors.items():
        a=np.array(rows);prepared[bor]=(a[:,:3],a[:,3]/a[:,3].sum())
    years=meta['cpi_u']
    catalogs={}
    for month in MONTHS:
        rows,_=module.load_catalogue(data/'firstmover'/f'{month}.csv')
        for bor in BOROUGHS:
            catalogs[month,bor]=np.array([(rent*years['2026-07']/years[month],beds)
                                         for b,rent,beds,_ in rows if b==bor],dtype=float)
    counters=defaultdict(Counter)
    for block in range(blocks):
        for t,month in enumerate(MONTHS):
            for k,bor in enumerate(BOROUGHS):
                for key,record in one_market(prepared[bor],catalogs[month,bor],block,t,k,64):
                    counters[(block,t,k,*key)].update(record)
    w=np.array([meta['by_borough'][b]['recent_weight'] for b in BOROUGHS],float)
    w/=w.sum()
    weights=np.broadcast_to(w[None,:]/3,(3,4))
    metrics=['renters','zero_degree','choices_gt5','blockers','matched','full_matched','rank_score_sum','initial_top3','unavoidable_loss',
             'screen_pass','screen_pass_vacuous','screen_pass_positive_threshold','screen_pass_variable_interval','screen_positive_threshold','screen_variable_interval','risk_candidates','markets','nonempty_group']
    data_array=np.array([[[[[[[counters[(block,t,k,prior,p,g)][m] for m in metrics] for g in GROUPS] for p in POLICIES] for prior in PRIORITIES] for k in range(4)] for t in range(3)] for block in range(blocks)])
    at={m:i for i,m in enumerate(metrics)}
    def summaries(counts):
        denom=counts[...,at['renters']]
        rates=np.einsum('tb,tbhpgm->hpgm',weights,counts/np.maximum(denom[...,None],1))
        assert np.all(denom>0)
        total=counts.sum(axis=(0,1))
        per={}
        contrasts={}
        for h,prior in enumerate(PRIORITIES):
            per[prior]={}
            for i,p in enumerate(POLICIES):
                per[prior][p]={}
                for k,g in enumerate(GROUPS):
                    row={m:int(total[h,i,k,at[m]]) for m in metrics if m!='rank_score_sum'}
                    for m in ('zero_degree','choices_gt5','matched','full_matched','unavoidable_loss','initial_top3','blockers','rank_score_sum'):
                        row['standardized_'+m]=float(rates[h,i,k,at[m]])
                    row['standardized_blockers_given_choices_gt5']=None if row['standardized_choices_gt5']==0 else row['standardized_blockers']/row['standardized_choices_gt5']
                    per[prior][p][g]=row
            d=rates[h,1,:,at['blockers']]-rates[h,0,:,at['blockers']]
            contrasts[prior]=dict(cheapest_minus_uniform_lower=float(d[0]),cheapest_minus_uniform_higher=float(d[1]),difference_lower_minus_higher=float(d[0]-d[1]))
        return per,contrasts
    per,difference=summaries(data_array.sum(axis=0))
    # Paired cluster bootstrap recomputes cell ratio estimands and all four rates.
    rng=np.random.default_rng(19112026)
    sampled=np.empty((5000,len(PRIORITIES),3))
    nums=data_array[...,at['blockers']];den=data_array[...,at['renters']]
    for rep in range(len(sampled)):
        ix=rng.integers(blocks,size=blocks)
        den_rep=den[ix].sum(axis=0);assert den_rep.min()>0
        r=np.einsum('tb,tbhpg->hpg',weights,nums[ix].sum(axis=0)/den_rep)
        d=r[:,1,:]-r[:,0,:]
        sampled[rep]=np.column_stack((d,d[:,0]-d[:,1]))
    for h,prior in enumerate(PRIORITIES):
        for j,m in enumerate(list(difference[prior])):
            difference[prior][m+'_ci95']=np.quantile(sampled[:,h,j],[.025,.975]).tolist()
    report=dict(design='Post-development exploratory NYC catalogue simulations, no AI model calls',size=64,
                independent_seed_blocks=blocks,months=MONTHS,boroughs=BOROUGHS,
                borough_weights=dict(zip(BOROUGHS,w)),markets=12*blocks,seed_root=20261112,
                bootstrap_reps=5000,bootstrap_seed=19112026,
                standardization='Average month and borough-specific renter rates using equal months and supported-borough ACS recent-mover household shares. Screen counts are raw market counts.',
                raw_hashes={f.name:sha(f) for f in [data/'acs/recent_donors.csv',data/'acs/summary.json',*(data/'firstmover'/f'{t}.csv' for t in MONTHS),module_path]},
                python=platform.python_version(),numpy=np.__version__,group_results=per,paired_contrasts=difference)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(difference,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--catalogue-module',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--blocks',type=int,default=40)
    a=ap.parse_args()
    main(a.data,a.catalogue_module,a.out,a.blocks)
