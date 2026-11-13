"""Exact fixed-volume rent-burden support on nested catalogues of rental arrivals."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


SIZES = (32, 64, 128, 256, 'whole')
BOROUGHS = ('Manhattan', 'Brooklyn', 'Queens', 'Bronx')
MONTHS = ('2026-06', '2026-07', '2026-08')
RNG_ROOT = 20261113


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Edges:
    borough: str
    month: str
    seed: int
    pool_size: int
    renters: np.ndarray
    slots: np.ndarray
    prices: np.ndarray
    burden: np.ndarray
    initial_matched: np.ndarray
    unit_ids: np.ndarray

    def select(self, size, price_floor):
        width = self.pool_size if size == 'whole' else size
        keep = (self.slots < width) & (self.prices >= price_floor)
        return Edges(self.borough, self.month, self.seed, width, self.renters[keep],
                     self.slots[keep], self.prices[keep], self.burden[keep],
                     self.initial_matched, self.unit_ids)

    def edges(self, cutoff=np.inf, keep_initial=False, blocked_r=(), blocked_s=()):
        keep = self.burden <= cutoff
        if keep_initial:
            keep &= self.initial_matched[self.renters]
        if len(blocked_r):
            keep &= ~np.isin(self.renters,blocked_r)
        if len(blocked_s):
            keep &= ~np.isin(self.slots,blocked_s)
        return self.renters[keep], self.slots[keep]

    def matching(self, cutoff=np.inf, keep_initial=False, blocked_r=(), blocked_s=()):
        r,s = self.edges(cutoff,keep_initial,blocked_r,blocked_s)
        if not len(r):
            return []
        # Compress empty listing columns on the full monthly catalogue.
        present, compact = np.unique(s,return_inverse=True)
        a = csr_matrix((np.ones(len(r),np.int8),(r,compact)),shape=(len(self.initial_matched),len(present)))
        match = maximum_bipartite_matching(a,perm_type='column')
        return [(int(i),int(present[j])) for i,j in enumerate(match) if j>=0]


def vertex_cover(r, s, pairs, n_rows):
    """Construct and verify the dual certificate by alternating reachability."""
    present = set(zip(r.tolist(),s.tolist()))
    bmate = dict(pairs)
    smate = {j:i for i,j in pairs}
    assert set(pairs)<=present and len(bmate)==len(smate)==len(pairs)
    adj=[[] for _ in range(n_rows)]
    for i,j in present:
        adj[i].append(j)
    reachable_left = set(range(n_rows))-set(bmate)
    reachable_right = set()
    queue = list(reachable_left)
    while queue:
        i = queue.pop()
        for j in adj[i]:
            if bmate.get(i)==j or j in reachable_right:
                continue
            reachable_right.add(j)
            b=smate.get(j)
            if b is not None and b not in reachable_left:
                reachable_left.add(b)
                queue.append(b)
    left = set(range(n_rows))-reachable_left
    right = reachable_right
    assert all(i in left or j in right for i,j in present)
    assert len(left)+len(right)==len(pairs)
    return sorted(left), sorted(right)


def checked_max(m, threshold, original, max_size=None):
    local=m.matching(threshold,keep_initial=original)
    r,s=m.edges(threshold,keep_initial=original)
    left,right=vertex_cover(r,s,local,len(m.initial_matched))
    if max_size is not None:
        local=local[:max_size]
    return local, {'left':left,'right':right}


def scoped_certificate(data,scope,n,tau,witnesses,size,price_floor):
    """Find and certify the 55th order statistic with optional market/recipient constraints."""
    required=(n-1)//4+1
    original = scope=='original_recipients'
    bounded = scope!='pooled'
    def allowance(m):
        return int(m.initial_matched.sum()) if bounded else n
    def count(t):
        return sum(min(allowance(m),len(m.matching(t,keep_initial=original))) for m in data)
    if count(np.inf)<n:
        return dict(exact_minimum_q25=None,status='unable to supply 217 with these constraints')
    cuts=np.unique(np.concatenate([m.burden for m in data]))
    lo,hi=-1,len(cuts)-1
    while hi-lo>1:
        mid=(hi+lo)//2
        if count(cuts[mid])>=required:
            hi=mid
        else:
            lo=mid
    val=float(cuts[hi])
    below=np.nextafter(val,-np.inf)
    low_cap=0
    at_cap=0
    target_cap=0
    cheap, extra, lower_covers, target_covers = [],[],[],[]
    lookup=[]
    for k,m in enumerate(data):
        limit=allowance(m)
        target,tcover = checked_max(m,tau,original,limit)
        strict,cover = checked_max(m,below,original,limit)
        local,_ = checked_max(m,val,original,limit)
        lower_covers.append([k,cover['left'],cover['right']])
        target_covers.append([k,tcover['left'],tcover['right']])
        target_cap+=len(target)
        at_cap+=len(local)
        low_cap+=len(strict)
        spare = m.matching(keep_initial=original,blocked_r=[b for b,_ in local],blocked_s=[s for _,s in local])
        if bounded:
            assert len(local)+len(spare)>=limit, (scope,k,len(local),len(spare),limit)
        cheap.extend((k,b,s) for b,s in local)
        extra.extend((k,b,s) for b,s in spare[:limit-len(local)])
        lookup.append(dict(zip(zip(m.renters.tolist(),m.slots.tolist()),m.burden)))
    selected=(cheap+extra)[:n]
    assert len(selected)==n
    assert len({(k,i) for k,i,_ in selected})==len({(k,j) for k,_,j in selected})==n
    if bounded:
        by_market=Counter(k for k,_,_ in selected)
        assert all(by_market[k]==allowance(m) for k,m in enumerate(data))
    if original:
        assert all(data[k].initial_matched[b] for k,b,_ in selected)
    assert np.quantile([lookup[k][b,s] for k,b,s in selected],.25)==val
    assert low_cap<required<=at_cap
    witnesses.append({'size':size,'price_floor':price_floor,'scope':scope,'attaining_edges':selected,
                      'cover_strictly_below_optimum':lower_covers,'cover_at_acs_target':target_covers})
    return dict(exact_minimum_q25=val,capacity_strictly_below=low_cap,capacity_at_minimum_q25=at_cap,
                threshold_capacity_or_upper_bound=target_cap,attaining_placements=n)


def extract(repo, selected):
    sys.path.insert(0, str(repo.resolve()))
    from hiddenoptions.markets import Catalog, MarketSpec, build_market
    from hiddenoptions.matching import deferred_acceptance
    rows = []
    cat = Catalog()
    for bi, borough in enumerate(BOROUGHS):
        for mi, month in enumerate(MONTHS):
            for seed in range(20):
                spec = MarketSpec(borough,month,seed,64,64,donors='calibration',**selected)
                m = build_market(spec,cat)
                _, pool = cat.frame(spec)
                pool = pool.reset_index(drop=True)
                keys = pool.listing_uid.astype(str).to_numpy()
                assert len(np.unique(keys)) == len(keys)
                orig = m.listings.listing_uid.astype(str).to_numpy()
                index = {s:i for i,s in enumerate(keys)}
                root = np.random.SeedSequence([RNG_ROOT, bi, mi, seed])
                orig_rng, rem_rng = map(np.random.default_rng,root.spawn(2))
                orig_order = orig_rng.permutation(64)
                base = np.array([index[s] for s in orig[orig_order]])
                remaining = np.setdiff1d(np.arange(len(pool)),base,assume_unique=True)
                perm = np.r_[base,rem_rng.permutation(remaining)]
                pool = pool.iloc[perm]
                low = m.renters.lower_income.to_numpy(bool)
                income = m.renters.income.to_numpy(float)[low]
                need = m.renters.need_bedrooms.to_numpy(int)[low]
                cap = m.renters.cap.to_numpy(float)[low]
                rent = pool.rent.to_numpy(float)
                bedrooms = pool.bedrooms.to_numpy()
                feasible = ((rent <= cap[:,None]) & (bedrooms >= need[:,None]) & (income[:,None]>0))
                assert np.array_equal(feasible[:,:64], m.inst.E[low][:,orig_order])
                b,s = np.where(feasible)
                burdens = 12*rent[s]/income[b]
                match,_ = deferred_acceptance(m.inst,m.inst.E)
                rows.append(Edges(borough,month,seed,len(pool),b,s,rent[s],burdens,
                                  match[low]>=0,keys[perm]))
    return rows


def evaluate(base,size,price_floor,tau,n,witnesses):
    data = [m.select(size,price_floor) for m in base]
    def capacity(t):
        return sum(len(m.matching(t)) for m in data)
    stats = Counter()
    unique_target_units = set()
    by_borough = {}
    for m in data:
        pairs = m.matching(tau)
        low = m.burden <= tau
        key = m.borough
        c = by_borough.setdefault(key,Counter())
        c['threshold_capacity'] += len(pairs)
        c['markets'] += 1
        c['threshold_markets_positive'] += bool(pairs)
        stats['pooled_threshold_capacity'] += len(pairs)
        stats['per_market_match_volume_threshold_upper_bound'] += min(int(m.initial_matched.sum()),len(pairs))
        stats['original_matched_renters_threshold_upper_bound'] += len(m.matching(tau,keep_initial=True))
        for u in m.unit_ids[np.unique(m.slots[low])]:
            unique_target_units.add((m.month,m.borough,str(u)))
    all_capacity = capacity(np.inf)
    result = {'size':size,'price_floor':price_floor,**stats,'full_eligible_capacity':all_capacity,
              'distinct_month_borough_uid_with_any_threshold_edge':len(unique_target_units),
              'by_borough':by_borough}
    result.update(scoped_certificate(data,'pooled',n,tau,witnesses,size,price_floor))
    if size=='whole':
        for scope in ['same_market_counts','original_recipients']:
            result[scope]=scoped_certificate(data,scope,n,tau,witnesses,size,price_floor)
    assert result['threshold_capacity_or_upper_bound']==stats['pooled_threshold_capacity']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True,type=Path)
    parser.add_argument('--calibration',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    cal=json.loads(args.calibration.read_text())
    base=extract(args.repo,cal['selected'])
    n=sum(int(m.initial_matched.sum()) for m in base)
    assert n==cal['fits'][0]['matched_lower']==217
    tau=cal['targets']['lower'][0]
    inputs = {f'hiddenoptions/{k}.py':sha(args.repo/'hiddenoptions'/f'{k}.py') for k in ('markets','data','matching')}
    cleaned={}
    for month in MONTHS:
        raw = args.repo/'data/firstmover/raw'/f'{month}.manifest.json'
        clean = args.repo/'data/firstmover/clean'/f'firstmover_listings_{month}.csv.gz'
        # Hash uncompressed table contents rather than gzip timestamps.
        import gzip
        cleaned[month]=hashlib.sha256(gzip.decompress(clean.read_bytes())).hexdigest()
        inputs[f'{month}.manifest.json']=sha(raw)
    out={'root_seed':RNG_ROOT,'versions':{'numpy':np.__version__,'scipy':scipy.__version__},
         'source_sha256':inputs,'calibration_sha256':sha(args.calibration),'clean_csv_sha256':cleaned,
         'markets':len(base),'all_renter_instances':len(base)*64,
         'lower_income_renter_instances':sum(len(m.initial_matched) for m in base),
         'lower_income_placement_volume':n,'threshold_quantile':.25,'target_burden':tau,
         'threshold_required_count':(n-1)//4+1,
         'indices':'Rows are lower-income-only renter positions in the original draw; columns index a nested listing permutation. Markets traverse borough, month, then seed as in code.',
         'sizes':[]}
    witnesses=[]
    for floor in (0,1000):
        for size in SIZES:
            result=evaluate(base,size,floor,tau,n,witnesses)
            out['sizes'].append(result)
            print(floor,size,result['pooled_threshold_capacity'],result['full_eligible_capacity'],result['exact_minimum_q25'],flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    (args.out/'witnesses.json').write_text(json.dumps(witnesses,separators=(',',':'))+'\n')


if __name__=='__main__':
    main()
