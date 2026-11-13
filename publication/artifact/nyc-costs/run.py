"""Rerun a two-sided DA screen and bound the repair premium in the residual cases."""
import argparse
import csv
import importlib.util
import json
import platform
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dev = load_module('matching_dev', HERE.parent / 'nyc-matching-dev/run.py')


def normalize(name, listing_id):
    value = ' '.join(unicodedata.normalize('NFKC', name).casefold().split())
    if value in {'', 'na', 'n/a', 'unknown', 'none', 'null', 'owner', 'by owner'}:
        return 'listing:' + listing_id
    return 'source:' + value


def inputs(data, module_path, baseline):
    sys.path.insert(0, str(module_path.resolve().parent))
    scarcity = load_module('scarcity', module_path)
    for path in [data/'acs/recent_donors.csv', data/'acs/summary.json', module_path,
                 *(data/'firstmover'/f'{t}.csv' for t in dev.MONTHS)]:
        assert dev.sha(path) == baseline['raw_hashes'][path.name], path
    meta = json.loads((data/'acs/summary.json').read_text())
    raw_donors = defaultdict(list)
    with (data/'acs/recent_donors.csv').open() as f:
        for r in csv.DictReader(f):
            if r['donor_split'] == 'calibration' and r['borough'] in dev.BOROUGHS:
                raw_donors[r['borough']].append((float(r['income_july2026']),
                    max(0, (int(r['household_size'])-1)//2),
                    float(r['income_2024']) <= meta['median_income_2024'], float(r['weight'])))
    donors = {}
    for borough, rows in raw_donors.items():
        a = np.array(rows)
        donors[borough] = a[:, :3], a[:, 3]/a[:, 3].sum()
    pools = {}
    for month in dev.MONTHS:
        path = data/'firstmover'/f'{month}.csv'
        rows, _ = scarcity.load_catalogue(path)
        # An ID join must identify a unique group even if the source repeats an ID.
        groups = defaultdict(set)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for r in csv.DictReader(stream):
                groups[r['id']].add(normalize(r['source_group'], r['id']))
        for borough in dev.BOROUGHS:
            kept = [(p*meta['cpi_u']['2026-07']/meta['cpi_u'][month], b, j)
                    for bor, p, b, j in rows if bor == borough]
            assert all(len(groups[j]) == 1 for _, _, j in kept)
            pools[month, borough] = (np.array([(p,b) for p,b,_ in kept]),
                                     [next(iter(groups[j])) for _, _, j in kept])
    return donors, pools


class Market:
    def __init__(self, donors, pool, block, month, borough, size=64):
        rng = [np.random.default_rng(child) for child in
               np.random.SeedSequence([20261112, block, month, borough]).spawn(9)]
        profiles, weights = donors
        inc, bed_min, lower = profiles[rng[0].choice(len(profiles), size, replace=True, p=weights)].T
        rows, labels = pool
        indices = rng[1].choice(len(rows), size, replace=False)
        prices, beds = rows[indices].T
        sources = [labels[i] for i in indices]
        groups = sorted(set(sources))
        self.firm = np.array([size+groups.index(s) for s in sources])
        self.count = size+len(groups)
        budget = np.maximum(0, .4*inc/12)
        self.e = (prices[None,:] <= budget[:,None]) & (beds[None,:] >= bed_min[:,None])
        self.d, self.ld = self.e.sum(axis=1), self.e.sum(axis=0)
        u = (-2*prices[None,:]/np.maximum(budget[:,None],1) + .5*np.minimum(2,beds[None,:]-bed_min[:,None])
             + .5*rng[2].standard_normal(size)[None,:] + .75*rng[3].standard_normal((size,size)))
        self.order = np.argsort(-u, axis=1, kind='stable')
        full = dev.lists(self.order, self.e)
        self.rr = dev.row_ranks(full, size)
        self.v = dev.shortlist_orders(rng[4].random((size,size)),prices,self.e,rng[5].random(size))['uniform']
        receiver = np.argsort(rng[7].random((size,size)), axis=0, kind='stable')
        rel = dev.lists(receiver.T, self.e.T)
        self.lr = dev.row_ranks(rel,size).T
        self.best, self.best_l = dev.deferred(full, self.lr)
        self.worst_l, self.worst = dev.deferred(rel, self.rr.T)
        self.initial, init_l = dev.deferred(dev.lists(self.order,self.v), self.lr)
        self.r0 = self.rank(self.initial)
        self.br, self.wr = self.rank(self.best), self.rank(self.worst)
        self.target = lower.astype(bool) & (self.initial >= 0) & (self.br <= self.r0)
        self.cross = self.target & (self.r0 < self.wr)
        assert (self.br <= self.wr).all()
        assert np.array_equal(self.best >= 0, self.worst >= 0)
        self.id = dict(block=block, month=dev.MONTHS[month], borough=dev.BOROUGHS[borough])
        self.record = dict(renters=int(sum(lower)), choices_gt5=int(sum(lower.astype(bool) & (self.d>5))),
                           risk_candidates=int(sum(self.cross)), screen_pass=int(not self.cross.any()),
                           screen_positive_threshold=int(self.target.any()),
                           blockers=int(sum(lower.astype(bool) & dev.block_edges(self.initial,init_l,self.rr,self.lr,
                                                    self.e,self.d,self.ld).any(axis=1))))

    def rank(self, m):
        return dev.assigned_ranks(m, self.rr, self.d)

    def evaluate(self, actions):
        bits = np.zeros(self.count,bool)
        bits[list(actions)] = True
        visible = self.v | (self.e & (bits[:len(self.d),None] | bits[self.firm][None,:]))
        m, l = dev.deferred(dev.lists(self.order,visible),self.lr)
        blocks = dev.block_edges(m,l,self.rr,self.lr,self.e,self.d,self.ld)
        harms = self.target & (self.rank(m) > self.r0)
        return m, blocks, harms


def stable_assignments(market):
    """Enumerate only partners between both DA endpoints on both sides."""
    m = market
    assignment = m.best.copy()
    variable = np.flatnonzero(m.best != m.worst)
    available = set(m.best[variable])
    lower_l = dev.assigned_ranks(m.worst_l,m.lr.T,m.ld)
    upper_l = dev.assigned_ranks(m.best_l,m.lr.T,m.ld)
    options = {i: [j for j in sorted(available) if m.e[i,j] and m.br[i]<=m.rr[i,j]<=m.wr[i]
                   and lower_l[j]<=m.lr[i,j]<=upper_l[j]] for i in variable}
    sequence = sorted(variable,key=lambda b:len(options[b]))

    def visit(depth):
        if depth == len(sequence):
            inverse = np.full(len(assignment),-1,int)
            for i in np.flatnonzero(assignment >= 0):
                inverse[assignment[i]] = i
            if not dev.block_edges(assignment,inverse,m.rr,m.lr,m.e,m.d,m.ld).any():
                yield assignment.copy()
            return
        i = sequence[depth]
        for j in options[i]:
            if j in available:
                assignment[i] = j
                available.remove(j)
                yield from visit(depth+1)
                available.add(j)
    return list(visit(0))


def stars(m, assignment):
    hidden = defaultdict(list)
    for b, j in enumerate(assignment):
        if j>=0 and not m.v[b,j]:
            hidden[int(m.firm[j])].append(b)
    return dict(hidden)


def min_covers(groups):
    # In a witness every renter occurs at most once. Its hidden edges form
    # disjoint stars with a brokerage at each center.
    centers = set(groups)
    ties = [f for f, renters in groups.items() if len(renters)==1]
    for mask in range(1 << len(ties)):
        chosen = centers.copy()
        for k,f in enumerate(ties):
            if mask >> k & 1:
                chosen.remove(f)
                chosen.add(groups[f][0])
        yield chosen


def independently_verify(m, actions):
    """Synchronous proposals and direct comparison, separate from snapshot DA/checker."""
    shown = [[int(j) for j in row if m.e[i,j] and
              (m.v[i,j] or i in actions or m.firm[j] in actions)]
             for i,row in enumerate(m.order)]
    cursor = [0]*len(shown)
    held = {}
    while True:
        pool = defaultdict(list)
        held_renters = set(held.values())
        for i,choices in enumerate(shown):
            if i not in held_renters and cursor[i] < len(choices):
                j = choices[cursor[i]]
                cursor[i] += 1
                pool[j].append(i)
        if not pool:
            break
        for j,applicants in pool.items():
            if j in held:
                applicants.append(held[j])
            held[j] = min(applicants,key=lambda i:m.lr[i,j])
    assigned = {i:j for j,i in held.items()}
    blockers = []
    for i,j in zip(*np.where(m.e)):
        beats_r = i not in assigned or m.rr[i,j] < m.rr[i,assigned[i]]
        beats_l = j not in held or m.lr[i,j] < m.lr[held[j],j]
        if beats_r and beats_l:
            blockers.append((int(i),int(j)))
    failures = [int(i) for i in np.flatnonzero(m.target)
                if i not in assigned or m.rr[i,assigned[i]] > m.r0[i]]
    return np.array([assigned.get(i,-1) for i in range(len(shown))]), blockers, failures


def solve(m):
    stable = stable_assignments(m)
    assert any(np.array_equal(w,m.best) for w in stable)
    assert any(np.array_equal(w,m.worst) for w in stable)
    records = []
    for w in stable:
        groups = stars(m,w)
        records.append(dict(assignment=w.tolist(), protected=not (m.target & (m.rank(w)>m.r0)).any(),
                            star_bound=len(groups), singletons=sum(len(v)==1 for v in groups.values())))
    ordinary_floor = min(r['star_bound'] for r in records)
    protected_floor = min(r['star_bound'] for r in records if r['protected'])
    incumbent = set(range(len(m.d),m.count))
    _, blocks, harm = m.evaluate(incumbent)
    assert not blocks.any() and not harm.any()
    bucket = Counter()
    outcomes = defaultdict(set)
    shared_blocks = {}
    for level in sorted({r['star_bound'] for r in records}):
        if level>ordinary_floor+1:
            break
        for r,w in zip(records,stable):
            if r['star_bound'] != level:
                continue
            for plan in min_covers(stars(m,w)):
                if len(incumbent) <= level:
                    break
                response, blockers, harms = m.evaluate(plan)
                bucket[level,'checks'] += 1
                if blockers.any():
                    bucket[level,'unstable'] += 1
                    outcomes[level].add(tuple(response.tolist()))
                    edges = set(map(tuple,np.argwhere(blockers).tolist()))
                    shared_blocks[level] = shared_blocks.get(level,edges) & edges
                    check_m, check_blocks, _ = independently_verify(m,plan)
                    assert np.array_equal(check_m,response) and set(check_blocks)==edges
                elif harms.any():
                    bucket[level,'unprotected'] += 1
                else:
                    incumbent = plan
                    bucket[level,'protected'] += 1
    upper = len(incumbent)
    ordinary = ordinary_floor
    if upper>ordinary and sum(1 << r['singletons'] for r in records if r['star_bound']==ordinary) == bucket[ordinary,'unstable']:
        ordinary += 1
    protected = max(ordinary,protected_floor)
    da,blockers,harms = independently_verify(m,incumbent)
    da2,block2,harm2 = m.evaluate(incumbent)
    assert np.array_equal(da,da2) and not (blockers or harms or block2.any() or harm2.any())
    assert ordinary <= protected <= upper
    return {**m.id, 'size':len(m.d),'listing_groups':m.count-len(m.d),
            'eligible_edges':int(m.e.sum()),'variable_renters':int(sum(m.best!=m.worst)),
            'crossing_renters':np.flatnonzero(m.cross).tolist(),
            'thresholds':{int(i):int(m.r0[i]) for i in np.flatnonzero(m.target)},
            'stable_assignments':records,'base_support_min':ordinary_floor,'protected_support_min':protected_floor,
            'cover_checks':{level:{s:bucket[level,s] for s in ['checks','unstable','unprotected','protected']}
                            for level in sorted(set(l for l,_ in bucket))},
            'failed_response_diagnostics':{level:dict(distinct_da_outcomes=len(outcomes[level]),
                  blocking_pairs_common_to_all=sorted(edges)) for level,edges in shared_blocks.items()},
            'base_bounds':[ordinary,upper],'protected_bounds':[protected,upper],
            'premium_bounds':[max(0,protected-upper),upper-ordinary],'incumbent_actions':sorted(incumbent),
            'verified_renter_assignment':da.tolist(),'verified_full_blockers':len(blockers),
            'verified_threshold_failures':len(harms)}


def main(data,module,output):
    baseline = json.loads((HERE.parent/'nyc-matching-dev/results.json').read_text())
    donors,pools = inputs(data,module,baseline)
    counts = Counter()
    rows = []
    for t,month in enumerate(dev.MONTHS):
        for k,borough in enumerate(dev.BOROUGHS):
            for block in range(baseline['independent_seed_blocks']):
                m = Market(donors[borough], pools[month,borough],block,t,k)
                counts.update(m.record)
                if m.cross.any():
                    rows.append(solve(m))
    expected = baseline['group_results']['heterogeneous']['uniform']['lower']
    assert all(v==expected[k] for k,v in counts.items())
    classified = counts['screen_pass'] + len(rows)
    assert classified==baseline['markets']
    zero = sum(r['premium_bounds']==[0,0] for r in rows)
    result = dict(design='Development-only 64 x 64 markets; heterogeneous priorities, uniform shortlist, lower-income thresholds',
        ordinary_and_protected_repair='renter-proposing DA rerun on OR-exposed graph, full graph blocking test',
        audit='unit-cost renter singletons and normalized listing source groups; generic owners and missing names are listing singletons',
        raw_hashes=baseline['raw_hashes'],python=platform.python_version(),numpy=np.__version__,
        screen=counts, n_markets=classified, screen_pass=counts['screen_pass'],screen_fail=len(rows),
        exact_zero_among_failures=zero, exact_zero_total=counts['screen_pass']+zero,
        unresolved=sum(r['premium_bounds'][0]!=r['premium_bounds'][1] for r in rows),
        strict_positive_among_failures=sum(r['premium_bounds'][0]>0 for r in rows),
        timeouts=0, instances=rows)
    output.write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        print(r['month'],r['borough'],r['block'],r['base_bounds'],r['protected_bounds'],r['cover_checks'])
    print('exact zero',result['exact_zero_total'], '/', classified)


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--catalogue-module',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    args = ap.parse_args()
    main(args.data,args.catalogue_module,args.out)
