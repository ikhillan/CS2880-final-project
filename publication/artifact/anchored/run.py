"""Replay the 480 NYC development markets with an exposure rule and a full-DA oracle."""
import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


costs = load('costs', HERE.parent / 'nyc-costs/run.py')
dev = costs.dev
POLICIES = ('full', 'uniform', 'cheapest', 'balanced', 'anchored')
METRICS = ('renters', 'positive', 'matched', 'own_blocker', 'changed_assignment',
           'slots', 'replacements', 'display_rent_sum')


def balanced_mask(price_rows, order_rng, size):
    shown = np.zeros((size, size), bool)
    exposures = np.zeros(size, int)
    for _ in range(5):
        for b in order_rng.permutation(size):
            row = price_rows[b]
            available = row[~shown[b, row]]
            if len(available):
                j = available[np.argmin(exposures[available])]
                shown[b, j] = True
                exposures[j] += 1
    return shown


def anchor_mask(cheap, price_rows, best):
    shown = cheap.copy()
    for b, j in enumerate(best):
        if j >= 0 and not shown[b, j]:
            shown[b, price_rows[b][4]] = False
            shown[b, j] = True
    return shown


def market(donor, pool, seed, size):
    rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(9)]
    profiles, weights = donor
    incomes, minbeds, lower = profiles[rng[0].choice(len(profiles), size, replace=True, p=weights)].T
    records, _ = pool
    prices, beds = records[rng[1].choice(len(records), size, replace=False)].T
    budget = np.maximum(0, .4*incomes/12)
    feasible = (prices[None, :] <= budget[:, None]) & (beds[None, :] >= minbeds[:, None])
    d, ld = feasible.sum(axis=1), feasible.sum(axis=0)
    u = (-2*prices[None, :]/np.maximum(1, budget[:, None])
         + .5*np.minimum(2, beds[None, :]-minbeds[:, None])
         + .5*rng[2].standard_normal(size)[None, :] + .75*rng[3].standard_normal((size,size)))
    order = np.argsort(-u, axis=1, kind='stable')
    full_lists = dev.lists(order, feasible)
    rr = dev.row_ranks(full_lists, size)
    uniform_keys, ties = rng[4].random((size, size)), rng[5].random(size)
    masks = dev.shortlist_orders(uniform_keys, prices, feasible, ties)
    price_order = np.broadcast_to(np.lexsort((ties, prices)), feasible.shape)
    price_rows = [np.asarray(row, dtype=int) for row in dev.lists(price_order, feasible)]
    cheap = masks['cheapest']
    masks['balanced'] = balanced_mask(price_rows, rng[8], size)
    masks['full'] = feasible
    lotteries = (np.broadcast_to(rng[6].permutation(size)[:, None], feasible.shape),
                 np.argsort(rng[7].random(feasible.shape), axis=0, kind='stable'))
    lower = lower.astype(bool)
    people = np.arange(size)
    for priority, lottery in zip(dev.PRIORITIES, lotteries):
        lr = dev.row_ranks(dev.lists(lottery.T, feasible.T), size).T
        best, best_holders = dev.deferred(full_lists, lr)
        best_ranks = dev.assigned_ranks(best, rr, d)
        masks['anchored'] = anchor_mask(cheap, price_rows, best)
        matched = best >= 0
        assert masks['anchored'][people[matched], best[matched]].all()
        for name in POLICIES:
            mask = masks[name]
            q = mask.sum(axis=1)
            assert not (mask & ~feasible).any()
            if name != 'full':
                assert np.array_equal(q, np.minimum(d, 5))
            m, l = dev.deferred(dev.lists(order, mask), lr)
            blocks = dev.block_edges(m, l, rr, lr, feasible, d, ld)
            stable = not blocks.any()
            different = m != best
            if name == 'full':
                assert stable and not different.any()
            if name == 'anchored':
                assert (dev.assigned_ranks(m, rr, d) <= best_ranks).all()
                assert np.array_equal(m >= 0, best >= 0) and np.array_equal(l >= 0, best_holders >= 0)
                assert stable == (not different.any())
            assert not (blocks & mask).any()
            display_rent = (mask @ prices) / np.maximum(q, 1)
            replacements = (mask & ~cheap).sum(axis=1)
            if name == 'anchored':
                assert (replacements <= 1).all()
            for group, sel in [('lower', lower), ('higher', ~lower)]:
                counts = [sel.sum(), (sel & (d>0)).sum(), (sel & (m>=0)).sum(),
                          (sel & blocks.any(axis=1)).sum(), (sel & different).sum(),
                          q[sel].sum(), replacements[sel].sum(), display_rent[sel].sum()]
                yield (priority, name, group), counts
            yield (priority, name), Counter(markets=1, full_stable=int(stable),
                       differs_from_full=int(different.any()), blocking_pairs=int(blocks.sum()),
                       listings_exposed=int(mask.any(axis=0).sum()))


def run(data, module_path, out):
    baseline_path = HERE.parent/'nyc-matching-dev/results.json'
    baseline = json.loads(baseline_path.read_text())
    donors, pools = costs.inputs(data, module_path, baseline)
    B, T, K, H, P, G = 40, 3, 4, 2, len(POLICIES), 2
    counts = np.zeros((B,T,K,H,P,G,len(METRICS)))
    aggregate = defaultdict(Counter)
    for b in range(B):
        for t, month in enumerate(dev.MONTHS):
            for k, borough in enumerate(dev.BOROUGHS):
                for key, record in market(donors[borough], pools[month, borough], [20261112,b,t,k],64):
                    if len(key)==2:
                        aggregate[key].update(record)
                    else:
                        priority, name, group = key
                        counts[b,t,k,dev.PRIORITIES.index(priority),POLICIES.index(name),dev.GROUPS.index(group)] = record
    w = np.broadcast_to([baseline['borough_weights'][g]/3 for g in dev.BOROUGHS], (T,K))
    metric = {m:i for i,m in enumerate(METRICS)}

    def standard(c):
        n = c[...,metric['renters']]
        assert np.min(n)>0
        standardized = np.einsum('tb,tbhpgm->hpgm', w, c/n[...,None])
        r = {m:standardized[...,metric[m]] for m in
             ('positive','matched','own_blocker','changed_assignment','slots','replacements')}
        r['display_rent_given_positive'] = standardized[...,metric['display_rent_sum']]/r['positive']
        positive = c[...,metric['positive']]
        if np.any(positive == 0):
            raise ValueError('A supported cell lacks a positive-degree renter; its conditional mean is unavailable.')
        r['display_rent_cell_conditional'] = np.einsum(
            'tb,tbhpg->hpg', w, c[...,metric['display_rent_sum']]/positive)
        r['replacement_fraction_of_slots'] = r['replacements']/r['slots']
        return r

    pooled = counts.sum(axis=0)
    rates = standard(pooled)
    raw = pooled.sum(axis=(0,1))
    results = {}
    for h, prior in enumerate(dev.PRIORITIES):
        results[prior] = {}
        for p, name in enumerate(POLICIES):
            record = results[prior][name] = dict(aggregate[prior,name])
            record['groups'] = {}
            for g, group in enumerate(dev.GROUPS):
                group_row = record['groups'][group] = {
                    'counts':{m:int(raw[h,p,g,i]) for i,m in enumerate(METRICS) if m!='display_rent_sum'},
                    'standardized':{m:float(value[h,p,g]) for m,value in rates.items()}}
                if name in dev.POLICIES:
                    old = baseline['group_results'][prior][name][group]
                    for new_m, old_m in [('matched','matched'),('own_blocker','blockers')]:
                        assert group_row['counts'][new_m] == old[old_m]
                        assert abs(group_row['standardized'][new_m]-old['standardized_'+old_m])<1e-12
                    assert group_row['counts']['renters'] == old['renters']
                if name=='full':
                    old = baseline['group_results'][prior]['cheapest'][group]
                    assert group_row['counts']['matched']==old['full_matched']
                    assert abs(group_row['standardized']['matched']-old['standardized_full_matched'])<1e-12
    # Pair the exact same 40 seed blocks across all market conditions.
    tracked = ('matched','own_blocker','display_rent_given_positive','display_rent_cell_conditional')
    contrasts = {}
    comparisons = (('balanced','cheapest'), ('anchored','cheapest'), ('balanced','uniform'))
    first = [POLICIES.index(a) for a, _ in comparisons]
    second = [POLICIES.index(b) for _, b in comparisons]
    # Use the original bootstrap sequence of independent uniform block indices.
    rng = np.random.default_rng(19112026)
    C = counts.reshape(B, -1)
    weights = np.array([np.bincount(rng.integers(B,size=B),minlength=B) for _ in range(5000)])
    # Batch to bound memory.
    by_metric = {m:[] for m in tracked}
    for start in range(0, len(weights), 100):
        batch = (weights[start:start+100]@C).reshape(-1,*counts.shape[1:])
        for sample in batch:
            r=standard(sample)
            for m in tracked:
                by_metric[m].append(r[m][:,first]-r[m][:,second])
    for h, prior in enumerate(dev.PRIORITIES):
        contrasts[prior]={}
        for j, (a,b) in enumerate(comparisons):
            label=a+'_minus_'+b
            contrasts[prior][label]={}
            for g, group in enumerate(dev.GROUPS):
                contrasts[prior][label][group]={
                    m:{'estimate':float(rates[m][h,first[j],g]-rates[m][h,second[j],g]),
                       'mc_percentile_95':np.quantile(np.asarray(by_metric[m])[:,h,j,g],[.025,.975]).tolist()}
                    for m in tracked}
    report=dict(markets=480,blocks=B,borough_weights=baseline['borough_weights'],months=dev.MONTHS,
                market_seed_root=20261112, balanced_permutations='seed child 8, independent of original market shocks; one new permutation per round',
                bootstrap_reps=5000,bootstrap_seed=19112026,raw_hashes=baseline['raw_hashes'],
                baseline_results_sha256=dev.sha(baseline_path), rates=results, paired_contrasts=contrasts)
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--catalogue-module',required=True,type=Path)
    p.add_argument('--out',type=Path,default=HERE/'results.json')
    a=p.parse_args()
    run(a.data,a.catalogue_module,a.out)
