"""Minimal shortlist edits for maximum feasible matching size on the stage-one development sample."""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from experiments.config import expand_labeled, load_config
from hiddenoptions.data import borough_weights
from hiddenoptions.exposure import transparent_exposure, uniform_exposure
from hiddenoptions.markets import Catalog, InsufficientSupply, build_market, maximum_matching_size
from hiddenoptions.matching import blocking_mask, deferred_acceptance

POLICIES = ('transparent', 'capacity_repair', 'random_replacement', 'uniform', 'full')


def repair(feasible, shown, score):
    """Return a quota-preserving minimum-edit graph, and the edits (renter, old, new)."""
    n, m = feasible.shape
    if np.any(shown & ~feasible) or np.any(feasible.any(1) & ~shown.any(1)):
        raise ValueError('Each nonempty feasible row needs at least one displayed option, all feasible.')
    # One unmatched dummy per renter. One less dummy outranks any difference in
    # edits and ranks. One fewer hidden witness edge outranks any difference in ranks.
    rank = np.argsort(np.argsort(-score, axis=1, kind='stable'), axis=1, kind='stable')
    edit_cost = n * m + 1
    dummy_cost = (n + 1) * edit_cost
    costs = np.full((n, m + n), float(dummy_cost))
    costs[:, :m] = np.where(feasible, (~shown) * edit_cost + rank, np.inf)
    renters, partners = linear_sum_assignment(costs)
    output, changes = shown.copy(), []
    for b, s in zip(renters, partners):
        if s < m and not shown[b, s]:
            old = int(np.argmax(np.where(shown[b], rank[b], -1)))
            output[b, old] = False
            output[b, s] = True
            changes.append((int(b), old, int(s)))
    return output, changes


def random_replacement(shown, feasible, changes, seed):
    """Same initial graph, same renters, same removed slots, uniform hidden feasible alternative."""
    rng = np.random.default_rng(seed)
    out = shown.copy()
    for b, old, _ in changes:
        out[b, old] = False
        out[b, rng.choice(np.flatnonzero(feasible[b] & ~shown[b]))] = True
    return out


def observe(market, graph, cap, baseline, price):
    inst = market.inst
    partners, inverse = deferred_acceptance(inst, graph)
    blocked = blocking_mask(inst, partners, inverse)
    assert not (blocked & graph).any()
    assert not np.any(graph & ~inst.E)
    if graph is not inst.E:
        assert np.array_equal(graph.sum(axis=1), baseline.sum(axis=1))
    groups = market.renters.lower_income.to_numpy(bool)
    mean_price = (graph * price).sum(1) / np.maximum(1, graph.sum(1))
    row = dict(capacity=cap, da_matches=int((partners >= 0).sum()), full_stable=int(not blocked.any()),
               displayed=int(graph.sum()), replaced=int((graph & ~baseline).sum()))
    for name, mask in [('lower', groups), ('higher', ~groups)]:
        positive = mask & graph.any(1)
        row.update({name+'_n':int(mask.sum()), name+'_matched':int((mask & (partners >= 0)).sum()),
                    name+'_blocked':int((mask & blocked.any(1)).sum()), name+'_positive':int(positive.sum()),
                    name+'_mean_price_sum':float(mean_price[positive].sum())})
    return row


def collect():
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/stage1_pilot.json')
    catalog, rows, missing = Catalog(), [], []
    for spec, label in expand_labeled(cfg):
        if spec.donors != 'calibration' or spec.month not in ('2026-06','2026-07','2026-08'):
            raise ValueError('This analysis is defined only on the development calibration markets.')
        try:
            market = build_market(spec, catalog)
        except InsufficientSupply as e:
            missing.append(dict(borough=spec.borough, month=spec.month, variant=label, seed=spec.seed, reason=str(e)))
            continue
        inst, prices = market.inst, market.listings.rent.to_numpy(float)
        maximum = maximum_matching_size(inst.E)
        for budget, kw in [('k5', {'k':5}), ('q40', {'q':.4})]:
            base = transparent_exposure(market, **kw)
            edited, changes = repair(inst.E, base, market.u_observable)
            salt = zlib.crc32((spec.market_id + '|' + budget + '|capacity-placebo').encode())
            placebo = random_replacement(base, inst.E, changes, [spec.seed, salt])
            graphs = [base, edited, placebo, uniform_exposure(market, **kw), inst.E]
            for policy, graph in zip(POLICIES, graphs):
                cap = maximum_matching_size(graph)
                if policy == 'capacity_repair':
                    assert cap == maximum and int((graph & ~base).sum()) == len(changes)
                if policy == 'random_replacement':
                    assert int((graph & ~base).sum()) == len(changes)
                outcome = observe(market, graph, cap, base, prices)
                rows.append(dict(market_id=spec.market_id, variant=label, budget=budget, policy=policy,
                                 month=spec.month, borough=spec.borough, seed=spec.seed, n=spec.n_renters, **outcome))
    return pd.DataFrame(rows), missing, borough_weights(catalog.renters)


def summarize(frame, borough_shares, repeats):
    """Pool within a cell before computing group rates; paired bootstrap reuses all cells and policies."""
    fields = ['capacity','da_matches','full_stable','displayed','replaced','n']
    for g in ['lower','higher']:
        fields += [g+s for s in ['_n','_matched','_blocked','_positive','_mean_price_sum']]
    cells = sorted(set(zip(frame.month, frame.borough)))
    seeds = sorted(frame.seed.unique())
    index = pd.MultiIndex.from_tuples([(s,m,b,p) for s in seeds for m,b in cells for p in POLICIES],
                                      names=['seed','month','borough','policy'])
    aligned = frame.set_index(index.names).reindex(index)[fields]
    if aligned.isna().any().any():
        raise ValueError('Missing paired seed/cell/policy observation.')
    cube = aligned.to_numpy().reshape(len(seeds), len(cells), len(POLICIES), -1)
    weights = np.array([borough_shares[b] for _,b in cells]); weights /= weights.sum()
    stats = {}
    for name in ['capacity','da_matches','full_stable','displayed','replaced']:
        stats[name] = (name, None)
    stats['percent_slots_changed'] = ('replaced','displayed')
    for g in ['lower','higher']:
        for key in ['matched','blocked']:
            stats[g+'_'+key+'_pct'] = (g+'_'+key,g+'_n')
        stats[g+'_displayed_dollars'] = (g+'_mean_price_sum',g+'_positive')
    rng = np.random.default_rng(20261113)
    # Point estimate is first, then clusters resampled with replacement.
    freq = np.vstack([np.ones(len(seeds)),rng.multinomial(len(seeds), [1/len(seeds)]*len(seeds),size=repeats)])
    report = {}
    for key,(num,den) in stats.items():
        counts = np.einsum('rs,scp->rcp',freq,cube[...,fields.index(num)])
        if den is None:
            percell = counts / len(seeds)
        else:
            denom = np.einsum('rs,scp->rcp',freq,cube[...,fields.index(den)])
            with np.errstate(divide='ignore', invalid='ignore'):
                percell = counts/denom
            if not key.endswith('dollars'):
                percell *= 100
        out = np.einsum('c,rcp->rp', weights, percell)
        entry = {'rates':dict(zip(POLICIES,out[0].tolist())), 'repair_minus':{}}
        for base in ['transparent','random_replacement','uniform']:
            delta = out[:,1]-out[:,POLICIES.index(base)]
            ci = np.quantile(delta[1:],[.025,.975]).tolist() if np.isfinite(delta[1:]).all() else None
            entry['repair_minus'][base] = {'estimate':float(delta[0]),'paired_mc95':ci}
        report[key] = entry
    # A full graph has no shortlist quota; slot-edit metrics do not apply to it.
    for key in ['percent_slots_changed','replaced']:
        report[key]['rates'].pop('full')
    report['market_count'] = int(frame.market_id.nunique())
    baseline = frame[frame.policy == 'transparent']
    repaired = frame[frame.policy == 'capacity_repair']
    report['already_full_capacity'] = int((baseline.capacity.to_numpy()==repaired.capacity.to_numpy()).sum())
    report['full_stable_counts'] = frame.groupby('policy').full_stable.sum().astype(int).to_dict()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, missed, borough_shares = collect()
    args.output.mkdir(parents=True,exist_ok=True)
    rows.to_csv(args.output/'capacity_markets.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    sums = {v+'_'+k:summarize(f,borough_shares,5000) for (v,k),f in rows.groupby(['variant','budget'])}
    report = {'design':'600 attempted stage-one development markets, 20 paired blocks per design; supported borough ACS weights and equal months',
              'unavailable':len(missed),'available':int(rows.market_id.nunique()),'borough_shares':borough_shares,
              'panels':sums}
    (args.output/'capacity_results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
