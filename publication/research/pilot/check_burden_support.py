"""How far can the ACS burden targets be reached using these simulated listings?

Matched-budget tests here allow arbitrary reassignment and priority violations.
For group g and threshold t, the maximum matching on feasible edges with burden
<=t is an upper bound on the number of placements at or below t under any
preference model. Pool market capacities exactly as the original calibrator
pools achieved burdens; each listing instance has capacity one in its market.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from hiddenoptions.data import BOROUGHS, DEV_MONTHS
    from hiddenoptions.markets import Catalog, InsufficientSupply, MarketSpec, build_market, maximum_matching_size
    from hiddenoptions.matching import deferred_acceptance

    report = json.loads(args.calibration.read_text())
    cat = Catalog()
    groups = ['lower','higher']
    qs = [.25,.5,.75]
    targets = np.array([report['targets'][g] for g in groups])
    capacities = np.zeros((2,3,3), int)
    achieved = np.zeros((2,3), int)
    n = np.zeros(2,int)
    low_matrices = []
    markets = 0
    for bor in BOROUGHS:
        for month in DEV_MONTHS:
            for seed in range(20):
                spec = MarketSpec(bor,month,seed,64,64,donors='calibration',**report['selected'])
                try:
                    m = build_market(spec,cat)
                except InsufficientSupply:
                    continue
                mb,_ = deferred_acceptance(m.inst,m.inst.E)
                placements = mb>=0
                inc = m.renters.income.to_numpy(float)
                low = m.renters.lower_income.to_numpy(bool)
                prices = m.listings.rent.to_numpy(float)
                bur = np.full(m.inst.E.shape,np.inf)
                np.divide(12*prices[None,:], inc[:,None], out=bur, where=(inc[:,None]>0))
                for g,mask in enumerate((low,~low)):
                    assigned = np.flatnonzero(placements & mask)
                    if g==0:
                        low_matrices.append(np.where(m.inst.E[mask],bur[mask],np.inf))
                    n[g] += len(assigned)
                    for j,target in enumerate(targets[g]):
                        threshold_edges = m.inst.E & (bur<=target)
                        count = int(np.sum(bur[assigned,mb[assigned]]<=target))
                        local_fixed = maximum_matching_size(threshold_edges[assigned])
                        local_all = maximum_matching_size(threshold_edges[mask])
                        # Use an independently implemented Hopcroft–Karp routine.
                        other = maximum_bipartite_matching(csr_matrix(threshold_edges[mask]),perm_type='column')
                        assert local_all == np.count_nonzero(other>=0)
                        # Hold only the group match count per market fixed, allowing new identities.
                        cap_if_count_fixed = min(local_all,len(assigned))
                        assert count<=local_fixed<=cap_if_count_fixed<=local_all
                        achieved[g,j] += count
                        capacities[g,j] += (local_fixed,cap_if_count_fixed,local_all)
                markets += 1
    assert markets==report['markets_used']
    result = {}
    for i,g in enumerate(groups):
        assert int(n[i]) == report['fits'][0]['matched_'+g]
        result[g] = {'matched_at_selected_preferences':int(n[i]),'thresholds':[]}
        for j,q in enumerate(qs):
            # numpy's linearly interpolated pth sample quantile can be <= t only
            # if the sample has at least floor((N-1)p)+1 observations <=t.
            required = int(np.floor((n[i]-1)*q))+1
            fixed,per_market,pooled = map(int,capacities[i,j])
            result[g]['thresholds'].append({
                'quantile':q,'survey_burden':float(targets[i,j]),'necessary_for_sample_quantile':required,
                'actual_placements_at_or_below':int(achieved[i,j]),
                'upper_bound_same_renters':fixed,'upper_bound_same_per_market_group_counts':per_market,
                'upper_bound_any_renters_or_priorities_pooled':pooled,
                'impossible_at_observed_total_group_matches':bool(pooled<required)})
    # Invert the aggregated capacity curve on its finite breakpoints.
    thresholds = np.unique(np.concatenate([a[np.isfinite(a)] for a in low_matrices]))
    def pooled_capacity(t):
        return sum(maximum_matching_size(a<=t) for a in low_matrices)
    for row in result['lower']['thresholds']:
        floor = row['necessary_for_sample_quantile']
        lo, hi = -1, len(thresholds)-1
        assert pooled_capacity(thresholds[hi]) >= floor
        while hi-lo>1:
            mid = (lo+hi)//2
            if pooled_capacity(thresholds[mid])>=floor:
                hi = mid
            else:
                lo = mid
        row['lower_bound_on_order_stat_at_observed_total'] = float(thresholds[hi])
        row['capacity_at_lower_bound'] = pooled_capacity(thresholds[hi])
        row['capacity_just_below_lower_bound'] = pooled_capacity(np.nextafter(thresholds[hi], -np.inf))
        assert row['capacity_just_below_lower_bound']<floor<=row['capacity_at_lower_bound']
        cap = row['upper_bound_any_renters_or_priorities_pooled']
        row['max_total_matches_consistent_with_necessary_count'] = int(np.ceil(cap / row['quantile']))
    # At the selected grid point N=217 makes the first quartile exactly the 55th order statistic.
    # Construct a group-only matching by preserving every threshold placement, then extending
    # using disjoint vertices at any feasible cost. Higher-income renters take no units.
    q = result['lower']['thresholds'][0]
    threshold = q['lower_bound_on_order_stat_at_observed_total']
    def pairs(E):
        assignment = maximum_bipartite_matching(csr_matrix(E),perm_type='column')
        return [(i,int(j)) for i,j in enumerate(assignment) if j>=0]
    cheap_edges = []
    additional = []
    for i,a in enumerate(low_matrices):
        cheap = pairs(a<=threshold)
        cheap_edges.extend((i,b,s) for b,s in cheap)
        residual = np.isfinite(a)
        for b,s in cheap:
            residual[b,:] = False
            residual[:,s] = False
        additional.extend((i,b,s) for b,s in pairs(residual))
    sample_size = int(n[0])
    selected = cheap_edges + additional[:sample_size-len(cheap_edges)]
    assert len(cheap_edges) == q['capacity_at_lower_bound']
    assert len(selected)==sample_size
    assert len({(i,b) for i,b,s in selected})==len({(i,s) for i,b,s in selected})==sample_size
    burdens = [low_matrices[i][b,s] for i,b,s in selected]
    assert np.isfinite(burdens).all()
    achieved_q25 = float(np.quantile(burdens,.25))
    assert achieved_q25 == threshold
    witness = dict(target_quantile=.25,exact_minimum_q25=achieved_q25,
                   group='lower_income',placements=sample_size,
                   cheap_edges=len(cheap_edges),maximum_residual_given_chosen_cheap_matching=len(additional),
                   total_retainable_placements=len(cheap_edges)+len(additional),
                   checked_edges=selected, indices='market in calibration traversal, lower-group row in that market, listing column')
    args.out.write_text(json.dumps({'markets':markets,'cases':result,'attainment':witness},indent=2)+'\n')


if __name__=='__main__':
    main()
