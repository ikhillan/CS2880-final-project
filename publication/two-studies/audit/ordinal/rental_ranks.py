"""Assignment rank, on the original feasible order, in the archived NYC policy comparison.

Writes market-level counts only. Run with the project dependencies and June--August catalogs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(repo, out):
    sys.path.insert(0, str(repo))
    from experiments.analyze import cell_weights
    from experiments.config import expand_labeled, load_config
    from experiments.pilot_outcomes import CONFIG, DEFAULT_CACHE, load_pilot_cache, pilot_displays
    from hiddenoptions.markets import Catalog, InsufficientSupply, build_market
    from hiddenoptions.matching import deferred_acceptance

    plan = expand_labeled(load_config(CONFIG))
    weights = cell_weights(sorted({(s.borough, s.month) for s, _ in plan}))
    cache, catalog = load_pilot_cache(DEFAULT_CACHE), Catalog()
    rows, changes, skipped = [], [], []
    for s, design in plan:
        try:
            market = build_market(s, catalog)
        except InsufficientSupply:
            skipped.append(s.borough)
            continue
        inst = market.inst
        meta = dict(design=design, seed=s.seed, borough=s.borough, month=s.month,
                    cell_weight=weights[s.borough, s.month], applicants=inst.n_b)
        a, b = pilot_displays(market, cache)
        ranks = []
        for display in (a, b):
            partner, _ = deferred_acceptance(inst, display)
            matched = partner >= 0
            rank = np.full(inst.n_b, inst.n_s + 1, dtype=int)
            for i in np.flatnonzero(matched):
                rank[i] = inst.prefs_b[i].index(partner[i]) + 1
                assert display[i, partner[i]] and rank[i] == inst.rank_b[i, partner[i]] + 1
            ranks.append(rank)
        ra, rb = ranks
        ma, mb = ra <= inst.n_s, rb <= inst.n_s
        changes.append(dict(**meta, better=int((rb < ra).sum()), worse=int((rb > ra).sum()),
                            unchanged=int((rb == ra).sum()), enter=int((~ma & mb).sum()),
                            exit=int((ma & ~mb).sum()), both_improve=int((ma & mb & (rb < ra)).sum()),
                            both_worsen=int((ma & mb & (rb > ra)).sum())))
        for r in range(1, inst.n_s + 1):
            rows.append(dict(**meta, cutoff=r, uniform=int((ra <= r).sum()), haiku=int((rb <= r).sum())))
    assert len(changes) == 480 and len(skipped) == 120 and set(skipped) == {'Staten Island'}
    counts = pd.DataFrame(rows)
    paired = pd.DataFrame(changes)
    assert ((paired.better + paired.worse + paired.unchanged) == paired.applicants).all()
    assert (paired.better == paired.enter + paired.both_improve).all()
    assert (paired.worse == paired.exit + paired.both_worsen).all()
    assert (counts.groupby(['design', 'cutoff']).size() == 240).all()
    endpoint = counts[counts.cutoff == 64].reset_index(drop=True)
    np.testing.assert_array_equal(endpoint.haiku - endpoint.uniform, paired.enter - paired.exit)
    std = []
    for (design, r), panel in counts.groupby(['design', 'cutoff']):
        # All rows have equal n=64 per cell per block. Bootstrap the entire paired block.
        weights_per_seed = panel.groupby('seed').cell_weight.transform('sum')
        diff = panel.cell_weight / weights_per_seed * 100 * (panel.haiku-panel.uniform)/panel.applicants
        byseed = diff.groupby(panel.seed).sum().to_numpy()
        assert len(byseed) == 20
        boot = np.random.default_rng(0).choice(byseed, (10000, len(byseed)), replace=True).mean(axis=1)
        lo, hi = np.quantile(boot, [.025, .975])
        std.append(dict(design=design, cutoff=r, delta_pp=byseed.mean(), lo_pp=lo, hi_pp=hi,
                        n_uniform=panel.uniform.sum(), n_haiku=panel.haiku.sum()))
    summary = pd.DataFrame(std)
    endpoints = pd.read_csv(repo/'publication/two-studies/evidence/policy-rates.csv').query(
        "comparison=='haiku_vs_uniform' and group=='all' and metric=='placed'").sort_values('design')
    np.testing.assert_allclose(summary[summary.cutoff==64].sort_values('design')[['delta_pp','lo_pp','hi_pp']],
                               endpoints[['delta_pp','lo_pp','hi_pp']], atol=1e-10)
    out.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out/'rental-rank-counts.csv.gz', index=False, compression={'method':'gzip','mtime':0})
    paired.to_csv(out/'rental-rank-transitions.csv', index=False)
    summary.to_csv(out/'rental-rank-summary.csv', index=False)
    print(summary[summary.cutoff.isin([1,3,5,64])].to_string(index=False))
    print(paired.groupby('design')[['better','worse','unchanged','enter','exit','both_improve','both_worsen']].sum())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    main(args.repo.resolve(), args.out)
