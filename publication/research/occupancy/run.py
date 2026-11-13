"""Account for a paired change in the number of lower-income renters matched."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


MONTHS = ('2026-06', '2026-07', '2026-08')
BOROUGHS = ('Bronx', 'Brooklyn', 'Manhattan', 'Queens')
BUDGETS = (('k5', {'k': 5}), ('q40', {'q': 0.4}))
LABELS = ('vacant', 'lower', 'higher')


def summarize(counts, weights, freq):
    # Fields are lower denominator, nine row-major transitions, and three coverage counts.
    pooled = np.einsum('rs,scm->rcm', freq, counts)
    assert (pooled[:, :, 0] > 0).all()
    rates = np.einsum('c,rcm->rm', weights, pooled[:, :, 1:10] / pooled[:, :, :1]) * 100
    coverage = np.einsum('c,rcm->rm', weights, pooled[:, :, 10:]) / counts.shape[0]
    scores = {f'{old}_to_{new}': rates[:, 3 * i + j] for i, old in enumerate(LABELS)
              for j, new in enumerate(LABELS) if old != new and 'lower' in (old, new)}
    net_high = scores['higher_to_lower'] - scores['lower_to_higher']
    net_vacant = scores['vacant_to_lower'] - scores['lower_to_vacant']
    scores.update(net_higher_to_lower=net_high, net_vacant_to_lower=net_vacant,
                  lower_match_change_pp=net_high + net_vacant)
    for col, name in enumerate(['higher_reached_change', 'shared_reached_change', 'listings_reached_by_lower']):
        scores[name] = coverage[:, col]
    return {key: {'estimate': float(vals[0]), 'simulation_ci95': np.quantile(vals[1:], [.025, .975]).tolist()}
            for key, vals in scores.items()}


def run(repo, reference):
    sys.path.insert(0, str(repo.resolve()))
    from experiments.config import expand_labeled, load_config
    from hiddenoptions.data import borough_weights
    from hiddenoptions.exposure import transparent_exposure, uniform_exposure
    from hiddenoptions.markets import Catalog, InsufficientSupply, build_market
    from hiddenoptions.matching import deferred_acceptance

    catalog = Catalog()
    shares = borough_weights(catalog.renters)
    cells = [(b, m) for b in BOROUGHS for m in MONTHS]
    weights = np.array([shares[b] / 3 for b, m in cells])
    weights /= weights.sum()
    config = repo / 'configs' / 'stage1_pilot.json'
    arrays = {v: np.zeros((20, len(cells), 2, 13), dtype=int) for v in ('primary', 'enriched')}
    seen = set()
    for spec, variant in expand_labeled(load_config(config)):
        assert spec.month in MONTHS and spec.donors == 'calibration'
        try:
            market = build_market(spec, catalog)
        except InsufficientSupply:
            assert spec.borough == 'Staten Island'
            continue
        block = spec.seed - (100 if variant == 'primary' else 150)
        cell = cells.index((spec.borough, spec.month))
        assert (variant, block, cell) not in seen
        seen.add((variant, block, cell))
        lower = market.renters.lower_income.to_numpy(bool)
        for qi, (_, quota) in enumerate(BUDGETS):
            baseline = uniform_exposure(market, **quota)
            treated = np.where(lower[:, None], baseline, transparent_exposure(market, **quota))
            assert np.array_equal(baseline[lower], treated[lower])
            assert np.array_equal(baseline.sum(1), treated.sum(1))
            states, n_lower = [], []
            for graph in (baseline, treated):
                renter_match, listing_match = deferred_acceptance(market.inst, graph)
                state = np.zeros(listing_match.size, dtype=int)
                filled = listing_match >= 0
                state[filled] = np.where(lower[listing_match[filled]], 1, 2)
                n_lower.append(np.count_nonzero(lower & (renter_match >= 0)))
                assert np.count_nonzero(state == 1) == n_lower[-1]
                states.append(state)
            transitions = np.bincount(3 * states[0] + states[1], minlength=9)
            mat = transitions.reshape(3, 3)
            assert mat.sum() == spec.n_listings
            assert mat[:, 1].sum() - mat[1, :].sum() == n_lower[1] - n_lower[0]
            reached_lower = baseline[lower].any(0)
            reached_h0, reached_h1 = baseline[~lower].any(0), treated[~lower].any(0)
            coverage = [reached_h1.sum() - reached_h0.sum(),
                        (reached_h1 & reached_lower).sum() - (reached_h0 & reached_lower).sum(),
                        reached_lower.sum()]
            arrays[variant][block, cell, qi] = [lower.sum(), *transitions, *coverage]
    assert len(seen) == 480
    reps = 10_000
    freq = np.vstack([np.ones(20), np.random.default_rng(731911).multinomial(20, np.repeat(.05, 20), size=reps)])
    ref = json.loads(reference.read_text())
    panels = {}
    for variant in arrays:
        panels[variant] = {}
        for qi, (budget, _) in enumerate(BUDGETS):
            result = summarize(arrays[variant][:, :, qi, :], weights, freq)
            independent = ref['panels'][variant][budget]['UT_minus_UU_matched_lower']
            here = result['lower_match_change_pp']
            assert abs(here['estimate'] - independent['estimate']) < 1e-10
            assert np.allclose(here['simulation_ci95'], independent['interval_95'], atol=1e-10, rtol=0)
            panels[variant][budget] = result
    return {'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
            'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
            'design': '20 original blocks per design; four boroughs, Jun–Aug 2026, 240 markets per design',
            'units': 'Occupancy transitions per 100 lower-income renter draws; coverage in listings per market',
            'bootstrap': {'whole_seed_blocks': 20, 'draws': reps, 'seed': 731911}, 'panels': panels}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--repo', required=True, type=Path)
    cli.add_argument('--reference', required=True, type=Path, help='independent transparent_spillover.json')
    cli.add_argument('--out', required=True, type=Path)
    args = cli.parse_args()
    args.out.write_text(json.dumps(run(args.repo, args.reference), indent=2, allow_nan=False) + '\n')
