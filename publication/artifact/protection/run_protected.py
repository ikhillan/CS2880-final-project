"""Compare end-to-end algorithms for minimum-cost all-proposer protection.

From the manuscript's artifact directory run
python protection/run_protected.py --archive /path/to/hidden-blocking-pairs/artifact
The archive supplies the original core instances and audit_solver.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy


def protection_thresholds(market, initial, full):
    thresholds = {}
    for buyer, old in enumerate(initial.buyer_match):
        if old < 0:
            continue
        best = full.buyer_match[buyer]
        order = market.buyer_rank[buyer]
        if best >= 0 and order[best] <= order[old]:
            thresholds[buyer] = order[old]
    return thresholds


def violations(market, outcome, thresholds):
    return [b for b, limit in thresholds.items()
            if outcome.buyer_match[b] < 0 or market.buyer_rank[b][outcome.buyer_match[b]] > limit]


def protected_reverse_delete(market, initial_mask, thresholds, deadline):
    mask = initial_mask
    checks = 0
    while True:
        changed = False
        for f in sorted(range(market.n_firms), key=lambda i: (-market.costs[i], i)):
            if perf_counter() >= deadline:
                return mask, checks
            if not (mask >> f) & 1:
                continue
            trial = mask ^ (1 << f)
            assignment, blockers = evaluate(market, trial)
            checks += 1
            if not blockers and not violations(market, assignment, thresholds):
                mask = trial
                changed = True
        if not changed:
            return mask, checks


def solve(market, threshold_rows, cap):
    start = perf_counter()
    deadline = start + cap
    full_mask = (1 << market.n_firms) - 1
    initial, _ = evaluate(market, 0)
    full, full_blockers = evaluate(market, full_mask)
    assert not full_blockers
    thresholds = protection_thresholds(market, initial, full)
    incumbent, warm_checks = protected_reverse_delete(market, full_mask, thresholds, deadline)
    upper = market.cost(incumbent)
    warm_cost = upper
    warm_seconds = perf_counter() - start
    master = _Master(market, stable_support=True)
    if threshold_rows:
        for buyer, old_rank in thresholds.items():
            columns = {master.p + master.m + market.edge_index[buyer, seller]: 1
                       for seller in market.buyer_prefs[buyer][:old_rank + 1]}
            master.add_row(columns, 1, 1)
    buyer_components = {b: market.edge_component[k] for k, (b, _) in enumerate(market.edges)}
    lower = 0.0
    calls = []
    status = 'timeout'
    tolerance = 1e-6
    while perf_counter() < deadline and upper > lower + tolerance:
        result = master.solve(deadline - perf_counter(), upper)
        if result.status not in (0, 1):
            raise RuntimeError(f'{market.name}: unexpected MILP status {result.status}: {result.message}')
        dual = getattr(result, 'mip_dual_bound', None)
        if dual is not None and math.isfinite(dual):
            assert dual <= upper + tolerance
            lower = max(lower, min(float(dual), upper))
        record = {'solver_status': int(result.status), 'bound': lower}
        calls.append(record)
        if result.x is None:
            assert result.status == 1
            break
        binary = result.x[:market.n_firms]
        assert np.max(np.abs(binary - np.round(binary)), initial=0) < 1e-5
        candidate = sum(1 << f for f, value in enumerate(binary) if value > 0.5)
        cost = market.cost(candidate)
        assert cost <= upper + tolerance
        if result.status == 0:
            lower = max(lower, cost)
        assignment, blockers = evaluate(market, candidate)
        failed = violations(market, assignment, thresholds)
        if threshold_rows:
            assert not failed, 'Protected support witness failed to transfer protection to DA'
        record.update(mask=candidate, bound=lower, candidate_cost=cost,
                      blockers=len(blockers), harmed_above_floor=len(failed))
        if not blockers and not failed:
            if cost < upper:
                upper, incumbent = cost, candidate
        if upper <= lower + tolerance or result.status == 1:
            break
        assert blockers or failed
        components = {market.edge_component[edge] for edge in blockers}
        components.update(buyer_components[b] for b in failed)
        exposure = market.exposure(candidate)
        added = 0
        for component in sorted(components):
            prefix = [k for k in proposal_prefix(market, assignment, component)
                      if market.edges[k] not in market.initial_edges]
            added += master.nogood((master.p + k for k in prefix), (exposure[k] for k in prefix))
        assert added > 0
    if upper <= lower + tolerance:
        status = 'optimal'
    returned, blockers = evaluate(market, incumbent)
    assert not blockers and not violations(market, returned, thresholds)
    return {'status': status, 'mask': incumbent, 'upper': upper, 'lower': lower,
            'absolute_gap': max(0., upper - lower), 'calls': len(calls), 'cuts': len(master.cut_keys),
            'warm_cost': warm_cost, 'warm_checks': warm_checks, 'warm_seconds': warm_seconds,
            'seconds': perf_counter() - start, 'rounds': calls}


def independent_response(data, mask):
    """Round-based DA and direct blocking check, independent of audit_solver.evaluate."""
    bp, sp = data['buyer_prefs'], data['seller_prefs']
    bf, sf = data['buyer_firms'], data['seller_firms']
    initial_edges = {tuple(edge) for edge in data['initial_edges']}
    lists = [[s for s in choices if (b, s) in initial_edges or
              mask & ((1 << bf[b]) | (1 << sf[s]))] for b, choices in enumerate(bp)]
    held, cursor = [-1] * len(sp), [0] * len(bp)
    while True:
        assigned = set(held)
        applicants = defaultdict(list)
        for b, choices in enumerate(lists):
            if b not in assigned and cursor[b] < len(choices):
                applicants[choices[cursor[b]]].append(b)
                cursor[b] += 1
        if not applicants:
            break
        for s, buyers in applicants.items():
            if held[s] >= 0:
                buyers.append(held[s])
            held[s] = min(buyers, key=sp[s].index)
    partners = [-1] * len(bp)
    for s, b in enumerate(held):
        if b >= 0:
            partners[b] = s
    ranks = [prefs.index(s) if s != -1 else len(prefs) for prefs, s in zip(bp, partners)]
    has_blocker = any(held[s] == -1 or sp[s].index(b) < sp[s].index(held[s])
                      for b, prefs in enumerate(bp) for s in prefs[:ranks[b]])
    return tuple(partners), tuple(ranks), has_blocker


def oracle(data, ordinary, ordinary_masks, exhaustive=False):
    nf = len(data['costs'])
    _, initial, _ = independent_response(data, 0)
    _, full, blocks = independent_response(data, (1 << nf) - 1)
    assert not blocks
    floor = {b for b, (x, y) in enumerate(zip(full, initial)) if x > y}

    def feasible(mask):
        _, ranks, blocks = independent_response(data, mask)
        return not blocks and {b for b, (x, y) in enumerate(zip(ranks, initial)) if x > y} == floor

    def cost(mask):
        return sum(c for f, c in enumerate(data['costs']) if mask >> f & 1)

    if not exhaustive:
        assert ordinary_masks and all(cost(m) == ordinary for m in ordinary_masks)
        checks = [feasible(m) for m in ordinary_masks]
        if any(checks):
            return ordinary, feasible, len(checks)
    else:
        checks = []
    masks = sorted(range(1 << nf), key=lambda a: (cost(a), a))
    for mask in masks:
        if exhaustive or cost(mask) > ordinary:
            checks.append(feasible(mask))
            if checks[-1]:
                return cost(mask), feasible, len(checks)
    raise AssertionError('Full visibility provides a protected solution')


def tiny_checks():
    rng = random.Random(481006)
    controls = []
    for i in range(40):
        nb, ns = rng.randint(2, 5), rng.randint(2, 5)
        edges = [(b, s) for b in range(nb) for s in range(ns) if rng.random() < .62]
        buyers = [[s for b0, s in edges if b0 == b] for b in range(nb)]
        sellers = [[b for b, s0 in edges if s0 == s] for s in range(ns)]
        for p in buyers + sellers:
            rng.shuffle(p)
        data = {'buyer_prefs': buyers, 'seller_prefs': sellers,
                'buyer_firms': [rng.randrange(5) for _ in buyers],
                'seller_firms': [rng.randrange(5) for _ in sellers],
                'costs': [rng.randint(1, 3) for _ in range(5)],
                'initial_edges': [e for e in edges if rng.random() < .48]}
        gold, feasible, _ = oracle(data, 0., [], exhaustive=True)
        market = Market.from_dict(data)
        for rows in (False, True):
            solution = solve(market, rows, 15.)
            assert solution['status'] == 'optimal' and solution['upper'] == gold
            assert feasible(solution['mask'])
        controls.append(gold)
    return len(controls)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=here.parents[2] / 'hidden-blocking-pairs/artifact')
    parser.add_argument('--out', type=Path, default=here / 'protected_results.json')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cap', type=float, default=15.)
    args = parser.parse_args()
    assert args.repeats >= 1 and args.cap > 0
    sys.path.insert(0, str(args.archive / 'methodology'))
    global Market, _Master, evaluate, proposal_prefix
    from audit_solver import Market, _Master, evaluate, proposal_prefix
    small = tiny_checks()
    files = sorted((args.archive / 'experiments/core/instances').glob('*.json'))
    assert len(files) == 160
    input_digest = hashlib.sha256()
    for filename in files:
        input_digest.update(filename.name.encode() + b'\0' + hashlib.sha256(filename.read_bytes()).digest())
    random.Random(190971).shuffle(files)
    rows = []
    methods = ['ordinary_rows', 'protection_rows']
    for order_index, filename in enumerate(files):
        instance = json.loads(filename.read_text())
        data = instance['market']
        market = Market.from_dict(data)
        ordinary = next(r['cost'] for r in instance['results'] if r['method'] == 'enumeration')
        gold, feasible, checked = oracle(data, ordinary, instance['optimal_outcome_sensitivity']['masks'])
        runs = {key: [] for key in methods}
        for rep in range(args.repeats):
            for key in (methods if (order_index + rep) % 2 == 0 else reversed(methods)):
                run = solve(market, key == 'protection_rows', args.cap)
                assert run['lower'] <= gold + 1e-6 and run['upper'] >= gold - 1e-6
                assert feasible(run['mask'])
                if run['status'] == 'optimal':
                    assert abs(run['upper'] - gold) <= 1e-6
                runs[key].append(run)
        assert len({run['warm_cost'] for method in runs.values() for run in method}) == 1
        rows.append({'name': market.name, 'seed': data['metadata']['seed'],
                     'ordinary_optimum': ordinary, 'protected_optimum': gold,
                     'ordinary_optimum_masks': len(instance['optimal_outcome_sensitivity']['masks']),
                     'oracle_checks': checked, 'runs': runs})
        if (order_index + 1) % 40 == 0:
            print(f'{order_index + 1} markets completed', flush=True)
    counts = {}
    for key in methods:
        first = [r['runs'][key][0] for r in rows]
        counts[key] = {
            'certified_markets_first_repeat': sum(r['status'] == 'optimal' for r in first),
            'certified_total_runs': sum(t['status']=='optimal' for r in rows for t in r['runs'][key]),
            'master_calls_total_first_repeat': sum(r['calls'] for r in first),
            'master_calls_distribution_first_repeat': dict(sorted(Counter(r['calls'] for r in first).items())),
            'median_seconds_across_market_medians': statistics.median(
                statistics.median(t['seconds'] for t in r['runs'][key]) for r in rows),
            'positive_gap_markets_first_repeat': sum(r['absolute_gap'] > 1e-6 for r in first),
        }
    seed_blocks = defaultdict(list)
    for r in rows:
        pa = statistics.median(t['seconds'] for t in r['runs']['protection_rows'])
        ba = statistics.median(t['seconds'] for t in r['runs']['ordinary_rows'])
        seed_blocks[r['seed']].append(math.log(pa / ba))
    assert all(len(x) == 8 for x in seed_blocks.values()) and len(seed_blocks)==20
    seed_means = np.array([np.mean(seed_blocks[s]) for s in sorted(seed_blocks)])
    rng = np.random.default_rng(90503)
    boot = np.exp(rng.choice(seed_means, (20000,20), replace=True).mean(axis=1))
    differences = [r['runs']['ordinary_rows'][0]['calls'] - r['runs']['protection_rows'][0]['calls']
                   for r in rows]
    summary = {'markets': len(rows), 'tiny_checks': small, 'ordinary_optimum_masks_checked': sum(
        r['ordinary_optimum_masks'] for r in rows),
        'positive_premiums': sum(r['protected_optimum'] > r['ordinary_optimum'] for r in rows),
        'reverse_delete_optimal_markets': sum(r['runs'][methods[0]][0]['warm_cost']==r['protected_optimum'] for r in rows),
        'threshold_calls_fewer_equal_more': [sum(x>0 for x in differences),sum(x==0 for x in differences),sum(x<0 for x in differences)],
        'threshold_over_ordinary_geometric_runtime_ratio': float(np.exp(np.mean(seed_means))),
        'seed_block_bootstrap_95': np.quantile(boot,[.025,.975]).tolist(), 'methods': counts}
    result = {'design': {'objective': 'all proposers protected above unavoidable loss floor',
                         'repeats': args.repeats, 'wall_cap_seconds_each': args.cap, 'highs_threads': 1,
                         'warm_start': 'full visibility, repeated deterministic protected reverse-delete',
                         'cuts': 'component prefix at all components with global blocker or violated protection',
                         'order': 'seeded permutation of 160 input files; first method alternates by file and repeat',
                         'timing': 'one process, warm start/model construction/verification included, oracle truth and input parsing excluded',
                         'python':sys.version.split()[0], 'numpy':np.__version__, 'scipy':scipy.__version__,
                         'platform':platform.platform(), 'core_sha256':input_digest.hexdigest(),
                         'audit_solver_sha256':hashlib.sha256((args.archive/'methodology/audit_solver.py').read_bytes()).hexdigest()},
              'summary': summary,'markets': rows}
    tmp = args.out.with_suffix('.json.partial')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    os.replace(tmp,args.out)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
