"""Optimize school or rental shortlists subject to a fixed stable assignment.

Only aggregates are written. The score policy never receives the reference partners.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

import numpy as np


def rows(owner, selected, n):
    return np.bincount(owner[selected], minlength=n)


def stable_edges(owner, program, rank, priority, capacity, old):
    held = old[old >= 0]
    occupancy = np.bincount(program[held], minlength=len(capacity))
    worst = np.full(len(capacity), -np.inf)
    np.maximum.at(worst, program[held], priority[held])
    old_rank = np.full(len(old), np.inf)
    old_rank[old >= 0] = rank[held]
    blocks = ((rank < old_rank[owner])
              & ((occupancy[program] < capacity[program]) | (priority < worst[program])))
    return ~blocks


def best_menu(owner, pool, required, quota, score, policy_b, policy_a):
    """Select required choices, then highest scores in the allowable pool.

    Equal scores favor the candidate model menu, then the baseline menu, then
    ascending input edge index. The original score picks are thus reproducible
    even when a frozen school tie is otherwise unobserved here.
    """
    assert np.all(pool[required])
    n = len(quota)
    free = quota - rows(owner, required, n)
    assert np.all(free >= 0)
    candidates = np.flatnonzero(pool & ~required)
    selected = required.copy()
    order = np.lexsort((candidates, ~policy_a[candidates], ~policy_b[candidates],
                        -score[candidates], owner[candidates]))
    edges = candidates[order]
    groups = owner[edges]
    if len(groups):
        positions = np.arange(len(groups))
        starts = np.maximum.accumulate(np.where(np.r_[True, groups[1:] != groups[:-1]], positions, 0))
        selected[edges[positions - starts < free[groups]]] = True
    assert np.array_equal(rows(owner, selected, n), quota)
    return selected


def count_hits(display, target):
    return int(display[target[target >= 0]].sum())


def oracle_gain(target, baseline, pool, required, free, owner):
    # Each row has at most one reference edge. Retain all current hits. A
    # missing hit can be added iff eligible with a free discretionary slot.
    edges = target[target >= 0]
    return int((~baseline[edges] & pool[edges] &
                (required[edges] | (free[owner[edges]] > 0))).sum())


def compare(owner, program, rank, priority, capacity, a, b, protected, old,
            score, solve, refs, cohort, site):
    n = len(old)
    quota = rows(owner, a, n)
    assert np.array_equal(quota, rows(owner, b, n))
    assert np.all(a[protected]) and np.all(b[protected])
    allowed = stable_edges(owner, program, rank, priority, capacity, old)
    assert np.all(allowed[a])
    required = protected.copy()
    required[old[old >= 0]] = True
    assert np.all(a[required])
    free = quota - rows(owner, required, n)
    assert np.isfinite(score).all()
    # Check that the same score/tie order without stability constraints
    # exactly reproduces B at this quota on the entire candidate pool.
    all_edges = np.ones(len(a), bool)
    unrestricted = best_menu(owner, all_edges, protected, quota, score, b, a)
    assert np.array_equal(unrestricted, b)
    # Positive control when the entire stable allocation and all preferences
    # are known. Keep M's own edges, then use the same model score to fill.
    full = refs['full']
    full_required = protected.copy()
    full_required[full[full >= 0]] = True
    control = best_menu(owner, all_edges, full_required, quota, score, b, a)
    assert np.all(stable_edges(owner, program, rank, priority, capacity, full)[control])
    replay = solve(control)
    assert np.array_equal(replay >= 0, full >= 0)
    assert np.all(rank[replay[full >= 0]] <= rank[full[full >= 0]])
    control_stats = {'full_control_placed':int((replay >= 0).sum()),
                     'full_control_placement_gain_vs_b':int((replay >= 0).sum()-(solve(b)>=0).sum())}
    for label,ref in refs.items():
        control_stats[label+'_full_control_coverage'] = count_hits(control,ref)
        control_stats[label+'_full_control_gain_vs_b'] = count_hits(control,ref) - count_hits(b,ref)
    changed = rows(owner, a ^ b, n) > 0
    bad_rows = (rows(owner, required & ~b, n) + rows(owner, ~allowed & b, n)) > 0
    assert not np.any(bad_rows & ~changed)
    outcomes = []
    for name, candidates in [('union', a | b), ('all', all_edges)]:
        pool = candidates & allowed
        shown = best_menu(owner, pool, required, quota, score, b, a)
        assert np.all(pool[shown]) and np.all(shown[required])
        new = solve(shown)
        assert np.array_equal(old >= 0, new >= 0)
        matched = old >= 0
        assert np.all(rank[new[matched]] <= rank[old[matched]])
        rec = Counter(markets=1, people=n, placed=int(matched.sum()),
                      proposed_rows=int(changed.sum()), whole_rows_flagged=int(bad_rows.sum()),
                      score_edited_rows=int((rows(owner, shown ^ a, n)>0).sum()),
                      baseline_replaced_options=int((shown & ~a).sum()),
                      proposed_options_outside_baseline=int((b & ~a).sum()),
                      proposed_new_options_kept=int((shown & b & ~a).sum()),
                      improved_assignments=int((rank[new[matched]]<rank[old[matched]]).sum()))
        rec.update(control_stats)
        for label, ref in refs.items():
            valid = ref >= 0
            idx = ref[valid]
            assert np.array_equal(owner[idx], np.flatnonzero(valid))
            whole_ceiling = (~a[idx] & b[idx] & ~bad_rows[valid]).sum()
            baseline = count_hits(a, ref)
            optimum = oracle_gain(ref, a, pool, required, free, owner)
            achieved = count_hits(shown, ref) - baseline
            assert optimum >= whole_ceiling and optimum >= achieved
            # Construct an attaining display at the fixed quota. Choosing A in
            # ties retains every reference hit that the baseline already covers.
            oracle_scores = np.zeros(len(a), float)
            oracle_scores[idx] = 1
            oracle = best_menu(owner, pool, required, quota, oracle_scores, a, a)
            assert np.all(oracle[required]) and not np.any(oracle & ~allowed)
            assert count_hits(oracle, ref) == baseline + optimum
            missed = idx[~a[idx]]
            outside_pool = ~candidates[missed]
            blocked = candidates[missed] & ~allowed[missed]
            no_slot = pool[missed] & (free[owner[missed]] == 0)
            obtainable = pool[missed] & (free[owner[missed]] > 0)
            assert np.all(outside_pool.astype(int) + blocked + no_slot + obtainable == 1)
            assert int(obtainable.sum()) == optimum
            rec.update({label+'_reference_edges':len(idx), label+'_baseline':baseline,
                        label+'_missed':len(missed), label+'_outside_pool':int(outside_pool.sum()),
                        label+'_blocks':int(blocked.sum()),label+'_no_slot':int(no_slot.sum()),
                        label+'_candidate_gain':count_hits(b,ref)-baseline,
                        label+'_whole_oracle_gain':int(whole_ceiling),
                        label+'_slot_oracle_gain':optimum,label+'_slot_score_gain':achieved})
        outcomes.append((site,cohort,name,rec))
    return outcomes


def school(repo, numeric, years):
    sys.path.insert(0,str(repo/'publication/research/policy/equity'))
    from transitions import Market
    for year in years:
        m = Market(numeric/f'{year}.bin')
        a,b = [np.fromfile(numeric/f'{year}-{tag}.mask', np.uint8).astype(bool)
               for tag in ('blind','rank')]
        with (numeric/f'{year}.bin').open('rb') as f:
            f.seek(-8*m.ne, 2)
            score = np.fromfile(f, '<f8', m.ne)
        def edge_ids(partners):
            hit = np.flatnonzero(m.j == partners[m.i])
            assert np.unique(m.i[hit]).size == len(hit)
            out = np.full(m.n,-1,int)
            out[m.i[hit]] = hit
            return out
        old,full = [edge_ids(np.fromfile(numeric/f'{year}-{tag}.partners','<i4'))
                    for tag in ('blind','full')]
        assert np.array_equal(old, m.solve(a))
        refs = {'full':full, 'history':edge_ids(m.off)}
        yield from compare(m.i,m.j,m.cols,m.pr,m.cap,a,b,m.j==m.ret[m.i],old,
                           score,m.solve,refs,str(year),'Chile')


def housing(repo):
    sys.path.insert(0,str(repo))
    from experiments.config import expand_labeled, load_config
    from experiments.pilot_outcomes import CONFIG, DEFAULT_CACHE, load_pilot_cache, pilot_displays
    from hiddenoptions.markets import Catalog, InsufficientSupply, build_market
    from hiddenoptions.matching import deferred_acceptance
    cache, catalog = load_pilot_cache(DEFAULT_CACHE), Catalog()
    for spec, panel in expand_labeled(load_config(CONFIG)):
        try:
            market = build_market(spec,catalog)
        except InsufficientSupply:
            continue
        inst = market.inst
        owner, prog = np.nonzero(inst.E)
        index = np.full(inst.E.shape,-1,int)
        index[owner,prog] = np.arange(len(owner))
        a,b = [v[owner,prog] for v in pilot_displays(market,cache)]
        def solve(mask):
            display = np.zeros_like(inst.E, dtype=bool)
            display[owner,prog] = mask
            p = deferred_acceptance(inst,display)[0]
            return np.where(p>=0,index[np.arange(len(p)), p],-1)
        old,full = solve(a),solve(np.ones(len(a),bool))
        # The cached model provides a selected set and no calibrated scores for
        # unselected listings. Its explicit objective is to retain those choices.
        yield from compare(owner,prog,inst.rank_b[owner,prog],inst.rank_s[owner,prog],
                           np.ones(inst.n_s,int),a,b,np.zeros(len(a),bool),old,
                           b.astype(float),solve,{'full':full},panel,'NYC')


def main(args):
    repo = args.repo.resolve()
    stream = school(repo,args.numeric,args.years) if args.site=='school' else housing(repo)
    totals = {}
    for site,panel,pool,rec in stream:
        key=(site,panel,pool)
        totals.setdefault(key,Counter()).update(rec)
        if site=='Chile':
            print(panel,pool,dict(rec),flush=True)
    records = [dict(site=site,panel=panel,pool=pool,**counts)
               for (site,panel,pool),counts in totals.items()]
    with args.out.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0],lineterminator='\n')
        w.writeheader()
        w.writerows(records)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--site',choices=['school','housing'],required=True)
    p.add_argument('--numeric',type=Path)
    p.add_argument('--years',type=int,nargs='+',default=[2022,2023])
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
