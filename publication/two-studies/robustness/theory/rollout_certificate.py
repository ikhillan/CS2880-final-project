"""Check the exact baseline-stability row rule in the two frozen market replays."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys

import numpy as np


def set_rows(owner, condition, n):
    result = np.zeros(n, dtype=bool)
    result[owner[condition]] = True
    return result


def certify(capacity, owner, program, rank, priority, a, b, old, target, solve, recorded=None):
    """Edges are indexed once. Applicant rank and program priority use smaller-is-better."""
    n = len(old)
    m = len(capacity)
    holder = old >= 0
    old_rank = np.full(n, np.iinfo(np.int64).max)
    old_rank[holder] = rank[old[holder]]
    occupied = np.bincount(program[old[holder]], minlength=m)
    cutoff = np.full(m, -np.inf)
    np.maximum.at(cutoff, program[old[holder]], priority[old[holder]])
    blocking = ((rank < old_rank[owner])
                & ((occupied[program] < capacity[program]) | (priority < cutoff[program])))
    assert not np.any(blocking & a)
    removes = np.zeros(n, dtype=bool)
    removes[holder] = ~b[old[holder]]
    exposes = set_rows(owner, b & blocking, n)
    flagged = removes | exposes
    changed = set_rows(owner, a != b, n)
    safe = np.where(flagged[owner], a, b)
    assert not np.any(safe & blocking) and np.all(safe[old[holder]])
    new, guarded = solve(b), solve(safe)
    assert np.array_equal(holder, guarded >= 0) and np.all(rank[guarded[holder]] <= old_rank[holder])
    k = np.bincount(program[old[flagged & holder]], minlength=m)
    t = np.bincount(program[new[flagged & (new >= 0)]], minlength=m)
    overlap = np.minimum(k,t).sum()
    bound = np.count_nonzero(flagged & (holder | (new >= 0))) - overlap
    lost = (new < 0) & holder
    gained = (~holder) & (new >= 0)
    assert max(lost.sum(), gained.sum()) <= bound <= flagged.sum() <= changed.sum()
    result = Counter(markets=1, applicants=n, changed=changed.sum(), removes=removes.sum(),
                     exposes=exposes.sum(), both=(removes & exposes).sum(), flagged=flagged.sum(),
                     screened_changes=(changed & ~flagged).sum(), bound=bound,
                     baseline_placed=holder.sum(), policy_b_placed=(new>=0).sum(),
                     screened_placed=(guarded>=0).sum(), losses=lost.sum(), gains=gained.sum(),
                     screened_rank_improvements=(rank[guarded[holder]] < old_rank[holder]).sum())
    newly_shown = (target>=0) & ~a[target.clip(0)] & b[target.clip(0)]
    named = newly_shown & (new == target)
    result.update(full_newly_shown_stable_row_and_named=(newly_shown & ~flagged & named).sum(),
                  full_newly_shown_flagged_and_named=(newly_shown & flagged & named).sum())
    for key, ref in [('full',target), ('recorded',recorded)]:
        if ref is None:
            continue
        valid = ref >= 0
        index = ref[valid]
        improvement = b[index].astype(int) - a[index].astype(int)
        free_rows = ~flagged[valid]
        result[f'{key}_safe_improving_rows'] = int(np.sum(free_rows & (improvement > 0)))
        result[f'{key}_safe_worsening_rows'] = int(np.sum(free_rows & (improvement < 0)))
        result[f'{key}_flagged_improving_rows'] = int(np.sum(~free_rows & (improvement > 0)))
        result[f'{key}_flagged_worsening_rows'] = int(np.sum(~free_rows & (improvement < 0)))
        for label, displayed in [('a',a),('b',b),('screened',safe)]:
            result[f'{key}_covered_{label}'] = int(displayed[ref[ref>=0]].sum())
    return result


def housing(repo):
    sys.path.insert(0,str(repo))
    from experiments.config import expand_labeled, load_config
    from experiments.pilot_outcomes import DEFAULT_CACHE, CONFIG, load_pilot_cache, pilot_displays
    from hiddenoptions.markets import Catalog, InsufficientSupply, build_market
    from hiddenoptions.matching import deferred_acceptance
    cache, catalog = load_pilot_cache(DEFAULT_CACHE), Catalog()
    for spec, panel in expand_labeled(load_config(CONFIG)):
        try:
            market = build_market(spec,catalog)
        except InsufficientSupply:
            continue
        inst = market.inst
        a,b = pilot_displays(market,cache)
        ii,jj = np.nonzero(inst.E)
        index = np.full(inst.E.shape,-1,int)
        index[ii,jj] = np.arange(len(ii))
        def solve(display):
            expanded = np.zeros_like(inst.E, dtype=bool)
            expanded[ii,jj] = display
            before = deferred_acceptance(inst,expanded)[0]
            return np.where(before>=0, index[np.arange(len(before)),before],-1)
        ma, mb = a[ii,jj], b[ii,jj]
        old, full = solve(ma), solve(np.ones(len(ii),bool))
        yield 'NYC', panel, certify(np.ones(inst.n_s,int), ii,jj,inst.rank_b[ii,jj],
                                    inst.rank_s[ii,jj],ma,mb,old,full,solve)


def schools(repo,numeric):
    sys.path.insert(0, str(repo/'publication/research/policy/equity'))
    from transitions import Market
    for year in (2022,2023):
        m = Market(numeric/f'{year}.bin')
        masks = {pol:np.fromfile(numeric/f'{year}-{pol}.mask',dtype=np.uint8).astype(bool)
                 for pol in ('blind','rank','full')}
        assert masks['full'].all()
        old = m.solve(masks['blind'])
        full = m.solve(masks['full'])
        recorded = np.full(m.n,-1,int)
        ids = np.flatnonzero(m.j == m.off[m.i])
        assert np.unique(m.i[ids]).size == ids.size
        recorded[m.i[ids]] = ids
        yield 'Chile', str(year), certify(m.cap,m.i,m.j,m.cols,m.pr,masks['blind'],masks['rank'],
                                          old,full,m.solve,recorded)


def main(args):
    totals = defaultdict(Counter)
    for source in (housing(args.repo.resolve()),schools(args.repo.resolve(),args.numeric)):
        for setting, panel, values in source:
            totals[setting,panel].update(values)
    columns = ['setting','panel',*sorted(set.union(*(set(v) for v in totals.values())))]
    with args.out.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for (setting,panel),v in sorted(totals.items()):
            writer.writerow(dict(setting=setting,panel=panel,**v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--numeric',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
