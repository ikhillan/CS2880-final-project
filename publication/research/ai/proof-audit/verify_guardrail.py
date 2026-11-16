"""Independent, exhaustive stable-match and quota-optimizer checks on small markets."""
import argparse
from functools import lru_cache
from itertools import combinations, permutations, product
import json
from pathlib import Path

import numpy as np
from guardrail import certificate, repair, repair_with_freezes, threat_mask


def ordered_subsets(m):
    return [x for size in range(m+1) for x in permutations(range(m), size)]


def stable(pref, priority):
    """Enumerate all assignments, reject infeasible pairs, then check each visible block."""
    possibilities = [p + (-1,) for p in pref]
    for match in product(*possibilities):
        inverse = {j:i for i,j in enumerate(match) if j >= 0}
        if len(inverse) != sum(j >= 0 for j in match):
            continue
        for i, row in enumerate(pref):
            blocks = False
            for j in row:
                if j == match[i]:
                    break
                if j not in inverse or priority[j].index(i) < priority[j].index(inverse[j]):
                    blocks = True
                    break
            if blocks:
                break
        else:
            yield match


def applicant_optimum(pref, allocations):
    ordered=[row+(-1,) for row in pref]
    optimum=tuple(min((mu[i] for mu in allocations),key=ordered[i].index) for i in range(len(pref)))
    assert optimum in allocations
    return optimum


@lru_cache(maxsize=None)
def maximum_matching(rows):
    """Tiny recursive oracle independent of SciPy's bipartite matching implementation."""
    if not rows:
        return 0
    others = rows[1:]
    return max([maximum_matching(others)] + [1 + maximum_matching(tuple(r & ~(1<<j) for r in others))
                                              for j in range(rows[0].bit_length()) if rows[0] & (1<<j)])


def exhaustive(n, m):
    all_lists = ordered_subsets(m)
    all_orders = list(permutations(range(n)))
    counts = dict(stable_pair_checks=0, zero_net_turnover=0, gross_positive=0, positive_certificate_loose=0,
                  informative_priority=0, stable_pairs_differ=0, zero_threat_rank_tests=0)
    # Fix I as a prefix of each size, which suffices by applicant relabeling.
    for priority in product(all_orders, repeat=m):
        results = {p:list(stable(p, priority)) for p in product(all_lists, repeat=n)}
        for ni in range(1,n):
            for fixed in product(all_lists,repeat=ni):
                edits = list(product(all_lists,repeat=n-ni))
                # The original higher group may have any stable match given these incumbent lists.
                # Cache unique baselines: certification depends on the baseline assignment alone.
                baselines = {mu for p in edits for mu in results[fixed+p]}
                for pre in baselines:
                    q = {pre[i]:i for i in range(ni) if pre[i]>=0}
                    initial = {i for i in range(ni) if pre[i]>=0}
                    for edited in edits:
                        danger = []
                        blind = []
                        for h, row in enumerate(edited, start=ni):
                            blind.append(sum(1<<j for j in row if j in q))
                            danger.append(sum(1<<j for j in row if j in q and priority[j].index(h)<priority[j].index(q[j])))
                        bound = maximum_matching(tuple(danger))
                        blind_bound = maximum_matching(tuple(blind))
                        if bound==0:
                            best=applicant_optimum(fixed+edited,results[fixed+edited])
                            for i in range(ni):
                                if pre[i]>=0:
                                    order=fixed[i]+(-1,)
                                    assert order.index(best[i])<=order.index(pre[i])
                            counts['zero_threat_rank_tests']+=1
                        for post in results[fixed+edited]:
                            matched = {i for i in range(ni) if post[i]>=0}
                            loss = len(initial-matched)
                            assert loss <= bound <= blind_bound
                            counts['stable_pair_checks'] += 1
                            counts['stable_pairs_differ'] += pre != post
                            counts['zero_net_turnover'] += loss > 0 and len(initial)==len(matched)
                            counts['gross_positive'] += loss>0
                            counts['positive_certificate_loose'] += bound>loss
                            counts['informative_priority'] += bound<blind_bound
    return counts


def find_chain_and_tightness():
    # An actual loss can occur at a different unit from the outsider's acceptance, with
    # an equally large gain for a different incumbent. A net measure misses this loss.
    fixed = ((0,), (1,0), (2,))
    before = fixed + ((2,),)
    after = fixed + ((1,),)
    priority = ((1,0,2,3),(3,1,0,2),(3,2,0,1))
    pre, = stable(before, priority)
    post, = stable(after, priority)
    assert pre == (0,1,-1,2) and post == (-1,0,2,1)
    # Every graph (up to 3-by-3) can attain its matching number, also at fixed quota.
    checked = 0
    for bits in range(1<<9):
        C = [[j for j in range(3) if bits & (1 << (3*h+j))] for h in range(3)]
        max_match = max((mu for mu in product(*[[-1]+row for row in C])
                         if len({j for j in mu if j>=0})==sum(j>=0 for j in mu)),
                        key=lambda mu:sum(j>=0 for j in mu))
        covered = [h for h,j in enumerate(max_match) if j>=0]
        pri_order = tuple(covered + [h for h in range(3) if h not in covered] + [3,4,5])
        # Common priority implies serial dictatorship. Each outsider has three private fallback units.
        new_rows = []
        old_rows = []
        for h,row in enumerate(C):
            private = list(range(3+3*h,6+3*h))
            old_rows.append(private)
            first = [] if max_match[h]<0 else [max_match[h]]
            new_rows.append(first+[j for j in row if j not in first]+private[:3-len(row)])
        for j in range(3):
            old_rows.append([j]);new_rows.append([j])
        def sd(rows):
            held = set();served = set()
            for i in pri_order:
                for j in rows[i]:
                    if j not in held:
                        held.add(j);served.add(i);break
            return served
        assert len((sd(old_rows)-sd(new_rows)) & {3,4,5})==len(covered)
        checked += 1
    return dict(chain_before=pre,chain_after=post,common_priority_graph_witnesses=checked)


def optimizers():
    scores=np.array([[7,11,2],[13,5,17]],float)
    stats=dict(quota_programs=0,infeasible=0)
    # All threat masks in 2-by-3, all quotas 0..3 per row, loss budgets 0..2.
    for mask in range(64):
        danger=np.array([[(mask>>(3*h+j))&1 for j in range(3)] for h in range(2)], bool)
        feasible=np.ones_like(danger)
        for q in product(range(4),repeat=2):
            for b in range(3):
                best=None
                for sets in product(*(combinations(range(3),k) for k in q)):
                    rows = tuple(sum(1<<j for j in js if danger[h,j]) for h,js in enumerate(sets))
                    if maximum_matching(rows)>b:
                        continue
                    score=sum(scores[h,j] for h,js in enumerate(sets) for j in js)
                    best=score if best is None else max(best,score)
                x=repair(scores,feasible,danger,np.array(q),b)
                if x is None:
                    assert best is None
                    stats['infeasible']+=1
                else:
                    assert best==scores[x].sum()
                    assert certificate(danger&x)<=b
                stats['quota_programs']+=1
    # Negative scores, ties, incomplete feasible rows and variable quotas.
    rng=np.random.default_rng(91992)
    for _ in range(100):
        feasible=rng.random((3,4))<.65
        danger=feasible & (rng.random((3,4))<.55)
        values=rng.integers(-3,4,(3,4))
        q=np.array([rng.integers(n+1) for n in feasible.sum(1)])
        b=int(rng.integers(4))
        best=None
        for combo in product(*(combinations(np.flatnonzero(row),k) for row,k in zip(feasible,q))):
            selected=tuple(sum(1<<int(j) for j in js if danger[h,j]) for h,js in enumerate(combo))
            if maximum_matching(selected)>b:
                continue
            val=sum(values[h,j] for h,js in enumerate(combo) for j in js)
            best=val if best is None else max(best,val)
        x=repair(values,feasible,danger,q,b)
        assert (x is None and best is None) or (x is not None and values[x].sum()==best)
    stats['incomplete_quota_programs']=100
    return stats


def closure_check():
    rng=np.random.default_rng(139988)
    counter=0;cascades=0
    for _ in range(300):
        n, m=5,4
        feasible=rng.random((n,m))<0.65
        ranks=np.argsort(rng.random((n,m)),axis=1)
        pri_rank=np.argsort(rng.random((n,m)),axis=0)
        pri=tuple(tuple(np.argsort(pri_rank[:,j])) for j in range(m))
        initial=feasible & (rng.random((n,m))<0.75)
        preferences=tuple(tuple(j for j in np.argsort(ranks[i]) if initial[i,j]) for i in range(n))
        group=np.zeros(n,bool);group[rng.integers(0,n)]=True
        scores=rng.normal(size=(n,m))
        for base in stable(preferences,pri):
            for priority in [None,pri_rank]:
                slate,fixed,rounds=repair_with_freezes(feasible,scores,initial,group,np.array(base),priority)
                assert np.array_equal(slate.sum(1),initial.sum(1)) and np.array_equal(slate[fixed],initial[fixed])
                for mask in range(1<<n):
                    masked=np.array([bool(mask & (1<<i)) for i in range(n)])
                    if (group & ~masked).any():continue
                    bad=threat_mask(feasible,masked,np.array(base),priority)
                    if np.all(((feasible & ~bad).sum(1)>=initial.sum(1)) | masked):
                        assert not (fixed & ~masked).any()
                newp=tuple(tuple(j for j in np.argsort(ranks[i]) if slate[i,j]) for i in range(n))
                all_stable=list(stable(newp,pri))
                best=applicant_optimum(newp,all_stable)
                for i in np.flatnonzero(fixed & (np.array(base)>=0)):
                    order=newp[i]+(-1,)
                    assert order.index(best[i])<=order.index(base[i])
                for post in all_stable:
                    assert not (fixed & (np.array(base)>=0) & (np.array(post)<0)).any()
                counter+=1;cascades+=rounds>1
    assert cascades>0
    return dict(closure_checks=counter,realized_cascades=cascades)



def reduction_check():
    checks=0;n=5;ep=list(combinations(range(n),2))
    for mask in range(1<<len(ep)):
        edges=[edge for j,edge in enumerate(ep) if mask & 1<<j]
        cover=min(c.bit_count() for c in range(1<<n)
                  if all((c & 1<<a) or (c & 1<<b) for a,b in edges))
        minimum=min(len(set(chosen)) for chosen in product(*edges))
        assert minimum==cover
        checks+=1
    return checks



def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).with_name('checks.json'));a=p.parse_args()
    result={f'{n}x{m}': exhaustive(n,m) for n,m in [(2,3),(3,2)]}
    result.update(find_chain_and_tightness(),optimizer=optimizers(),closure=closure_check(),vertex_cover_reductions=reduction_check());a.out.write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':
    main()
