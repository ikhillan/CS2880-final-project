"""Exhaustive tests of matched-set predictions in strict, incomplete one-to-one markets.

Uses parallel integer proposals, and independently enumerates all stable matchings to
identify the DA outcome. No repository DA, source data or additional packages required.
"""
import argparse
import json
from functools import lru_cache
from itertools import combinations, permutations, product
from math import comb, factorial
from pathlib import Path


def bits(mask, n):
    return [i for i in range(n) if mask & (1 << i)]


def orders(mask, n):
    return tuple(permutations(bits(mask, n)))


@lru_cache(maxsize=None)
def partials(n, m):
    """All partial one-to-one allocations, including empty."""
    return [mu for mu in product(range(-1, m), repeat=n)
            if len([s for s in mu if s >= 0]) == len({s for s in mu if s >= 0})]


def run_da(preferences, priority):
    m, n = len(priority), len(preferences)
    held = [-1] * m
    cursor = [0] * n
    active = list(range(n))
    while active:
        offers = [[] for _ in range(m)]
        for i in active:
            if cursor[i] < len(preferences[i]):
                j = preferences[i][cursor[i]]
                cursor[i] += 1
                offers[j].append(i)
        active = []
        for j, applicants in enumerate(offers):
            if applicants:
                if held[j] >= 0:
                    applicants.append(held[j])
                held[j] = min(applicants, key=priority[j].index)
                active.extend(i for i in applicants if i != held[j])
    assigned = [-1] * n
    for j, i in enumerate(held):
        if i >= 0:
            assigned[i] = j
    return tuple(assigned)


def stable_optimum(pref, pri):
    """Return coordinatewise applicant optimum derived from all stable allocations."""
    n = len(pref)
    stable = []
    for mu in partials(n, len(pri)):
        if any(j >= 0 and j not in pref[i] for i, j in enumerate(mu)):
            continue
        inverse = {j:i for i,j in enumerate(mu) if j >= 0}
        blocks = False
        for i, order in enumerate(pref):
            own = mu[i]
            for j in order:
                if j == own:
                    break
                a = inverse.get(j)
                if a is None or pri[j].index(i) < pri[j].index(a):
                    blocks = True
                    break
            if blocks:
                break
        if not blocks:
            stable.append(mu)
    assert stable
    ranks = [{j:k for k,j in enumerate(p+(-1,))} for p in pref]
    best = tuple(min((m[i] for m in stable),key=ranks[i].__getitem__) for i in range(n))
    assert best in stable
    return best


def profiles(n,m):
    """Every labeled graph with every strict neighbor ordering on both sides."""
    for raw in range(1 << (n*m)):
        edges = tuple((raw >> (m*i)) % (1<<m) for i in range(n))
        lists = [orders(row, m) for row in edges]
        receiver_masks = [sum(1<<i for i,row in enumerate(edges) if row & (1<<j)) for j in range(m)]
        for pri in product(*(orders(mask,n) for mask in receiver_masks)):
            for pref in product(*lists):
                yield pref,pri


def check(n,m):
    counts = dict(profiles=0, all_outsider_coalitions=0, focal_truncated_reports=0,
                  uniform_quota_cases=0, retentions_with_different_partners=0,
                  singleton_capacity_labels_with_displaced_incumbent=0,
                  independent_full_stable_set_enumerations=0,
                  single_listing_resource_additions=0, gross_change_bounds=0,
                  exact_quota_winner_sets=0)
    for pref, pri in profiles(n,m):
        counts['profiles'] += 1

        def matched(mu):
            return frozenset(i for i,j in enumerate(mu) if j!=-1)

        @lru_cache(maxsize=None)
        def da(p):
            return run_da(p,pri)

        full = da(pref)
        assert full == stable_optimum(pref,pri)
        counts['independent_full_stable_set_enumerations']+=1
        resource_sets = []
        for stock in range(1 << m):
            p = tuple(tuple(j for j in row if stock & (1<<j)) for row in pref)
            mu = da(p)
            resource_sets.append(matched(mu))
            # Reverse proposers; priorities restricted to edges present in p.
            rp = tuple(tuple(i for i in pri[j] if j in p[i]) for j in range(m))
            reverse_mu = run_da(rp, p)
            assert {i for i in reverse_mu if i>=0} == matched(mu)
            assert {s for s in range(m) if reverse_mu[s]>=0} == {j for j in mu if j>=0}
        for stock in range(1 << m):
            for added in range(m):
                if not stock & (1<<added):
                    enlarged = resource_sets[stock | (1<<added)]
                    assert resource_sets[stock] <= enlarged
                    assert len(enlarged - resource_sets[stock]) <= 1
                    counts['single_listing_resource_additions']+=1
        for x in range(1 << m):
            for y in range(1 << m):
                assert len(resource_sets[x]-resource_sets[y]) <= (x & ~y).bit_count()
                assert len(resource_sets[y]-resource_sets[x]) <= (y & ~x).bit_count()
                counts['gross_change_bounds']+=1
        for outsiders in range(1 << n):
            incumbent_ids = frozenset(i for i in range(n) if not outsiders & (1<<i))
            occupied = {full[i] for i in range(n) if i not in incumbent_ids and full[i]>=0}
            residual = tuple(tuple(s for s in row if s not in occupied) if i in incumbent_ids else ()
                             for i,row in enumerate(pref))
            assert matched(da(residual)) == matched(full) & incumbent_ids
            counts['all_outsider_coalitions'] += 1

        for h in range(n):
            def with_report(p):
                return pref[:h]+(p,)+pref[h+1:]
            baseline = da(with_report(()))
            remaining = matched(baseline)
            label = {-1: frozenset()}
            for j in pref[h]:
                mu = da(with_report((j,)))
                if mu[h] == j:
                    removal = tuple(tuple(s for s in row if s!=j) if i!=h else ()
                                    for i,row in enumerate(pref))
                    residual = matched(da(removal))
                    assert residual == matched(mu)-{h}
                    assert residual <= remaining and len(remaining-residual)<=1
                    label[j] = remaining-residual
                    counts['singleton_capacity_labels_with_displaced_incumbent'] += bool(label[j])
            menu = [j for j in pref[h] if j in label]
            d=len(pref[h])
            for k in range(d+1):
                choice_counts = {j:0 for j in menu+[-1]}
                for subset in combinations(pref[h],k):
                    mu=da(with_report(subset))
                    j=next((s for s in subset if s in label),-1)
                    assert mu[h] == j
                    choice_counts[j] += 1
                    assert matched(mu)-{h} == remaining-label[j]
                    singleton = da(with_report(() if j==-1 else (j,)))
                    if any(mu[i]!=singleton[i] for i in range(n) if i!=h):
                        counts['retentions_with_different_partners']+=1
                    counts['focal_truncated_reports'] += 1
                assert choice_counts[-1]==comb(d-len(menu),k)
                for r,j in enumerate(menu,1):
                    assert choice_counts[j]==(comb(d-r,k-1) if k else 0)
                possible = set(menu[:min(len(menu),d-k+1)]) if k else set()
                if k <= d-len(menu):
                    possible.add(-1)
                assert possible == {j for j,v in choice_counts.items() if v>0}
                counts['exact_quota_winner_sets']+=1
                counts['uniform_quota_cases']+=1
        da.cache_clear()
    return counts


def examples():
    # At quota one the same editing outsider can cost G one place, earn G one
    # place, or exchange the identity of a matched member with no net change.
    pref = [(0,), (1,), (0,1,2)]
    pri = [(2,0),(2,1),(2,)]
    table={}
    for j in range(3):
        p=pref[:2]+[(j,)]
        table[str(j)] = run_da(p,pri)
    assert table=={'0':(-1,1,0),'1':(0,-1,1),'2':(0,1,2)}
    # Common priorities. Available-seat cardinality has both complementarity
    # and substitution, even for two incumbents.
    pref2=((1,0,2),(0,))
    pri2=((0,1),(0,),(0,))
    size = {}
    for stock in range(8):
        p=tuple(tuple(j for j in row if stock & (1<<j)) for row in pref2)
        size[stock]=sum(j>=0 for j in run_da(p,pri2))
    assert size[6]-size[4]==0 < size[7]-size[5]==1
    assert size[1]-size[0]==1 > size[5]-size[4]==0
    return dict(sharp_displacements=table,non_submodular_or_supermodular=size)


def check_saturation():
    """Check every rollout in the interval-neighborhood construction, with quotas."""
    checked, markets = 0, 0
    for m in range(1, 8):
        for r in range(m+1):
            for k in sorted({r+1, max(5,r+1)}):
                # Applicants are in increasing common priority: dedicated, outsiders, incumbents.
                filler_count = m*(k-1)
                n_listings = 2*m + filler_count
                dedicated = [(2*m+q,) for q in range(filler_count)]
                inc = [tuple(range(i,i+r+1)) for i in range(m-r)]
                old, new, full = [], [], []
                for i in range(m):
                    fillers = tuple(2*m+i*(k-1)+z for z in range(k-1))
                    old.append((m+i,)+fillers)
                    new.append((i,)+fillers)
                    full.append((i,m+i)+fillers)
                full_pref = dedicated+full+inc
                priority = [tuple(i for i,row in enumerate(full_pref) if j in row) for j in range(n_listings)]
                for rollout in range(1<<m):
                    visible = dedicated+[new[i] if rollout&(1<<i) else old[i] for i in range(m)]+inc
                    assert all(len(row)==min(k,len(feasible)) for row,feasible in zip(visible,full_pref))
                    mu = run_da(visible,priority)
                    lows = mu[filler_count+m:]
                    if rollout.bit_count()<=r:
                        assert all(x>=0 for x in lows)
                    if rollout==(1<<m)-1:
                        assert all(x<0 for x in lows)
                    checked += 1
                markets+=1
    return dict(m_max=7,distinct_designs=markets,all_rollout_assignments=checked)


def check_encoding():
    """Count distinct report functions in the two information lower bounds."""
    per_size = {}
    for n in range(2, 7):
        background = range(n - 1)
        priority = [tuple(range(n))] * n
        reports = [()] + [(j,) for j in background]
        labeled_functions = set()
        group_functions = set()
        for assignment in permutations(background):
            incumbents = [(j,) for j in assignment]
            responses = tuple(tuple(j >= 0 for j in run_da([row] + incumbents, priority)[1:])
                              for row in reports)
            labeled_functions.add(responses)
        for subset in range(1 << (n - 1)):
            incumbents = [(i,) if subset & (1 << i) else () for i in background]
            counts = tuple(sum(j >= 0 for j in run_da([row] + incumbents, priority)[1:])
                           for row in reports)
            group_functions.add(counts)
        assert len(labeled_functions) == factorial(n - 1)
        assert len(group_functions) == 2 ** (n - 1)
        subset_checks = 0
        for g in range(1, n):
            b = n - 1 - g
            groups = set(range(1 + b, n))
            t = n - 1
            q = n - 1
            hard_functions = set()
            for subset in range(1 << (n - 2)):
                incumbents = []
                for i in range(1, q):
                    site = i - 1
                    present = bool(subset & (1 << site))
                    if i in groups:
                        incumbents.append((site,) if present else ())
                    else:
                        incumbents.append((site, t) if present else (site,))
                incumbents.append((t,))
                counts = []
                for row in [()] + [(j,) for j in range(n - 2)]:
                    mu = run_da([row] + incumbents, priority)
                    counts.append(sum(mu[i] >= 0 for i in groups))
                assert [counts[0] - c for c in counts[1:]] == [
                    int(bool(subset & (1 << j))) for j in range(n - 2)]
                hard_functions.add(tuple(counts))
            assert len(hard_functions) == 2 ** (n - 2)
            subset_checks += len(hard_functions)
        per_size[n] = dict(labeled_response_functions=len(labeled_functions),
                           group_count_response_functions=len(group_functions),
                           group_function_checks_across_nonempty_group_sizes=subset_checks)
    return per_size


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,default=Path(__file__).with_name('evidence.json'))
    args=cli.parse_args()
    evidence={f'{n}x{m}':check(n,m) for n,m in [(2,2),(2,3),(3,2),(3,3)]}
    evidence['sharp_three_listing_example']=examples()
    evidence['saturation_witness']=check_saturation()
    evidence['encoding_lower_bound_witnesses']=check_encoding()
    args.out.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':
    main()
