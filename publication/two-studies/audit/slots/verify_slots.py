"""Independent exhaustive check on two applicants and two programs with capacities one or two.

Enumerates displays, preferred directions on each side, occupancy assignments, candidate pools
and protected choices. Direct stability and direct enumeration define the comparison oracle.
"""
from collections import Counter
from functools import lru_cache
from itertools import product
import json

import numpy as np

from slot_repair import best_menu, stable_edges


OWNER = np.array([0,0,1,1])
PROGRAM = np.array([0,1,0,1])
MASKS = [np.array([bool(k & (1 << e)) for e in range(4)]) for k in range(16)]


def bits(mask):
    return sum((1 << e) for e in np.flatnonzero(mask))


def quotas(mask):
    return ((mask & 3).bit_count(), (mask >> 2).bit_count())


def direct_stable(match, mask, capacity, rank, priority):
    holders = [[i for i,j in enumerate(match) if j == program] for program in range(2)]
    if any(len(holders[j]) > capacity[j] for j in range(2)):
        return False
    if any(j >= 0 and not (mask & (1 << (2*i+j))) for i,j in enumerate(match)):
        return False
    for i,j in product(range(2),repeat=2):
        if not (mask & (1 << (2*i+j))):
            continue
        now = match[i]
        if now >= 0 and rank[2*i+now] <= rank[2*i+j]:
            continue
        if len(holders[j]) < capacity[j] or any(priority[2*i+j] < priority[2*h+j] for h in holders[j]):
            return False
    return True


SCORES = tuple(np.array(s,dtype=float) for s in [
    (1,-3,0,4), (-2,5,3,0),
    *[tuple(int((2*i+j)==e) for e in range(4)) for i,j in product(range(2),repeat=2)],
])


@lru_cache(None)
def optimized(allowed_pool, required, quota, case):
    mask = best_menu(OWNER, MASKS[allowed_pool], MASKS[required], np.array(quota),
                     SCORES[case],MASKS[0],MASKS[0])
    return bits(mask)


def run():
    counts = Counter()
    value = [[float(s[MASKS[display]].sum()) for display in range(16)] for s in SCORES]
    for r0,r1,p0,p1 in product(range(2),repeat=4):
        rank = np.array([r0,1-r0,r1,1-r1])
        priority = np.array([p0,p1,1-p0,1-p1])
        for cap in product((1,2),repeat=2):
            for match in product((-1,0,1),repeat=2):
                stable = {d for d in range(16) if direct_stable(match,d,cap,rank,priority)}
                if not stable:
                    continue
                old = np.array([-1 if j<0 else 2*i+j for i,j in enumerate(match)])
                contracts = sum(1 << e for e in old if e>=0)
                safe = bits(stable_edges(OWNER,PROGRAM,rank,priority,np.array(cap),old))
                characterized = {d for d in range(16) if d & contracts == contracts and not d & ~safe}
                assert characterized == stable
                counts['stability_statuses_directly_compared'] += 16
                for a in stable:
                    quota = quotas(a)
                    for pool in (u for u in range(16) if (u & a)==a):
                        for right in (h for h in range(16) if (h & a)==h):
                            feasible = {d for d in stable if not d & ~pool and (d & right)==right and quotas(d)==quota}
                            assert a in feasible
                            for c,values in enumerate(value):
                                oracle = max(values[d] for d in feasible)
                                calculated = optimized(safe & pool,contracts | right,quota,c)
                                assert calculated in feasible and oracle == values[calculated]
                                counts['optimal_objectives_directly_compared'] += 1
                            counts['baseline_pool_rights_conditions_checked'] += 1
    print(json.dumps(dict(counts),indent=2))


if __name__=='__main__':
    run()
