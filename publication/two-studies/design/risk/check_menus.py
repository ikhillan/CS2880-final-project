"""Enumerate the entire menu space and check the maximum-overlap design."""
from collections import Counter, deque
from itertools import combinations, product
import argparse
import json
import random

def clear(prefs, menus, priorities, caps):
    held = [[] for _ in caps]
    queues = [deque(j for j in row if j in menu) for row,menu in zip(prefs,menus)]
    applicants = deque(range(len(menus)))
    while applicants:
        i = applicants.popleft()
        if not queues[i]:
            continue
        j = queues[i].popleft()
        held[j].append(i)
        held[j].sort(key=priorities[j].index)
        if len(held[j]) > caps[j]:
            applicants.append(held[j].pop())
    result = [None] * len(prefs)
    for j, occupants in enumerate(held):
        for i in occupants:
            result[i] = j
    return result

def blockers(i, baseline, prefs, prio, caps):
    partner = baseline[i]
    preferred = prefs[i] if partner is None else prefs[i][:prefs[i].index(partner)]
    out = set()
    for j in preferred:
        holders = [h for h,p in enumerate(baseline) if p == j]
        if len(holders)<caps[j] or any(prio[j].index(i)<prio[j].index(h) for h in holders):
            out.add(j)
    return out

def run(markets,seed):
    rng = random.Random(seed)
    counts = Counter()
    for rep in range(markets):
        n, m, draws = 3, 3, 3
        prefs = [rng.sample(range(m),m) for _ in range(n)]
        prios = [[rng.sample(range(n),n) for _ in range(m)] for _ in range(draws)]
        caps = [rng.choice([0,1,1,2]) for _ in range(m)]
        quotas = [1+((i+rep)%2) for i in range(n)]
        rows = [[set(z) for z in combinations(range(m),q)] for q in quotas]
        a,b = [[rng.choice(r) for r in rows] for _ in range(2)]
        old = [clear(prefs,a,p,caps) for p in prios]
        bad = [[blockers(i,baseline,prefs,p,caps) for i in range(n)] for baseline,p in zip(old,prios)]
        exact = [0]*draws
        for menu in product(*rows):
            if any(not (base & target) <= d for d,base,target in zip(menu,a,b)):
                continue
            failures = 0
            for baseline,blocked,p in zip(old,bad,prios):
                success = all((j is None or j in menu[i]) and not blocked[i]&menu[i] for i,j in enumerate(baseline))
                if success:
                    after = clear(prefs,menu,p,caps)
                    assert all((j is None)==(k is None) and (j is None or prefs[i].index(k)<=prefs[i].index(j)) for i,(j,k) in enumerate(zip(baseline,after)))
                failures += not success
                counts['menu_scenario_pairs'] += 1
            for budget in range(failures,draws):
                exact[budget] = max(exact[budget],sum(len((d&target)-base) for d,target,base in zip(menu,b,a)))
        for budget,answer in enumerate(exact):
            objectives = []
            for good in combinations(range(draws),draws-budget):
                score = 0
                for i,(base,target) in enumerate(zip(a,b)):
                    keep = {old[s][i] for s in good} & (base-target)
                    omit = set().union(*(bad[s][i] for s in good)) & (target-base)
                    score += len(target-base) - max(len(keep),len(omit))
                objectives.append(score)
            assert answer == max(objectives)
            counts['exact_optima'] += 1
        counts['market_configurations'] += 1
    print(json.dumps(counts,sort_keys=True))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets",type=int,default=8000)
    parser.add_argument("--seed",type=int,default=711853)
    args=parser.parse_args()
    run(args.markets,args.seed)
