"""Independent small-market check of incumbent retention with seat-type preferences."""
from itertools import combinations, permutations, product
import random


def clear(reports, menu, routes, priorities, capacity):
    choices = [[(j,t) for j in report if j in menu[i] for t in routes[i,j]]
               for i,report in enumerate(reports)]
    next_choice = [0]*len(reports)
    held = {s:[] for s in capacity}
    assigned = [None]*len(reports)
    while True:
        proposals = {s:list(incumbents) for s,incumbents in held.items()}
        made_offer = False
        for i,options in enumerate(choices):
            if assigned[i] is None and next_choice[i] < len(options):
                proposals[options[next_choice[i]]].append(i)
                next_choice[i] += 1
                made_offer = True
        if not made_offer:
            return assigned
        assigned=[None]*len(reports)
        for slot,applicants in proposals.items():
            held[slot]=sorted(applicants,key=priorities[slot].index)[:capacity[slot]]
            for i in held[slot]:
                assigned[i]=slot


def eligible(i, added, old, reports, routes, priorities, capacity, guaranteed):
    if added in guaranteed:
        return True
    sequence = [(j,t) for j in reports[i] for t in routes[i,j]]
    floor = len(sequence) if old[i] is None else sequence.index(old[i])
    for slot in sequence[:floor]:
        if slot[0] != added:
            continue
        pool=[p for p,dest in enumerate(old) if dest==slot]
        if len(pool)<capacity[slot]:
            return False
        if any(priorities[slot].index(i)<priorities[slot].index(q) for q in pool):
            return False
    return True


def test(repetitions=2000):
    rng=random.Random(166039)
    students=range(4);schools=range(3);types=range(2)
    orders=list(permutations(schools))
    counts=dict(markets=0,menus=0,checked_baseline_recipients=0,school_switches=0,
                category_only=0,old_winners_placed_at_worse_course=0)
    for _ in range(repetitions):
        reports=[rng.choice(orders) for _ in students]
        a=[set(rng.sample(list(schools),2)) for _ in students]
        b=[set(rng.sample(list(schools),2)) for _ in students]
        routes={(i,j):rng.sample(list(types),2) for i in students for j in schools}
        priorities={(j,t):rng.sample(list(students),4) for j in schools for t in types}
        capacity={(j,t):rng.choice((0,0,1,1,2,2,3)) for j in schools for t in types}
        guaranteed = {j for j in schools if sum(capacity[j,t] for t in types)>=sum(j in s|v for s,v in zip(a,b))}
        old=clear(reports,a,routes,priorities,capacity)
        options=[]
        for i in students:
            mandatory=a[i]&b[i]
            if old[i]:mandatory.add(old[i][0])
            offered=a[i]|{j for j in b[i]-a[i] if eligible(i,j,old,reports,routes,priorities,capacity,guaranteed)}
            options.append([set(v) for v in combinations(sorted(offered),len(a[i])) if mandatory<=set(v)])
        for menus in product(*options):
            after=clear(reports,menus,routes,priorities,capacity)
            assert all(old[i] is None or after[i] is not None for i in students)
            for i in students:
                if old[i] is None:continue
                counts['checked_baseline_recipients']+=1
                p,q=old[i],after[i]
                counts['school_switches']+=p[0]!=q[0]
                counts['category_only']+=p!=q and p[0]==q[0]
                counts['old_winners_placed_at_worse_course']+=reports[i].index(q[0])>reports[i].index(p[0])
            counts['menus']+=1
        counts['markets']+=1
    assert counts['school_switches'] and counts['category_only']
    print(counts)


if __name__=='__main__':
    test()
