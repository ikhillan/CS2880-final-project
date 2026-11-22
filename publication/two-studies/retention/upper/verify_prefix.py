"""Exhaustively check relaxed cutoffs with two seat categories and applicant demand changes."""
from collections import deque
from itertools import permutations, product

def clear(rows,schools):
    free=deque(range(len(rows)))
    nxt=[0]*len(rows)
    winners=[None]*len(schools)
    while free:
        i=free.popleft()
        if nxt[i]==len(rows[i]):continue
        seat=rows[i][nxt[i]]
        nxt[i]+=1
        incumbent=winners[seat]
        if incumbent is None or schools[seat].index(i)<schools[seat].index(incumbent):
            winners[seat]=i
            if incumbent is not None:free.append(incumbent)
        else:free.append(i)
    return winners

def verify():
    # Course X has 2 unit-capacity seat categories, Y has one. Distinct
    # pupils can order X categories differently, some omit either course.
    full=[(0,1,2),(2,1,0),(1,0,2)]
    alt=[(2,), (1,0), (2,)]
    priorities=list(permutations(range(3)))
    checks=0
    # For every priority triple, consider candidate contraction of each row
    # from its prefix to empty, to one contract, or to the full base row.
    for order in product(priorities,repeat=3):
        for pattern in product(range(4),repeat=3):
            low=[r[:k] for r,k in zip(full,pattern)]
            old=clear(low,order)
            modes=[([full[i],alt[i]] if not low[i] else [full[i],low[i]]) for i in range(3)]
            for hi in product(*modes):
                new=clear(hi,order)
                for j,k in enumerate(old):
                    if k is None:continue
                    assert new[j] is not None and order[j].index(new[j])<=order[j].index(k)
                checks+=1
    print(f"{checks} typed-seat prefix comparisons passed")

if __name__=="__main__":
    verify()
