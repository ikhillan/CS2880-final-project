"""Stress-test the priority veto for incumbents compelled to stay at a program."""
from itertools import product
from random import Random

def clear(menus, orders):
    n = len(menus)
    held = [None] * len(orders)
    index = [0]*n
    pool = list(range(n))
    while pool:
        i=pool.pop()
        if index[i]==len(menus[i]):continue
        c=menus[i][index[i]];index[i]+=1
        j=held[c]
        if j is None or orders[c].index(i)<orders[c].index(j):
            held[c]=i
            if j is not None:pool.append(j)
        else:pool.append(i)
    assigned=[None]*n
    for c,i in enumerate(held):
        if i is not None:assigned[i]=c
    return assigned

def main():
    rng=Random(13895)
    veto_examples=0
    for _ in range(1800):
        n=4
        courses=[0,0,1,2] if rng.randrange(2) else [0,1,2]
        priority=[rng.sample(range(n),n) for _ in courses]
        prefs=[rng.sample(range(3),3) for _ in range(n)]
        fixed=[rng.sample([0,1,2],rng.randint(0,2)) for _ in range(n)]
        alt=[rng.sample([0,1,2],rng.randint(0,2)) for _ in range(n)]
        menus=[[[j for j in prefs[i] if j in arm[i]] for i in range(n)] for arm in [fixed,alt]]
        def contracts(rows):
            return [[c for j in row for c,school in enumerate(courses) if school==j] for row in rows]
        base=clear(contracts(menus[0]),priority)
        incumbent=[i for i,j in enumerate(base) if j is not None]
        if not incumbent:continue
        legal=[set(menus[0][i]+menus[1][i]) for i in incumbent]
        complete=[ys for ys in product(*legal) if all(ys.count(c)<=courses.count(c) for c in range(3))]
        assert complete
        supported={i:set(s[k] for s in complete) for k,i in enumerate(incumbent)}
        compulsory={j:[i for i in incumbent if supported[i]=={j}] for j in range(3)}
        for pattern in product((0,1),repeat=n):
            rows=[menus[pattern[i]][i] for i in range(n)]
            assigns=clear(contracts(rows),priority)
            all_safe=all(assigns[i] is not None for i in incumbent)
            for i in range(n):
                safe=supported.get(i, set(j for j in set(rows[i]) if any(ys.count(j)<courses.count(j) for ys in complete)))
                first_safe=next((k for k,j in enumerate(rows[i]) if j in safe),None)
                if first_safe is None and i in incumbent:
                    assert not all_safe
                    continue
                if first_safe is None:
                    first_safe=len(rows[i])
                for j in rows[i][:first_safe+1]:
                    seats=[k for k,s in enumerate(courses) if s==j]
                    holders=compulsory[j]
                    if i in holders or len(seats)!=len(holders):continue
                    if any(all(priority[k].index(i)<priority[k].index(h) for k in seats) for h in holders):
                        veto_examples+=1
                        assert not all_safe
                        break
    assert veto_examples>0
    print(f"Checked 1800 generated strict markets and all 16 binary-menu profiles each; {veto_examples} challenger-row vetoes.")

if __name__=="__main__": main()
