"""Exhaustive unit test of the strong first-choice rejection certificate.

Two typed seats, four applicants, three guaranteed first-choice competitors,
arbitrary strict orders at each seat and applicant-specific seat routes. Enumerate
which additional competitors enter, then check that every certified unavailable
course is indeed unattainable. A second course with slack fills deferred pupils.
"""
from itertools import permutations, product

def main():
    n=4
    pupils=range(n)
    orders=[dict(zip(p,range(n))) for p in permutations(pupils)]
    checks=certificates=0
    for o0,o1 in product(orders,repeat=2):
        score=(o0,o1)
        dominating={j for j in (1,2,3) if o0[j]<o0[0] and o1[j]<o1[0]}
        for first_mask in range(8):
            first={j for j in (1,2,3) if first_mask & (1<<(j-1))}
            if len(first & dominating)<2:
                continue
            certificates+=1
            for route in product((0,1),repeat=n):
                for additional in range(8):
                    active=first|{j for j in (1,2,3) if additional & (1<<(j-1))}
                    # DA here has capacity one at both categories and everyone has both.
                    held=[None]*2
                    queue=[0,*sorted(active)]
                    cursor=[0]*n
                    while queue:
                        i=queue.pop()
                        if cursor[i]==2:continue
                        cat=route[i] ^ cursor[i]
                        cursor[i]+=1
                        old=held[cat]
                        if old is None or score[cat][i]<score[cat][old]:
                            held[cat]=i
                            if old is not None:queue.append(old)
                        else:queue.append(i)
                    assert 0 not in held
                    checks+=1
    print(f"Verified {certificates} distinct typed-priority/guaranteed-list certificates in {checks} DA clears.")

if __name__=="__main__":main()
