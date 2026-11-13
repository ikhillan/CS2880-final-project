"""Exact effects of one-person and joint recommendation changes with fixed applicants and orders."""
import argparse
from collections import Counter
from itertools import combinations, product
import json
import math
from pathlib import Path


def da(lists, priority):
    pos={person:i for i,person in enumerate(priority)}
    held={}; queue=list(lists); next_option=dict.fromkeys(queue,0)
    while queue:
        person=queue.pop()
        idx=next_option[person]
        if idx==len(lists[person]):
            continue
        target=lists[person][idx];next_option[person]+=1
        old=held.get(target)
        if old is None or pos[person]<pos[old]:
            held[target]=person
            if old is not None: queue.append(old)
        else:
            queue.append(person)
    return held


def evaluate(group_sizes,quota):
    m=len(group_sizes)
    core=[f'x{i+1}' for i in range(m)]+['z']
    lower=[f'L_{s}' for s in core]
    initial={f'H{i+1}_{j+1}':core[i] for i,n in enumerate(group_sizes) for j in range(n)}
    higher=list(initial)
    extra=[f'd{i+1}' for i in range(quota)] if quota>1 else []
    aux=[f'P_{d}' for d in extra]
    priority=aux+higher+lower
    # Every listing follows this same priority restricted to its feasible applicants.
    # The first k-1 dummies are always outside the unpadded core, and mutually top dedicated pairs never move.
    fixed_extra=extra[:quota-1]
    base={u:[s]+fixed_extra for u,s in zip(lower,core)}
    base.update({p:[d]+[t for t in extra if t!=d] for p,d in zip(aux,extra)})
    sizes=[]
    exact=[]
    for k in range(len(higher)+1):
        for group in combinations(higher,k):
            change=set(group)
            lists={**base,**{p:['z' if p in change else initial[p]]+fixed_extra for p in higher}}
            assert all(len(row)==quota and len(set(row))==quota for row in lists.values())
            # Core high ranking: x_i first, z second, fixed extras below both.
            # A recommendation switch reveals z instead of x_i; the rankings themselves are identical.
            winners=da(lists,priority)
            assert all(winners[d]==p for d,p in zip(extra,aux))
            actual=sum(p in lower for p in winners.values())
            freed=sum(all(p in change for p,s in initial.items() if s==target) for target in core[:-1])
            predicted=freed+int(not change)
            assert actual==predicted
            sizes.append([len(group),actual]);exact.append({'switched':sorted(change),'low_matches':actual,'delta':actual-1})
    singles=[row['delta'] for row in exact if len(row['switched'])==1]
    joint=exact[-1]['delta']
    h=len(higher)
    for p in (0.,.125,.5,.875,1.):
        by_subsets=sum(row['low_matches']*p**len(row['switched'])*(1-p)**(h-len(row['switched'])) for row in exact)
        by_threats=sum(p**d for d in group_sizes)+(1-p)**h
        assert math.isclose(by_subsets,by_threats, abs_tol=1e-12)
    for j in range(h+1):
        seen=[row['low_matches'] for row in exact if len(row['switched'])==j]
        predicted=sum(math.comb(h-d,j-d)/math.comb(h,j) for d in group_sizes if j>=d)+int(j==0)
        assert math.isclose(sum(seen)/len(seen),predicted, abs_tol=1e-12)
    return {'group_sizes':group_sizes,'quota':quota,'lower_initial':exact[0]['low_matches'],'singleton_effects':singles,
            'joint_effect':joint,'interaction_residual':joint-sum(singles),'coalition_counts_by_size':dict(sorted(Counter(k for k,v in sizes).items())),
            'all_coalitions':exact}


def local_residual(removed, added):
    if removed == 0:
        return max(0, added - 1)
    if added == 0:
        return int(removed >= 2)
    return -int(removed == 1)


def cyclic_lower_example(h):
    starts = list(range(h))
    ends = [(j+1)%h for j in starts]
    def matches(movers):
        occupied = {ends[i] if i in movers else starts[i] for i in starts}
        return h-len(occupied)
    baseline=matches(set())
    singles=[matches({i})-baseline for i in starts]
    joint=matches(set(starts))-baseline
    assert singles==[1]*h and joint==0
    return {'h':h,'lower':joint-sum(singles),'upper':h-1+h//2}


def verify_threat_identity():
    # Endorse each listing by an arbitrary stationary low and allow arbitrary listing priorities.
    import random
    rng=random.Random(541997)
    checked=0
    for h in range(2,10):
        for n in range(1,9):
            for _ in range(100):
                starts=[rng.randrange(n) for _ in range(h)]
                ends=[rng.randrange(n) for _ in range(h)]
                threats=[set(i for i in range(h) if rng.randrange(2)) for _ in range(n)]
                before=[{i for i in threats[s] if starts[i]==s} for s in range(n)]
                after=[{i for i in threats[s] if ends[i]==s} for s in range(n)]
                levels=[]
                for switched in [set()]+[{i} for i in range(h)]+[set(range(h))]:
                    levels.append(sum(not ((x-switched)|(y&switched)) for x,y in zip(before,after)))
                exact=(levels[-1]-levels[0])-sum(l-levels[0] for l in levels[1:-1])
                by_listing=sum(0 if a & b else local_residual(len(a),len(b)) for a,b in zip(before,after))
                assert -h <= exact == by_listing <= h-1+h//2
                checked+=1
    return checked


def exhaustive_two_switches():
    # Scores determined by the high outranker at a listing can be encoded by independent ordered thresholds to the
    # best incumbent low. Enumerate every old/new listing for two high applicants and every background,
    # after stationary applicants have been collapsed. Each of at most four involved listings is always
    # unoccupied by low, or is initially eligible for low with a specified subset of high applicants outranking its low.
    slots=range(4);cases=0
    configurations=list(product(range(5),repeat=4))  # label 0 = always zero; 1+mask = admissible with mask of threats
    for a,b,c,d in product(slots,repeat=4):
        if a==b or c==d: continue
        # a->b and c->d may have coincident endpoints.
        for config in configurations:
            occ=[]
            for i,j in [(a,c),(b,c),(a,d),(b,d)]:
                n=0
                for s,t in enumerate(config):
                    threats=(int(i==s)+2*int(j==s))
                    n+=int(t>0 and not ((t-1)&threats))
                occ.append(n)
            solo=(occ[1]-occ[0])+(occ[2]-occ[0])
            joint=occ[3]-occ[0]
            assert not (solo<0 and joint>0), (a,b,c,d,config,occ)
            cases+=1
    return cases


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    checks=[evaluate(groups,quota) for groups in [[2,1],[2,2],[2,2,2],[2]*5] for quota in [1,5]]
    compact=[{k:v for k,v in row.items() if k!='all_coalitions'} for row in checks]
    bounds=[]
    for h in range(2,11):
        upper=evaluate([2]*(h//2) + ([1] if h%2 else []),1)['interaction_residual']
        bound=cyclic_lower_example(h)
        assert upper==bound['upper']
        bounds.append(bound)
    report={'interaction_extrema':bounds,'random_threat_identity_tests':verify_threat_identity(),'three_switch_full':checks[0], 'four_switch_full':checks[2],
            'family_verification':compact,'two_switch_local_configurations':exhaustive_two_switches()}
    args.out.write_text(json.dumps(report,indent=2)+'\n')
