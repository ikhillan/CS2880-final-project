"""Audit the sparse construction and two valid lower bounds using the archived solver."""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--artifact', type=Path,
                    default=Path(__file__).resolve().parent.parent/'hidden-blocking-pairs/artifact')
args = parser.parse_args()
sys.path.insert(0, str(args.artifact/'methodology'))
from audit_solver import Market, evaluate, _Master


def ring(n, padding):
    assert n >= 3 and n % 2 and padding in (0, 4)
    bp, sp, visible = [], [], set()
    for i in range(n):
        own, next_one, next_two = i, (i+1)%n, (i+2)%n
        extras = tuple(range(n+padding*i, n+padding*(i+1)))
        bp.append(((next_two, next_one, own) if i == 0 else
                   (next_two, own, next_one)) + extras)
        sp.append(((i-1)%n, i, (i-2)%n))
        visible.add((i, own))
        if i:
            visible.add((i, next_one))
        visible.update((i,z) for z in extras[:4 if i==0 else 3])
    for z in range(n, n*(1+padding)):
        bp.append((z,))
        sp.append((z, (z-n)//padding))
        visible.add((z,z))
    side=len(bp)
    return Market(bp,sp,range(side),range(side,2*side),[1]*(2*side),visible)


def ranks(market, out):
    return tuple(order.get(partner,len(order)) for partner,order in
                 zip(out.buyer_match,market.buyer_rank))


def ring_check(n, padding):
    m=ring(n,padding)
    baseline, blocks=evaluate(m,0)
    assert baseline.buyer_match==tuple(range(m.n_buyers))
    assert [m.edges[k] for k in blocks]==[(0,1)]
    full, blocks=evaluate(m,(1 << m.n_firms)-1)
    assert not blocks and all(m.buyer_prefs[i][0]==s for i,s in enumerate(full.buyer_match))
    # Firms of auxiliary agents only open offers to permanently occupied auxiliary listings.
    core_firms=list(range(n))+[m.n_buyers+j for j in range(n)]
    if padding:
        assert all(sum((i,s) in m.initial_edges for s in row)==min(5,len(row))
                   for i,row in enumerate(m.buyer_prefs))
    old=ranks(m,baseline)
    best_stable=best_protected=m.n_firms+1
    for code in range(1 << len(core_firms)):
        mask=sum(1 << firm for j,firm in enumerate(core_firms) if code & (1 << j))
        out,blocks=evaluate(m,mask)
        if blocks:
            continue
        cost=code.bit_count()
        best_stable=min(best_stable,cost)
        if all(x<=y for x,y in zip(ranks(m,out),old)):
            best_protected=min(best_protected,cost)
    assert best_stable==1 and best_protected==(n+1)//2
    plain,_=support_bound(m,{})
    protected,_=support_bound(m,dict(enumerate(baseline.buyer_match)))
    assert (plain,protected)==(best_stable,best_protected)
    return dict(n=n,padding=padding,initial_blockers=1,masks=1 << (2*n),
                ordinary=best_stable,protected=best_protected,
                plain_support=plain,protected_support=protected)


def support_bound(market,thresholds):
    master=_Master(market,stable_support=True)
    for buyer,partner in thresholds.items():
        good=market.buyer_prefs[buyer][:market.buyer_rank[buyer][partner]+1]
        row={master.p+master.m+market.edge_index[buyer,seller]:1 for seller in good}
        master.add_row(row,1,1)
    result=master.solve(60,sum(market.costs))
    assert result.status==0 and abs(result.fun-result.mip_dual_bound)<1e-6
    mask=sum(1 << i for i,v in enumerate(result.x[:market.n_firms]) if v>0.5)
    assert abs(market.cost(mask)-result.fun)<1e-6
    out,blocks=evaluate(market,mask)
    rr=ranks(market,out)
    assert all(rr[b]<=market.buyer_rank[b][s] for b,s in thresholds.items())
    return market.cost(mask), bool(blocks)


def core_check(root):
    total=Counter()
    rows=[]
    for path in sorted((root/'experiments/core/instances').glob('*.json')):
        d=json.loads(path.read_text())
        m=Market.from_dict(d['market'])
        initial,_=evaluate(m,0)
        full,blocks=evaluate(m,(1 << m.n_firms)-1)
        assert not blocks
        old=ranks(m,initial)
        strict={b:s for b,s in enumerate(initial.buyer_match)
                if s!=-1 and ranks(m,full)[b]<=old[b]}
        ordinary_opt=next(r['cost'] for r in d['results'] if r['method']=='enumeration')
        cheapest=d['optimal_outcome_sensitivity']['masks']
        def protected(mask):
            out,bad=evaluate(m,mask)
            rr=ranks(m,out)
            return not bad and all(rr[b]<=old[b] for b in strict)
        if any(protected(mask) for mask in cheapest):
            target=ordinary_opt
        else:
            target=next(m.cost(mask) for mask in sorted(range(1 << m.n_firms),key=m.cost)
                        if m.cost(mask)>ordinary_opt and protected(mask))
        plain,_=support_bound(m,{})
        stronger,has_blocker=support_bound(m,strict)
        assert plain<=stronger<=target
        total['markets']+=1
        total['plain_bound_equals_protected_optimum']+=int(plain==target)
        total['protected_bound_equals_protected_optimum']+=int(stronger==target)
        total['strict_improvement_in_bound']+=int(stronger>plain)
        total['protected_master_candidate_blocks']+=int(has_blocker)
        total['positive_premium']+=int(target>ordinary_opt)
        rows.append(dict(market=m.name,ordinary=ordinary_opt,protected=target,
                         plain_support=plain,protected_support=stronger,
                         protected_master_candidate_has_blocker=has_blocker))
    assert total['markets']==160 and total['positive_premium']==6
    return dict(counts=dict(total),instances=rows)


if __name__=='__main__':
    evidence={'ring':[ring_check(n,d) for n in [3,5,7] for d in [0,4]],
              'core':core_check(args.artifact)}
    print(json.dumps(evidence,indent=2))
