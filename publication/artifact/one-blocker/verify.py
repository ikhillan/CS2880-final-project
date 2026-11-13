"""Check support equivalence, the one-blocker theorem, and the singleton-cost formula."""
from collections import Counter
from importlib import util
from itertools import product, permutations
from pathlib import Path
import json

helper = Path(__file__).resolve().parents[1] / 'degree-boundary' / 'verify.py'
spec = util.spec_from_file_location('degree_two_checks', helper)
degree_two = util.module_from_spec(spec)
spec.loader.exec_module(degree_two)


def components(edges, n):
    graph = {i: set() for i in range(2*n)}
    for b,s in edges:
        graph[b].add(n+s)
        graph[n+s].add(b)
    unseen = set(graph)
    pieces = []
    while unseen:
        stack = [unseen.pop()]
        found = set(stack)
        while stack:
            for j in graph[stack.pop()] & unseen:
                unseen.remove(j)
                found.add(j)
                stack.append(j)
        proposers = tuple(sorted(i for i in found if i<n))
        if proposers:
            pieces.append(proposers)
    return pieces


def full_stables(bp, sr, edge_bits):
    result=[]
    for bm in product(*[(-1,)+row for row in bp]):
        paired=[s for s in bm if s!=-1]
        if len(paired)!=len(set(paired)):
            continue
        held=[-1]*len(sr)
        mask=0
        for i,s in enumerate(bm):
            if s!=-1:
                held[s]=i
                mask |= edge_bits[i,s]
        if not degree_two.full_blockers(bp,sr,bm,held):
            result.append((bm,mask))
    return result


def check(n=3):
    counts=Counter()
    first_witness=None
    possible=list(product(range(n),repeat=2))
    # Chosen fixed costs introduce ties between plans and asymmetries within pairs.
    costs=(1,3,2,2,1,4)
    subset_costs=[sum(c for i,c in enumerate(costs) if mask&(1<<i)) for mask in range(1<<(2*n))]
    for graph_mask in range(1<<len(possible)):
        edges=[e for i,e in enumerate(possible) if graph_mask & (1<<i)]
        neighbor_b=[tuple(s for b,s in edges if b==i) for i in range(n)]
        neighbor_s=[tuple(b for b,s in edges if s==j) for j in range(n)]
        if max(map(len,neighbor_b+neighbor_s))>2:
            continue
        counts['graphs']+=1
        eb={e:1<<i for i,e in enumerate(edges)}
        full=(1<<len(edges))-1
        opens=[sum(bit for (b,s),bit in eb.items() if mask & ((1<<b)|(1<<(n+s))))
               for mask in range(1<<(2*n))]
        groups=range(1<<n)
        local_components=components(edges,n)
        for bp,sp in product(product(*(permutations(x) for x in neighbor_b)),
                             product(*(permutations(x) for x in neighbor_s))):
            counts['profiles']+=1
            sr=[{b:k for k,b in enumerate(o)} for o in sp]
            stable=full_stables(bp,sr,eb)
            assert stable
            vis=[]
            for mask in range(full+1):
                bm,held=degree_two.assignments(bp,sr,eb,mask)
                ranks=degree_two.ranks(bp,bm)
                blockers=degree_two.full_blockers(bp,sr,bm,held)
                assert (not blockers)==any(m & mask==m for _,m in stable)
                counts['visible_graph_support_equivalence_checks']+=1
                vis.append((bm,ranks,blockers))
            _,best,_=vis[full]
            for initial,(bm0,r0,blocks0) in enumerate(vis):
                if len(blocks0)==1:
                    counts['one_blocker_baselines']+=1
                    counts['one_blocker_imperfect_baselines']+=int(-1 in bm0)
                # Only feasible successful outcomes enter the per-group direct optima.
                costs_by_loss={}
                for aud,(expose,cost) in enumerate(zip(opens,subset_costs)):
                    bm,r,blocks=vis[initial|expose]
                    if blocks: continue
                    # Include only losses beyond the unavoidable-loss floor.
                    loss=sum(1<<b for b in range(n) if best[b]<=r0[b]<r[b])
                    counts['successful_audits_at_one_blocker']+=int(len(blocks0)==1)
                    if len(blocks0)<=1: assert loss==0
                    costs_by_loss[loss]=min(costs_by_loss.get(loss,10**9),cost)
                assert costs_by_loss
                # Also check all visible expansions, whether obtainable by these actions or not.
                if len(blocks0)<=1:
                    for extra in degree_two.submasks(full^initial):
                        bm,r,blocks=vis[initial|extra]
                        if not blocks:
                            assert all(not (best[i]<=r0[i]<r[i]) for i in range(n))
                            counts['successful_expansions_from_at_most_one_blocker']+=1
                options=[]
                for comp in local_components:
                    oriented={tuple(bm[b] for b in comp) for bm,_ in stable}
                    assert 1<=len(oriented)<=2
                    comp_cost=[]
                    for assignment in oriented:
                        loss=cost=0
                        for b,s in zip(comp,assignment):
                            rank=bp[b].index(s) if s!=-1 else len(bp[b])
                            loss|=(1<<b) if best[b]<=r0[b]<rank else 0
                            if s!=-1 and not (initial & eb[b,s]):
                                cost+=min(costs[b],costs[n+s])
                        comp_cost.append((loss,cost))
                    if len(oriented)==2:
                        component_blockers=sum(b in comp for b,s in blocks0)
                        unmatched=sum(bm0[b]==-1 for b in comp)
                        assert component_blockers==2*unmatched
                        counts['flexible_component_blocker_identities']+=1
                    options.append(comp_cost)
                for group in groups:
                    direct=min(c for loss,c in costs_by_loss.items() if not loss&group)
                    formula=sum(min(c for loss,c in alternatives if not loss&group)
                                for alternatives in options)
                    assert direct==formula
                    counts['weighted_singleton_group_optimum_checks']+=1
                    if direct>min(costs_by_loss.values()):
                        counts['positive_group_premium_checks']+=1
                        assert len(blocks0)>=2 and -1 in bm0
                        if first_witness is None and len(blocks0)==2:
                            first_witness=dict(buyer_orders=bp,seller_orders=sp,initial_edges=[e for e,v in eb.items() if initial&v],
                                               initial_response=bm0,initial_blockers=blocks0,protected_group_mask=group,
                                               ordinary=min(costs_by_loss.values()),protected=direct)
    return dict(n_per_side=n,firm_costs=costs,checks=dict(counts),two_blocker_example=first_witness)


if __name__=='__main__':
    print(json.dumps(check(),indent=2))
