"""Exhaust tiny degree-two graphs and check an imperfect-initial-assignment family.

Requires only the Python standard library. An exposed graph is represented by
an integer bit mask on a fixed edge order. The degree-two boundary is checked
for arbitrary graph expansions, including all firm-audit-induced expansions.
"""
import argparse
import json
from collections import Counter, deque
from itertools import permutations, product


def assignments(buyer_lists, seller_ranks, edge_bits, visible):
    assigned = [-1] * len(buyer_lists)
    held = [-1] * len(seller_ranks)
    queue = deque(range(len(buyer_lists)))
    next_index = [0] * len(buyer_lists)
    while queue:
        buyer = queue.popleft()
        while next_index[buyer] < len(buyer_lists[buyer]):
            seller = buyer_lists[buyer][next_index[buyer]]
            next_index[buyer] += 1
            if not (visible & edge_bits[buyer, seller]):
                continue
            incumbent = held[seller]
            if incumbent == -1 or seller_ranks[seller][buyer] < seller_ranks[seller][incumbent]:
                if incumbent != -1:
                    assigned[incumbent] = -1
                    queue.append(incumbent)
                assigned[buyer] = seller
                held[seller] = buyer
                break
    return tuple(assigned), tuple(held)


def full_blockers(buyer_lists, seller_ranks, assigned, held):
    blockers = []
    for buyer, order in enumerate(buyer_lists):
        for seller in order:
            if seller == assigned[buyer]:
                break
            incumbent = held[seller]
            if incumbent == -1 or seller_ranks[seller][buyer] < seller_ranks[seller][incumbent]:
                blockers.append((buyer, seller))
    return blockers


def ranks(buyer_lists, assigned):
    return tuple(order.index(seller) if seller != -1 else len(order)
                 for order, seller in zip(buyer_lists, assigned))


def harm_set(preliminary_ranks, final_ranks):
    return frozenset(i for i, (old, new) in enumerate(zip(preliminary_ranks, final_ranks))
                     if new > old)


def submasks(mask):
    current = mask
    while True:
        yield current
        if current == 0:
            break
        current = (current - 1) & mask


def exhaustive_boundary(n=3):
    possible_edges = list(product(range(n), repeat=2))
    all_counts = Counter()
    regular_counts = Counter()
    for graph_mask in range(1 << len(possible_edges)):
        edges = [edge for index, edge in enumerate(possible_edges) if graph_mask & (1 << index)]
        buyer_neighbors = [tuple(s for b, s in edges if b == i) for i in range(n)]
        seller_neighbors = [tuple(b for b, s in edges if s == j) for j in range(n)]
        degrees = [len(row) for row in buyer_neighbors + seller_neighbors]
        if max(degrees) > 2:
            continue
        graph_total = Counter(graphs=1)
        edge_bits = {edge: 1 << index for index, edge in enumerate(edges)}
        full = (1 << len(edges)) - 1
        buyer_orders = tuple(product(*(permutations(row) for row in buyer_neighbors)))
        seller_orders = tuple(product(*(permutations(row) for row in seller_neighbors)))
        for bp, sp in product(buyer_orders, seller_orders):
            graph_total['strict_profiles'] += 1
            sr = [{b: i for i, b in enumerate(order)} for order in sp]
            visible_outcomes = []
            for visible in range(full + 1):
                bm, sm = assignments(bp, sr, edge_bits, visible)
                stable = not full_blockers(bp, sr, bm, sm)
                visible_outcomes.append((bm, ranks(bp, bm), stable))
                graph_total['da_runs'] += 1
            full_ranks = visible_outcomes[full][1]
            assert visible_outcomes[full][2]
            for initial, (bm0, initial_ranks, initially_stable) in enumerate(visible_outcomes):
                if -1 in bm0:
                    continue
                graph_total['perfect_initial_inputs'] += 1
                floor = harm_set(initial_ranks, full_ranks)
                if not initially_stable:
                    graph_total['perfect_initial_inputs_with_blockers'] += 1
                for extra in submasks(full ^ initial):
                    graph_total['expansions'] += 1
                    _, final_ranks, stable = visible_outcomes[initial | extra]
                    if stable:
                        graph_total['successful_expansions'] += 1
                        assert harm_set(initial_ranks, final_ranks) == floor
                        if floor:
                            graph_total['successful_expansions_with_unavoidable_losses'] += 1
        all_counts.update(graph_total)
        if all(d == 2 for d in degrees):
            regular_counts.update(graph_total)
    return {'n_per_side': n, 'full_degree_at_most_two': dict(all_counts),
            'regular_degree_two_only': dict(regular_counts)}


def imperfect_cycle(n):
    bp = [(i, (i+1) % n) for i in range(n)]
    sp = [((j-1) % n, j) for j in range(n)]
    sr = [{b: i for i, b in enumerate(order)} for order in sp]
    edge_bits = {(i, s): 1 << (2*i+j) for i in range(n) for j, s in enumerate(bp[i])}
    full = (1 << (2*n)) - 1
    initial = edge_bits[0, 0] | sum(edge_bits[i, (i+1) % n] for i in range(n-1))
    bm0, sm0 = assignments(bp, sr, edge_bits, initial)
    assert bm0 == (0,) + tuple(range(2, n)) + (-1,)
    full_bm, _ = assignments(bp, sr, edge_bits, full)
    assert full_bm == tuple(range(n))
    initial_blocks = full_blockers(bp, sr, bm0, sm0)
    incident = [0] * (2*n)
    for (buyer, seller), bit in edge_bits.items():
        incident[buyer] |= bit
        incident[n + seller] |= bit
    visible_outcomes = {}
    ordinary = protected = 2*n + 1
    for audits in range(1 << (2*n)):
        visible = initial
        for firm, bits in enumerate(incident):
            if audits & (1 << firm):
                visible |= bits
        if visible not in visible_outcomes:
            bm, sm = assignments(bp, sr, edge_bits, visible)
            visible_outcomes[visible] = (bm, not full_blockers(bp, sr, bm, sm))
        bm, stable = visible_outcomes[visible]
        if stable:
            ordinary = min(ordinary, audits.bit_count())
            if bm[0] == 0:
                protected = min(protected, audits.bit_count())
    assert (ordinary, protected) == (1, n-1)
    return {'n': n, 'audit_masks': 1 << (2*n),
            'number_of_initial_blockers': len(initial_blocks),
            'distinct_visibility_patterns': len(visible_outcomes),
            'ordinary_cost': ordinary, 'cost_protecting_buyer_0': protected}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-imperfect-n', default=8, type=int)
    options = parser.parse_args()
    results = {'degree_two_boundary': exhaustive_boundary(),
               'imperfect_cycle': [imperfect_cycle(n) for n in range(3, options.max_imperfect_n + 1)]}
    print(json.dumps(results, indent=2))
