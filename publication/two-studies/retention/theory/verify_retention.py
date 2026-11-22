"""Exhaustive checks for the seat-retention formulation and hardness reduction."""
import json
from collections import deque
from itertools import combinations, permutations, product
from pathlib import Path

def da(menus, priority, caps):
    pending = deque(range(len(menus)))
    position = [0] * len(menus)
    held = [[] for _ in priority]
    while pending:
        i = pending.popleft()
        if position[i] == len(menus[i]):
            continue
        j = menus[i][position[i]]
        position[i] += 1
        held[j].append(i)
        if len(held[j]) > caps[j]:
            worst = max(held[j], key=priority[j].index)
            held[j].remove(worst)
            pending.append(worst)
    result = [None] * len(menus)
    for j, winners in enumerate(held):
        for i in winners:
            result[i] = j
    return tuple(result)

def has_no_block(assignment, menus, priority, caps):
    for i, row in enumerate(menus):
        for j in row:
            if assignment[i] == j:
                break
            seats = [a for a, partner in enumerate(assignment) if partner == j]
            if len(seats) < caps[j] or any(priority[j].index(i) < priority[j].index(a) for a in seats):
                return False
    return True

def witness_inequalities(assignment, menus, priority, caps):
    for i, row in enumerate(menus):
        for k, j in enumerate(row):
            # sum of assignment indicators above j, including j, is zero or one.
            weakly_better = int(assignment[i] in row[:k+1])
            better_priority = sum(assignment[a] == j for a in priority[j][:priority[j].index(i)])
            if caps[j] * weakly_better + better_priority < caps[j]:
                return False
    return True

def check_characterization():
    people = tuple(range(3))
    orders = tuple(permutations(people))
    menus = [()] + [p for length in (1, 2) for p in permutations(range(2), length)]
    markets = stable_witnesses = feasible_witnesses = protected_sets = 0
    for caps in ((1,1),(1,2)):
        for rows in product(menus, repeat=len(people)):
            candidates = [state for state in product(*[(None,)+row for row in rows])
                          if all(state.count(j) <= caps[j] for j in range(2))]
            for priority in product(orders,repeat=2):
                predicted = da(rows,priority,caps)
                sets = set()
                for assignment in candidates:
                    feasible_witnesses += 1
                    stable = has_no_block(assignment, rows, priority, caps)
                    assert stable == witness_inequalities(assignment, rows, priority, caps)
                    if stable:
                        stable_witnesses += 1
                        sets.add(tuple(s is not None for s in assignment))
                recipients = tuple(s is not None for s in predicted)
                assert sets == {recipients}
                for protected in product((False,True), repeat=len(people)):
                    predicted_safe = all(not r or s for r,s in zip(protected,recipients))
                    exists = any(all(not r or s for r,s in zip(protected,other)) for other in sets)
                    assert exists == predicted_safe
                    protected_sets += 1
                markets += 1
    return dict(markets=markets,feasible_assignments=feasible_witnesses,
                stable_assignments=stable_witnesses,protected_set_checks=protected_sets)

def independent_set_gadget(n):
    # Four pupils and six private programs per vertex; z is the unique shared program.
    z = 6*n
    p, x, b, q, r, s = ([6*i+k for i in range(n)] for k in range(6))
    t, g, h, d = ([4*i+k for i in range(n)] for k in range(4))
    caps = (1,)*(z+1)
    full = []
    for i in range(n):
        full.extend([(p[i], x[i], b[i]), (p[i], q[i]), (x[i], z, r[i]), (r[i], s[i])])
    def common(order):
        assert sorted(order)==list(range(4*n))
        return [order]*(z+1)
    reference = common(g+t+h+d)
    labels = da(full,reference,caps)
    assert None not in labels and [labels[i] for i in t] == x

    def menus(selected):
        rows = full.copy()
        for i in range(n):
            rows[t[i]] = (p[i], x[i] if i in selected else b[i])
        return rows

    def score(rows):
        return sum(actual in row for actual,row in zip(labels,rows))

    edges = list(combinations(range(n),2))
    edge_rows = {}
    endpoint_checks = 0
    for edge in edges:
        order = d.copy()  # d holds r against any h throughout this design draw.
        for i in range(n):
            order += [g[i],t[i]] if i in edge else [t[i],g[i]]
        order += h
        priority = common(order)
        base = da(menus(set()),priority,caps)
        assert None not in base
        for zmask in range(1<<n):
            selected = {i for i in range(n) if zmask & (1<<i)}
            rows = menus(selected)
            final = da(rows,priority,caps)
            losses = set(i for i,j in enumerate(final) if j is None)
            assert len(losses) == int(set(edge)<=selected) and losses <= {h[edge[0]],h[edge[1]]}
            assert score(rows)-score(menus(set())) == len(selected)
            assert None not in da(rows,reference,caps)
            edge_rows[edge,zmask] = not losses
            endpoint_checks+=1

    graph_menu_checks = 0
    for graph_mask in range(1<<len(edges)):
        graph=[e for k,e in enumerate(edges) if graph_mask & (1<<k)]
        for zmask in range(1<<n):
            safe = all(edge_rows[e,zmask] for e in graph)
            independent = all(not (zmask & (1<<a) and zmask & (1<<b)) for a,b in graph)
            assert safe == independent
            graph_menu_checks+=1
    return dict(vertices=n,draw_menu_pairs=endpoint_checks,graph_menu_pairs=graph_menu_checks)

def check_examples():
    common = [(0,1),(0,1)]
    witness = [(1,),(0,)] # a and b switch schools, both stay placed.
    assert da([(0,),(1,)],common,(1,1)) == (0,1)
    assert da(witness,common,(1,1)) == (1,0)
    # A feasible reassignment that protects all incumbents can be unstable.
    rows = [(0,1),(0,)]
    assert da([(1,),(0,)],common,(1,1)) == (1,0)
    assert not has_no_block((1,0),rows,common,(1,1))
    assert da(rows,common,(1,1)) == (0,None)
    # Equal headcounts can conceal an incumbent's exit.
    before = da([(),(0,)],[(0,1)],(1,))
    after = da([(0,),(0,)],[(0,1)],(1,))
    assert before==(None,0) and after==(0,None)

if __name__ == "__main__":
    check_examples()
    summary = {"stable_witness":check_characterization(),
               "historical_DA_gadget":[independent_set_gadget(n) for n in range(2,6)]}
    print(json.dumps(summary, indent=2))
