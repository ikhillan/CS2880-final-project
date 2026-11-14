"""Reconstruct the reduction as a matching market and exhaust all labeled graphs up to n vertices."""
from argparse import ArgumentParser
from collections import deque
from itertools import combinations
import json
from math import comb


def deferred_acceptance(menus, priority):
    """Unit-seat, applicant-proposing DA. A smaller priority means earlier in the common order."""
    assigned = [None] * len(menus)
    held = {}
    pending = deque(range(len(menus)))
    next_option = [0] * len(menus)
    while pending:
        person = pending.popleft()
        index = next_option[person]
        if index == len(menus[person]):
            continue
        school = menus[person][index]
        next_option[person] += 1
        holder = held.get(school)
        if holder is None or priority[person] < priority[holder]:
            if holder is not None:
                assigned[holder] = None
                pending.append(holder)
            held[school] = person
            assigned[person] = school
        else:
            pending.append(person)
    return assigned


def stable(assignment, menus, priority):
    """Check feasibility and blocking against all displayed, preferred schools."""
    holder = {school: i for i, school in enumerate(assignment) if school is not None}
    if len(holder) != sum(school is not None for school in assignment):
        return False
    for person, options in enumerate(menus):
        partner = assignment[person]
        if partner is not None and partner not in options:
            return False
        for school in options:
            if school == partner:
                break
            if school not in holder or priority[person] < priority[holder[school]]:
                return False
    return True


def instance(edges, n_vertices):
    baseline = []
    for e in range(len(edges)):
        x, y, z = 3 * e, 3 * e + 1, 3 * e + 2
        baseline.extend([(x, z), (y,)])
    priorities = []
    for vertex in range(n_vertices):
        order = []
        for e, endpoints in enumerate(edges):
            i, h = 2 * e, 2 * e + 1
            order.extend((i, h) if vertex in endpoints else (h, i))
        priorities.append({person: rank for rank, person in enumerate(order)})
    return baseline, priorities


def check_graph(n, edges):
    baseline, priorities = instance(edges, n)
    old = [deferred_acceptance(baseline, order) for order in priorities]
    assert all(assignment == old[0] and None not in assignment for assignment in old)
    best_exact_failures = [-1] * (n + 1)
    for bits in range(2 ** len(edges)):
        menus = baseline.copy()
        exposed = set()
        score = bits.bit_count()
        for e, endpoints in enumerate(edges):
            if (bits >> e) & 1:
                menus[2 * e] = (3 * e + 1, 3 * e)
                exposed.update(endpoints)
        failed = set()
        for draw, order in enumerate(priorities):
            assigned = deferred_acceptance(menus, order)
            witnesses = stable(old[draw], menus, order)
            keeps_people = all(after is not None for before, after in zip(old[draw], assigned)
                               if before is not None)
            assert stable(assigned, menus, order)
            assert witnesses == keeps_people
            if not witnesses:
                failed.add(draw)
        assert failed == exposed
        failures = len(failed)
        best_exact_failures[failures] = max(best_exact_failures[failures], score)
    for b in range(n + 1):
        best_market = max(best_exact_failures[:b + 1])
        best_graph = max(sum(set(e) <= set(subset) for e in edges)
                         for subset in combinations(range(n), b))
        assert best_market == best_graph
        if b >= 2:
            has_clique = any(all(pair in edges for pair in combinations(subset, 2))
                             for subset in combinations(range(n), b))
            assert (best_market >= comb(b, 2)) == has_clique


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--max-vertices", type=int, default=5)
    args = parser.parse_args()
    if args.max_vertices < 2:
        parser.error("at least two vertices required")
    totals = {"labeled_graphs": 0, "graph_budget_pairs": 0, "candidate_displays": 0,
              "candidate_display_priority_scenarios": 0}
    for n in range(2, args.max_vertices + 1):
        complete_edges = tuple(combinations(range(n), 2))
        for bits in range(2 ** len(complete_edges)):
            edges = tuple(e for index, e in enumerate(complete_edges) if (bits >> index) & 1)
            check_graph(n, edges)
            displays = 2 ** len(edges)
            totals["labeled_graphs"] += 1
            totals["graph_budget_pairs"] += n + 1
            totals["candidate_displays"] += displays
            totals["candidate_display_priority_scenarios"] += n * displays
    print(json.dumps({"max_vertices": args.max_vertices, "passed": True, **totals}, indent=2))


if __name__ == "__main__":
    main()
