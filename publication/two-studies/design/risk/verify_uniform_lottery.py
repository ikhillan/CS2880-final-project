"""Exhaust small uniform common vs. school tie lotteries against the fixed-witness formulas."""
from fractions import Fraction
from itertools import combinations, permutations, product
from math import comb, factorial, prod
import json

from verify_lottery_boundary import deferred_acceptance, stable


def stable_with_capacities(assigned, menus, capacities, orders):
    holders = [[] for _ in capacities]
    for person, place in enumerate(assigned):
        if place is None:
            continue
        if place not in menus[person]:
            return False
        holders[place].append(person)
    for person, options in enumerate(menus):
        partner = assigned[person]
        for place in options:
            if place == partner:
                break
            if len(holders[place]) < capacities[place]:
                return False
            rank = orders[place].index(person)
            if any(orders[place].index(h) > rank for h in holders[place]):
                return False
    return True


def all_subsets(items):
    for length in range(len(items) + 1):
        yield from combinations(items, length)


def check_guarded_risk():
    counts = []
    for t in range(1, 7):
        baseline = [(0,), (0, 2, 3)] + [(1,)] * t
        proposed = [(0,), (0, 1, 2)] + [(1,)] * t
        failure = 0
        for order in permutations(range(t + 2)):
            priority = {person: index for index, person in enumerate(order)}
            old = deferred_acceptance(baseline, priority)
            new = deferred_acceptance(proposed, priority)
            loss = any(was is not None and now is None for was, now in zip(old, new))
            blocked = not stable(old, proposed, priority)
            assert loss == blocked == (order[:2] == (0, 1))
            failure += blocked
        assert failure == factorial(t)
        counts.append({"incumbent_contenders": t, "loss_probability": str(Fraction(failure, factorial(t + 2)))})
    return counts


def menus_with_own(base):
    others = [[j for j in range(max(base) + 1) if j != own] for own in base]
    for selected in product(*(tuple(all_subsets(row)) for row in others)):
        yield [tuple(row) + (own,) for row, own in zip(selected, base)]


def check_lottery_formulas():
    # All placements are held fixed. Every program is tied before the lottery.
    # Enumerate all displays of additional better-ranked options for each base allocation.
    instances = 0
    scenario_checks = 0
    for assigned in [(0, 1, 2), (0, 0, 1), (0, 0, 1, 1)]:
        people = tuple(range(len(assigned)))
        programs = range(max(assigned) + 1)
        capacities = [assigned.count(j) for j in programs]
        permutations_all = tuple(permutations(people))
        for menus in menus_with_own(assigned):
            arcs = {(h, i) for i, menu in enumerate(menus) for j in menu[:-1]
                    for h in people if assigned[h] == j}
            topological_orders = 0
            for order in permutations_all:
                satisfies = all(order.index(h) < order.index(i) for h, i in arcs)
                assert stable_with_capacities(assigned, menus, capacities, [order] * len(capacities)) == satisfies
                topological_orders += satisfies
            if arcs:
                assert topological_orders * 2 <= len(permutations_all)
            independent = sum(stable_with_capacities(assigned, menus, capacities, orders)
                              for orders in product(permutations_all, repeat=len(capacities)))
            denominator = len(permutations_all) ** len(capacities)
            predicted = prod(Fraction(1, comb(capacities[j] + sum(j in row[:-1] for row in menus),
                                              capacities[j])) for j in programs)
            assert Fraction(independent, denominator) == predicted
            scenario_checks += denominator + len(permutations_all)
            instances += 1
    return {"candidate_displays": instances, "priority_checks": scenario_checks}


if __name__ == "__main__":
    print(json.dumps({"fixed_assignment": check_lottery_formulas(), "draw_dependent": check_guarded_risk(),
                      "passed": True}, indent=2))
