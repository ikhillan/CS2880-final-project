"""Independent enumeration of feasible assignments, fixed-size menus, and three separating examples."""

from itertools import combinations, permutations, product
import json
import numpy as np
from chain_flow import Contracts, evaluate


def assignment(rows, priorities, capacities, menus):
    seats = [set() for _ in capacities]
    held = [-1] * len(rows)
    free = list(range(len(rows)))
    next_choice = [0] * len(rows)
    while free:
        i = free.pop()
        while next_choice[i] < len(rows[i]):
            j = rows[i][next_choice[i]]
            next_choice[i] += 1
            if j not in menus[i] or capacities[j] == 0:
                continue
            if len(seats[j]) == capacities[j]:
                worst = max(seats[j], key=priorities[j].index)
                if priorities[j].index(worst) < priorities[j].index(i):
                    continue
                seats[j].remove(worst)
                held[worst] = -1
                free.append(worst)
            held[i] = j
            seats[j].add(i)
            break
    return tuple(held)


def rank(row, school):
    return len(row) if school < 0 else row.index(school)


def one_case(rows, priorities, capacities, menus, implement):
    n = len(rows)
    old = assignment(rows, priorities, capacities, menus)
    occ = [old.count(j) for j in range(len(capacities))]
    claims = [
        [
            i
            for i in range(n)
            if j in menus[i] and rank(rows[i], j) < rank(rows[i], old[i])
        ]
        for j in range(len(capacities))
    ]
    at = np.r_[0, np.cumsum(list(map(len, rows)))]
    j = np.array([school for row in rows for school in row], dtype=int)
    owner = np.repeat(np.arange(n), np.diff(at))
    pr = np.array([priorities[school].index(i) for i, school in zip(owner, j)])
    a = Contracts(at, j, pr, np.array(capacities), np.full(n, -1))
    shown = np.array([school in menus[i] for i, school in zip(owner, j)], dtype=bool)

    def edges(partners):
        return np.array(
            [-1 if k < 0 else at[i] + rows[i].index(k) for i, k in enumerate(partners)]
        )

    def da(mask):
        selected = [
            set(j[at[i] : at[i + 1]][mask[at[i] : at[i + 1]]]) for i in range(n)
        ]
        return edges(assignment(rows, priorities, capacities, selected))

    result = evaluate(a, edges(old), shown, da)
    brute = dict(direct=0, chain=0, upper=0)
    for current in product(*[[*row, -1] for row in rows]):
        count = [current.count(k) for k in range(len(capacities))]
        if any(c > q for c, q in zip(count, capacities)):
            continue
        if any(
            rank(row, new) > rank(row, previous)
            for row, new, previous in zip(rows, current, old)
        ):
            continue
        extra = old.count(-1) - current.count(-1)
        brute["upper"] = max(brute["upper"], extra)
        if all(o == -1 or o == c for o, c in zip(old, current)):
            brute["direct"] = max(brute["direct"], extra)
        if any(c < o for c, o in zip(count, occ)):
            continue
        if any(
            priorities[k].index(i) > priorities[k].index(h)
            for i, k in enumerate(current)
            if k >= 0
            for h in claims[k]
        ):
            continue
        brute["chain"] = max(brute["chain"], extra)
    for key, value in brute.items():
        assert result[key] == value, (
            rows,
            priorities,
            capacities,
            menus,
            key,
            result,
            brute,
        )
    if implement:
        permissible = [
            [set(x) for x in combinations(row, len(menu)) if len(menu - set(x)) <= 1]
            for row, menu in zip(rows, menus)
        ]
        best = 0
        for edited in product(*permissible):
            actual = assignment(rows, priorities, capacities, edited)
            if all(
                rank(row, x) <= rank(row, y) for row, x, y in zip(rows, actual, old)
            ):
                best = max(best, old.count(-1) - actual.count(-1))
        brute["best_one_swap_DA"] = best
        assert brute["chain"] <= best <= brute["upper"]
    return brute


def main():
    seats = tuple(permutations((0, 1)))
    profiles = 0
    for rows in product(seats, repeat=2):
        for priorities in product(seats, repeat=2):
            for q in product(range(3), repeat=2):
                for menus in product(({0}, {1}, {0, 1}), repeat=2):
                    one_case(rows, priorities, q, menus, True)
                    profiles += 1
    # Examples with incomplete options and heterogeneous priorities; each row uses k=1 or k=2.
    examples = [
        (((1, 0), (0,)), ((0, 1), (1, 0)), (1, 1), ({0}, {0})),
        (
            ((2, 0, 1), (0, 1, 2), (0,)),
            ((0, 1, 2), (1, 0, 2), (0, 1, 2)),
            (1, 1, 1),
            ({0, 1}, {0, 1}, {0}),
        ),
        (
            ((2, 0, 1), (0, 1), (0,)),
            ((0, 1, 2), (1, 0, 2), (0, 1, 2)),
            (1, 1, 1),
            ({0, 1}, {0, 1}, {0}),
        ),
    ]
    observed = [one_case(*x, True) for x in examples]
    assert [
        (e["direct"], e["chain"], e["best_one_swap_DA"], e["upper"]) for e in observed
    ] == [(0, 1, 1, 1), (0, 0, 1, 1), (0, 0, 0, 1)]
    locked = Contracts(
        np.array([0, 1, 3]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([1, 1]),
        np.array([-1, 0]),
    )
    initial = np.array([0, -1])
    display = np.array([True, False, True])

    def unchanged(mask):
        assert np.array_equal(mask, display)
        return initial

    locked_result = evaluate(locked, initial, display, unchanged)
    assert (
        locked_result["direct"],
        locked_result["chain"],
        locked_result["upper"],
    ) == (0, 0, 1)
    print(
        json.dumps(
            dict(
                exhaustive_two_by_two=profiles,
                separating_examples=observed,
                locked_row=[0, 0, 1],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
