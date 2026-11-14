from itertools import product

import numpy as np

from experiments.pilot_referrals import extreme_extension


def test_income_bounds_match_enumeration():
    # Include empty rows, vacancies, competition and multiple maximum assignments.
    n = 3
    lower = np.array([True, False, True])
    candidates = [p for p in product(range(-1, n), repeat=n)
                  if len({j for j in p if j >= 0}) == sum(j >= 0 for j in p)]
    for bits in product([False, True], repeat=n*n):
        edges = np.array(bits).reshape(n, n)
        possibilities = [(sum(j >= 0 for j in p), sum(lower[i] for i,j in enumerate(p) if j >= 0))
                         for p in candidates if all(j < 0 or edges[i,j] for i,j in enumerate(p))]
        size = max(x for x, _ in possibilities)
        group = [g for x, g in possibilities if x == size]
        for favored, bound in ((True, max(group)), (False, min(group))):
            rows, _ = extreme_extension(edges, lower, favored)
            assert (len(rows), lower[rows].sum()) == (size, bound)
