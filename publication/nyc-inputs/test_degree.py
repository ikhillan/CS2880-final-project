"""Run from this directory with python3 -m unittest -v test_degree."""

import unittest
from itertools import combinations
from math import comb

from degree import budget_steps, degree_probs


class DegreeTests(unittest.TestCase):
    def test_all_small_catalogues_against_explicit_subsets(self):
        for L in range(11):
            for D in range(L + 1):
                for n in range(L + 1):
                    degrees = [
                        sum(i < D for i in draw)
                        for draw in combinations(range(L), n)
                    ]
                    expected = [
                        sum(check(d) for d in degrees) / len(degrees)
                        for check in (lambda d: d == 0, lambda d: 1 <= d <= 5,
                                      lambda d: d >= 6, lambda d: d >= 2)
                    ]
                    with self.subTest(L=L, D=D, n=n):
                        self.assertEqual(degree_probs(L, D, n), tuple(expected))

    def test_rare_tail_survives_complement(self):
        L, D, n = 1_000_000, 6, 6
        self.assertEqual(degree_probs(L, D, n)[2], 1 / comb(L, 6))
        self.assertGreater(degree_probs(L, D, n)[2], 0.0)

    def test_invalid_counts(self):
        for args in ((5, 6, 1), (5, 1, 6), (-1, 0, 0), (5, -1, 3), (5, 1, -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                degree_probs(*args)
        with self.assertRaises(TypeError):
            degree_probs(10, 2, 2.5)

    def test_no_non_price_eligible_units(self):
        self.assertEqual(budget_steps([], 2000), {
            'affordable_count': 0, 'first_rent': None, 'sixth_rent': None,
            'gap_to_first': None, 'gap_to_sixth': None})

    def test_thresholds_and_affordable_ties(self):
        rents = [1000, 1100, 1100, 1300, 1700, 1900, 1900]
        steps = budget_steps(rents, 900)
        self.assertEqual((steps['first_rent'], steps['sixth_rent']), (1000, 1900))
        self.assertEqual((steps['gap_to_first'], steps['gap_to_sixth']), (100, 1000))
        self.assertEqual(budget_steps(rents, 1100)['affordable_count'], 3)
        self.assertEqual(budget_steps(rents, 1900)['affordable_count'], 7)
        self.assertEqual(budget_steps(rents, 1900)['gap_to_sixth'], 0)
        under_six = budget_steps(rents[:5], 2000)
        self.assertEqual(under_six['gap_to_first'], 0)
        self.assertIsNone(under_six['sixth_rent'])
        self.assertIsNone(under_six['gap_to_sixth'])

    def test_invalid_budget(self):
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                budget_steps([], value)


if __name__ == '__main__':
    unittest.main()
