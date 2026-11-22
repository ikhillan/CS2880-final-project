"""Independent tiny-market optima and counterexamples for coordinated menus.

Run without additional test packages:
    python3 -m unittest tests.test_jobs_coordinated_policies
"""
from itertools import combinations, product
import unittest

import numpy as np

from experiments_jobs.capacity_anchor import apply_anchor, capacity
from experiments_jobs.coordinated_policies import (
    balanced_menu, cached_rank_relevance, normalize_score_ranks, original_anchor,
    permuted_anchor, randomized_anchor_assignment, score_anchor,
    score_anchor_assignment,
)
from hiddenoptions.matching import Instance, NOT_EDGE, deferred_acceptance


def all_matchings(E):
    """Enumerate partial injective assignments independently of SciPy."""
    n, m = E.shape
    for assignment in product(range(-1, m), repeat=n):
        used = [j for j in assignment if j >= 0]
        if len(set(used)) == len(used) and all(j < 0 or E[i, j] for i, j in enumerate(assignment)):
            yield assignment


def menu_choices(E, quotas):
    rows = [list(combinations(np.flatnonzero(E[i]), int(k))) for i, k in enumerate(quotas)]
    for choices in product(*rows):
        menu = np.zeros_like(E)
        for i, postings in enumerate(choices):
            menu[i, list(postings)] = True
        yield menu


def first_drop(i, current):
    return current


def balance_objective(menu, scores):
    return (int(np.square(menu.sum(axis=0)).sum()), -sum(int(x) for x in scores[menu]))


class CoordinatedPoliciesTests(unittest.TestCase):
    def test_archived_anchor_and_permuted_objective_exhaustive(self):
        # Each pair is infeasible, hidden feasible, or shown: all 3^6 cases.
        for state in product(range(3), repeat=6):
            state = np.asarray(state).reshape(2, 3)
            E, shown = state > 0, state == 2
            expected = apply_anchor(E, shown, np.arange(3), first_drop)
            actual = original_anchor(E, shown, np.arange(3), first_drop)
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])
            optimum = max((sum(j >= 0 for j in a), sum(j >= 0 and shown[i, j] for i, j in enumerate(a)))
                          for a in all_matchings(E))
            assignment = randomized_anchor_assignment(E, shown, seed=41)
            observed = (int((assignment >= 0).sum()),
                        sum(j >= 0 and shown[i, j] for i, j in enumerate(assignment)))
            self.assertEqual(observed, optimum)
            edited, _ = permuted_anchor(E, shown, np.arange(3), first_drop, seed=41)
            np.testing.assert_array_equal(edited.sum(1), shown.sum(1))
            self.assertFalse((edited & ~E).any())
            if all(shown[i].any() for i, j in enumerate(assignment) if j >= 0):
                self.assertEqual(capacity(edited), optimum[0])

    def test_index_remapping_and_seed_reproducibility(self):
        E = np.zeros((3, 4), bool)
        E[0, 2] = E[1, 0] = E[2, 3] = True
        for seed in range(20):
            a = randomized_anchor_assignment(E, E, seed)
            np.testing.assert_array_equal(a, [2, 0, 3])
        dense = np.ones((4, 4), bool)
        assignments = {tuple(randomized_anchor_assignment(dense, dense, seed)) for seed in range(20)}
        self.assertGreater(len(assignments), 1)
        np.testing.assert_array_equal(randomized_anchor_assignment(dense, dense, 7),
                                      randomized_anchor_assignment(dense, dense, 7))

    def test_score_normalization_and_weighted_assignment(self):
        E = np.ones((2, 3), bool)
        scores = np.array([[10., 10., 30.], [9., 4., 6.]])
        expected = np.array([[0, 0, 1000], [1000, 0, 500]])
        np.testing.assert_array_equal(normalize_score_ranks(E, np.arange(3), scores), expected)
        # All 2x3 feasibility patterns; brute-force lexicographic assignment.
        for bits in product([False, True], repeat=6):
            E = np.array(bits).reshape(2, 3)
            rank = normalize_score_ranks(E, np.arange(3), scores)
            optimum = max((sum(j >= 0 for j in a), sum(int(rank[i, j]) for i, j in enumerate(a) if j >= 0))
                          for a in all_matchings(E))
            for seed in [0, 3]:
                a = score_anchor_assignment(E, np.arange(3), scores, seed)
                self.assertEqual((int((a >= 0).sum()), sum(int(rank[i, j]) for i, j in enumerate(a) if j >= 0)),
                                 optimum)
        # A huge original score cannot trade away cardinality: ranks are bounded.
        E = np.array([[1, 1], [1, 0]], bool)
        a = score_anchor_assignment(E, np.arange(2), np.array([[1e200, -1e200], [0., 0.]]))
        self.assertEqual(int((a >= 0).sum()), 2)

    def test_balanced_exact_optimum_all_two_by_three_graphs_quotas(self):
        # All feasible graphs and all row quotas, with three contrasting score
        # matrices. Enumerate every legal output menu; do not reuse flow code.
        score_cases = [np.zeros((2, 3), dtype=int),
                       np.array([[8, -3, 2], [1, 5, -6]]),
                       np.array([[10**12, 0, 0], [10**12, 0, 0]], dtype=np.int64)]
        for bits in product([False, True], repeat=6):
            E = np.array(bits).reshape(2, 3)
            for quotas in product(*(range(int(d) + 1) for d in E.sum(1))):
                choices = list(menu_choices(E, quotas))
                shown = choices[0]
                for scores in score_cases:
                    optimum = min(balance_objective(m, scores) for m in choices)
                    out = balanced_menu(E, shown, np.arange(3), scores, seed=11)
                    self.assertEqual(balance_objective(out, scores), optimum)
                    np.testing.assert_array_equal(out.sum(1), quotas)
                    self.assertFalse((out & ~E).any())

    def test_balanced_rectangular_menus_and_seed_objective(self):
        rng = np.random.default_rng(844)
        for n, p in [(3, 4), (4, 3)]:
            for _ in range(25):
                E = rng.random((n, p)) < .75
                quotas = np.minimum(E.sum(1), rng.integers(0, 3, n))
                choices = list(menu_choices(E, quotas))
                scores = rng.integers(-10, 11, size=(n, p))
                optimum = min(balance_objective(m, scores) for m in choices)
                for seed in [0, 1, 17]:
                    out = balanced_menu(E, choices[0], np.arange(p), scores, seed=seed)
                    self.assertEqual(balance_objective(out, scores), optimum)

    def test_posting_quotas_and_cloned_slots(self):
        s2p = np.array([0, 0, 1, 2, 2, 2])
        E_post = np.array([[1, 1, 1], [1, 1, 0], [0, 1, 1]], bool)
        shown_post = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]], bool)
        E, shown = E_post[:, s2p], shown_post[:, s2p]
        score = np.array([[1, 2, 3], [4, 5, 0], [0, 6, 7]])
        for method in [lambda: original_anchor(E, shown, s2p, first_drop)[0],
                       lambda: permuted_anchor(E, shown, s2p, first_drop, seed=9)[0],
                       lambda: score_anchor(E, shown, s2p, first_drop, score, seed=9)[0],
                       lambda: balanced_menu(E, shown, s2p, score, seed=9)]:
            out = method()
            collapsed = np.column_stack([out[:, s2p == p].any(1) for p in range(3)])
            np.testing.assert_array_equal(collapsed.sum(1), shown_post.sum(1))
            np.testing.assert_array_equal(out, collapsed[:, s2p] & E)

    def test_capacity_increases_while_da_placement_decreases(self):
        E = np.array([[1,1,1,1], [1,1,1,1], [1,0,1,1], [1,1,0,1]], bool)
        rb = np.array([[2,3,1,0], [2,3,0,1], [1,NOT_EDGE,2,0], [0,1,NOT_EDGE,2]])
        rs = np.array([[1,1,0,3], [2,2,1,2], [0,NOT_EDGE,2,0], [3,0,NOT_EDGE,1]])
        inst = Instance(rb, rs, np.arange(4), np.arange(4,8), np.ones(8, dtype=int))
        shown = np.array([[1,1,0,0], [1,1,0,0], [1,0,0,1], [0,1,0,1]], bool)
        out, changed = original_anchor(E, shown, np.arange(4), lambda i, cur: sorted(cur, reverse=True))
        before, after = deferred_acceptance(inst, shown)[0], deferred_acceptance(inst, out)[0]
        self.assertEqual((capacity(shown), capacity(out), changed), (3, 4, 1))
        np.testing.assert_array_equal(before, [0, -1, 3, 1])
        np.testing.assert_array_equal(after, [-1, -1, 0, 1])
        self.assertEqual(int(((before >= 0) & (after < 0)).sum()), 1)

    def test_fixed_quotas_can_make_maximum_da_cardinality_impossible(self):
        rb = np.array([[0,1], [0,NOT_EDGE]])
        rs = np.array([[0,0], [1,NOT_EDGE]])
        inst = Instance(rb, rs, np.arange(2), np.arange(2,4), np.ones(4, dtype=int))
        self.assertEqual(capacity(inst.E), 2)
        self.assertEqual(int((deferred_acceptance(inst, inst.E)[0] >= 0).sum()), 1)
        # Quotas equal feasible row degrees: there is only one legal menu.
        self.assertEqual(len(list(menu_choices(inst.E, inst.E.sum(1)))), 1)

    def test_balanced_loads_do_not_certify_matching_capacity(self):
        # Common k=2, truncated only where fewer feasible postings exist.
        # An optimal squared-load solution can still violate a Hall constraint.
        E = np.array([[0,1,1,1], [0,0,0,1], [1,1,1,0], [0,1,0,0]], bool)
        shown = np.array([[0,1,1,0], [0,0,0,1], [1,1,0,0], [0,1,0,0]], bool)
        scores = np.array([[4,2,0,2], [0,1,2,4], [1,2,2,3], [3,2,4,3]])
        out = balanced_menu(E, shown, np.arange(4), scores, seed=0)
        optimum = min(balance_objective(m, scores) for m in menu_choices(E, shown.sum(1)))
        self.assertEqual(balance_objective(out, scores), optimum)
        self.assertEqual((capacity(out), capacity(E)), (3,4))

    def test_empty_rows_break_full_capacity_certificate(self):
        E = np.ones((2,2), bool)
        shown = np.array([[0,0], [1,0]], bool)
        out, changed = original_anchor(E, shown, np.arange(2), first_drop)
        self.assertEqual(capacity(E), 2)
        self.assertEqual(capacity(out), 1)
        self.assertEqual(changed, 0)
        np.testing.assert_array_equal(out[0], [False, False])

    def test_censored_cached_relevance_and_input_validation(self):
        E = np.ones((2,3), bool)
        shown = np.array([[1,0,1], [0,1,0]], bool)
        scores = cached_rank_relevance(E, shown, np.arange(3), [[2,0], [1]])
        np.testing.assert_array_equal(scores, [[1,0,2], [0,1,0]])
        with self.assertRaises(ValueError):
            cached_rank_relevance(E, shown, np.arange(3), [[2], [1]])
        with self.assertRaises(ValueError):
            balanced_menu(E, shown, np.arange(3), np.ones((2,3), dtype=float))
        with self.assertRaises(ValueError):
            normalize_score_ranks(E, np.arange(3), np.full((2,3), np.nan))
        with self.assertRaises(ValueError):
            original_anchor(np.ones((1,2), bool), np.array([[1,0]], bool), np.array([0,0]), first_drop)
        empty = np.zeros((2,0), bool)
        self.assertEqual(balanced_menu(empty, empty, np.array([], int), np.zeros((2,0), int)).shape, (2,0))
        self.assertEqual(original_anchor(empty, empty, np.array([], int), first_drop)[1], 0)


if __name__ == "__main__":
    unittest.main()
