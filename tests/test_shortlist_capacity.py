"""Exhaust small shortlist choices to check the minimum-edit claim independent of the assignment oracle."""
from itertools import combinations, product

import numpy as np

from experiments.shortlist_capacity import repair
from hiddenoptions.markets import maximum_matching_size


def test_small_quota_repairs_are_minimal():
    rng = np.random.default_rng(901)
    for _ in range(100):
        eligible = rng.uniform(size=(3,4)) < .75
        scores = np.where(eligible, rng.normal(size=eligible.shape), -np.inf)
        quota = np.minimum(rng.integers(1,4,size=3), eligible.sum(1))
        choices = [list(combinations(np.flatnonzero(row),k)) for row,k in zip(eligible,quota)]
        shown = np.zeros_like(eligible)
        for i,opts in enumerate(choices):
            shown[i,list(opts[rng.integers(len(opts))])] = True
        result, edits = repair(eligible,shown,scores)
        full = maximum_matching_size(eligible)
        assert maximum_matching_size(result)==full
        assert np.array_equal(result.sum(1), quota)
        assert int((result & ~shown).sum())==len(edits)
        best = int(shown.sum())
        for replacement in product(*choices):
            g = np.zeros_like(eligible)
            for i,options in enumerate(replacement):
                g[i,list(options)] = True
            if maximum_matching_size(g)==full:
                best = min(best, int((g & ~shown).sum()))
        assert len(edits)==best
