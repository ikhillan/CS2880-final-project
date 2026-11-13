"""Verify objective gradients, linear oracle, quadrature and housing mask budgets."""
from itertools import permutations, product
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from figure2 import expected_interviews, exposures
from su_baseline import _ordered_gradient, relevance, su_topk


def checks(data):
    rng = np.random.default_rng(42931)
    f, g, x = rng.uniform(.1, .9, (3, 5, 4))
    order = np.argsort(-g, axis=0)
    value, grad = _ordered_gradient(f, g, x, order)
    finite = np.empty_like(grad)
    for i, j in np.ndindex(f.shape):
        pert = np.zeros_like(x)
        pert[i,j] = 1e-6
        finite[i,j] = (_ordered_gradient(f, g, x+pert, order)[0]
                       - _ordered_gradient(f, g, x-pert, order)[0]) / 2e-6
    grad_error = np.max(np.abs(grad - finite))
    assert grad_error < 1e-8
    brute = 0
    for j in range(g.shape[1]):
        for h, c in enumerate(order[:, j]):
            higher = f[order[:h,j],j] * x[order[:h,j],j]
            for bits in product([0,1], repeat=h):
                prob = np.prod(np.where(bits, higher, 1 - higher))
                brute += f[c,j] * x[c,j] * g[c,j] * prob / (1 + sum(bits))
    exact_error = abs(brute - expected_interviews(f, g, x))
    assert exact_error < 1e-12 and brute >= value
    for row, linear in zip(exposures(grad), grad):
        oracle = max(np.dot(linear, p) for p in permutations(1/np.arange(1,5)))
        assert abs(np.dot(row, linear) - oracle) < 1e-12
    e = np.array([[False]*4, [True,True,False,False], [True]*4,
                  [True,False,False,False],[False,False,False,True]])
    scores = np.where(e, 1, np.nan)
    assert np.all(relevance(scores, e, 1)[e] == .5)
    for k in range(6):
        mask = su_topk(scores, scores, e, k)
        assert np.array_equal(mask.sum(1), np.minimum(e.sum(1), k))
        assert not (mask & ~e).any()
    results = {"gradient_max_abs_error":float(grad_error), "expectation_enum_error":float(exact_error),
               "linear_oracle_permutations_per_applicant":24,
               "runtime":dict(python=sys.version.split()[0], numpy=np.__version__)}
    if data is not None:
        for split in ("cal","summer","winter"):
            with np.load(data/f"{split}.npz") as a:
                feasible = a["e"]
            with np.load(data/f"{split}-su.npz") as a:
                mask = a["mask"]
            assert not (mask & ~feasible).any()
            assert np.array_equal(mask.sum(2), np.minimum(feasible.sum(2), 5))
            results[split] = dict(markets=len(mask), edges=int(mask.sum()))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path)
    print(json.dumps(checks(p.parse_args().data), indent=2))
