"""Apply the published two-stage implementation to the released relevance matrix."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle

import numpy as np
from numpy.polynomial.legendre import leggauss

from figure2 import exposures
from su_baseline import _ordered_gradient


def constrained_exposure(f, g, *, inverse_order=False):
    """Reorder only the top-100 reciprocal options, keep the tail frozen."""
    rank_weights = 1 / np.arange(1, f.shape[1] + 1)
    reciprocal = np.argsort(-f * g, axis=1, kind="stable")
    head = reciprocal[:, :100]
    x = exposures(f * g)
    np.put_along_axis(x, head, np.mean(rank_weights[:100]), axis=1)
    order = np.argsort(-g, axis=0, kind="stable")
    if inverse_order:
        order = np.argsort(order, axis=0, kind="stable")
    prev = 0
    for _ in range(51):
        value, grad = _ordered_gradient(f, g, x, order)
        h = np.take_along_axis(grad, head, axis=1)
        vertex = exposures(h)
        current = np.take_along_axis(x, head, axis=1)
        np.put_along_axis(x, head, .8 * current + .2 * vertex, axis=1)
        if abs(prev - value) < 1e-3:
            break
        prev = value
    return x


def integrate(f, g, x, nodes):
    order = np.argsort(-g, axis=0, kind="stable")
    apply = np.take_along_axis(f * x, order, axis=0)
    benefit = apply * np.take_along_axis(g, order, axis=0)
    out = 0
    locations, weights = leggauss(nodes)
    for t, w in zip((locations + 1) / 2, weights / 2):
        prefix = np.ones_like(apply)
        prefix[1:] = np.cumprod(1 - (1 - t) * apply[:-1], axis=0)
        out += w * (benefit * prefix).sum()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = (args.upstream / "data/networking/networking_relevances.pkl").read_bytes()
    if sha256(data).hexdigest() != "a6ce5cadab45003a69e119c276e6af4b3d0b163b2b87804b6124f6674cbda1de":
        raise ValueError("Unexpected public benchmark file")
    rel = pickle.loads(data)  # pinned public NumPy ndarray, shape and bytes verified
    assert rel.shape == (925, 925) and np.all((0 <= rel) & (rel <= 1))
    rows = []
    for keep_self in (False, True):
        f = rel.copy()
        if not keep_self:
            np.fill_diagonal(f, 0)
        g = f.T
        policies = dict(naive=exposures(f), reciprocal=exposures(f*g),
                        equation_order=constrained_exposure(f, g),
                        inverse_order=constrained_exposure(f, g, inverse_order=True))
        for label, x in policies.items():
            v32 = integrate(f, g, x, 32)
            v16 = integrate(f, g, x, 16)
            rows.append(dict(keep_self=keep_self, policy=label, expected_replies=v32,
                             difference_16_to_32_nodes=float(abs(v32 - v16))))
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
