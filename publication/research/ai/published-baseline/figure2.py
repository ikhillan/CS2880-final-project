"""Numerical reproduction of the WWW 2022 Figure 2 inverse-rank experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from su_baseline import _ordered_gradient


def exposures(scores):
    position_weights = 1 / np.arange(1, scores.shape[1] + 1)
    out = np.empty_like(scores)
    ranks = np.argsort(-scores, axis=1, kind="stable")
    np.put_along_axis(out, ranks, position_weights[None, :], axis=1)
    return out


def optimize(f, g, *, inverse_order):
    """Projection of the release's doubly stochastic update onto exposure.

    The objective is a function of row exposures. Its matrix gradient is the
    outer product of the exposure gradient and 1/rank, so sorting implements
    the row linear oracle exactly by the rearrangement inequality.
    """
    order = np.argsort(-g, axis=0, kind="stable")
    if inverse_order:
        order = np.argsort(order, axis=0, kind="stable")
    x = np.full_like(f, np.mean(1 / np.arange(1, f.shape[1] + 1)))
    prev = 0
    for _ in range(51):
        value, grad = _ordered_gradient(f, g, x, order)
        x += 0.2 * (exposures(grad) - x)
        if abs(value - prev) < 1e-3:
            break
        prev = value
    return x


def expected_interviews(f, g, x):
    """Evaluate the original Bernoulli model exactly up to floating point.

    Conditional on an application, higher-ranked candidates independently
    apply with probabilities p_h. E[1/(1+N)] equals the integral from 0 to 1
    of the probability-generating function of their count N. The integrand is
    polynomial of degree < number of candidates, so Gauss-Legendre quadrature
    with ceil(C/2) nodes is exact in real arithmetic.
    """
    order = np.argsort(-g, axis=0, kind="stable")
    p = np.take_along_axis(f * x, order, axis=0)
    benefit = p * np.take_along_axis(g, order, axis=0)
    nodes, weights = leggauss((len(f) + 1) // 2)
    integrals = np.zeros_like(f)
    for t, w in zip((nodes + 1) / 2, weights / 2):
        prefix = np.ones_like(f)
        prefix[1:] = np.cumprod(1 - (1 - t) * p[:-1], axis=0)
        integrals += w * prefix
    return np.sum(benefit * integrals)


def study():
    rows = []
    for n, crowding_levels in [(20, [0.5]), (100, [0, 0.25, 0.5, 0.75, 1])]:
        c = 3 * n // 2
        for seed in range(621, 631):
            rng = np.random.RandomState(seed)   # matches the authors' legacy RNG
            f0, g0 = rng.random((c, n)), rng.random((n, c)).T
            for crowd in crowding_levels:
                f = (1 - crowd) * f0 + crowd * np.linspace(1, 0, n)[None, :]
                g = (1 - crowd) * g0 + crowd * np.linspace(1, 0, c)[:, None]
                variants = {"naive": exposures(f), "reciprocal": exposures(f * g),
                            "release_order": optimize(f, g, inverse_order=True),
                            "equation_order": optimize(f, g, inverse_order=False)}
                for label, x in variants.items():
                    rows.append(dict(jobs=n, applicants=c, seed=seed, crowding=crowd,
                                     policy=label, interviews=expected_interviews(f, g, x)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = study()
    with (args.outdir / "figure2-markets.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for jobs, crowding, policy in dict.fromkeys(
            (r["jobs"], r["crowding"], r["policy"]) for r in rows):
        values = [r["interviews"] for r in rows if
                  (r["jobs"], r["crowding"], r["policy"]) == (jobs, crowding, policy)]
        summary.append(dict(jobs=jobs, crowding=crowding, policy=policy,
                            mean=np.mean(values), market_sd=np.std(values, ddof=1)))
    (args.outdir / "figure2-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
