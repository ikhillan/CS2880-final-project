"""Two-sided ranking, after Su, Bayoumi & Joachims (WWW 2022).

All score matrices have shape (applicants, listings). The housing adapter
uses only the feasible mask and the two observable score matrices.
"""
from operator import index

import numpy as np
from scipy.stats import rankdata


def relevance(scores, feasible, axis):
    """Mid-distribution transform on feasible edges, preserving exact ties."""
    observed = np.where(feasible, scores, np.nan)
    ranks = rankdata(observed, axis=axis, method="average", nan_policy="omit")
    counts = feasible.sum(axis=axis, keepdims=True)
    return np.where(feasible, (ranks - 0.5) / np.maximum(counts, 1), 0.0)


def _ordered_gradient(f, g, x, order):
    """Jensen bound and its exposure gradient for inverse-rank attention."""
    apply = np.take_along_axis(f * x, order, axis=0)
    pref = np.take_along_axis(g, order, axis=0)
    rank = 1 + np.cumsum(apply, axis=0) - apply
    value = pref * apply / rank
    cost = value / rank
    downstream = np.cumsum(cost[::-1], axis=0)[::-1] - cost
    grad = np.empty_like(x)
    np.put_along_axis(grad, order, pref / rank - downstream, axis=0)
    return value.sum(), f * grad


def _topk(scores, feasible, budget):
    order = np.argsort(-np.where(feasible, scores, -np.inf), axis=1, kind="stable")
    result = np.zeros_like(feasible, dtype=bool)
    np.put_along_axis(result, order[:, :budget], True, axis=1)
    return result & feasible


def su_topk(renter_scores, listing_scores, feasible, k=5, *, steps=50):
    """Deterministic rounded shortlist using Frank–Wolfe on fractional top-k.

    Each row has min(k, feasible degree) ones. Equal landlord relevance and
    equal shortlist scores are resolved by input order; the panel runner
    independently permutes both axes with a reproducible market-specific key.
    """
    renter_scores = np.asarray(renter_scores)
    listing_scores = np.asarray(listing_scores)
    e = np.asarray(feasible, dtype=bool)
    if e.ndim != 2 or not (e.shape == renter_scores.shape == listing_scores.shape):
        raise ValueError("Need equally shaped applicant-by-listing matrices")
    k, steps = index(k), index(steps)
    if k < 0 or steps < 1:
        raise ValueError("k must be a nonnegative integer and steps a positive integer")
    if not (np.isfinite(renter_scores[e]).all() and np.isfinite(listing_scores[e]).all()):
        raise ValueError("Observable scores must be finite on feasible edges")
    if k == 0 or not e.any():
        return np.zeros_like(e)
    if k >= e.sum(axis=1).max():
        return e.copy()

    f = relevance(renter_scores, e, axis=1)
    g = relevance(listing_scores, e, axis=0)
    priority = np.argsort(-g, axis=0, kind="stable")
    counts = e.sum(axis=1, keepdims=True)
    x = e * (np.minimum(counts, k) / np.maximum(counts, 1))
    for _ in range(steps):
        _, grad = _ordered_gradient(f, g, x, priority)
        vertex = _topk(grad, e, k)
        x += 0.2 * (vertex - x)
    return _topk(x, e, k)
