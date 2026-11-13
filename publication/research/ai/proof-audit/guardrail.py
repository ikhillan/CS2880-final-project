"""Display repair for a strict, one-to-one market; indices are local to a market."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def threat_mask(feasible, incumbent, baseline, priority=None):
    """Build a prospective graph. Lower ordinal rank is better. -1 denotes no offer."""
    feasible = np.asarray(feasible, bool)
    incumbent = np.asarray(incumbent, bool)
    baseline = np.asarray(baseline)
    n, m = feasible.shape
    assert incumbent.shape == baseline.shape == (n,)
    assert ((baseline >= -1) & (baseline < m)).all()
    owners = np.flatnonzero(incumbent & (baseline >= 0))
    sites = baseline[owners]
    assert len(np.unique(sites)) == len(sites)
    assert feasible[owners, sites].all()
    threats = np.zeros_like(feasible)
    threats[:, sites] = feasible[:, sites] & ~incumbent[:, None]
    if priority is not None:
        assert priority.shape == feasible.shape
        threats[:, sites] &= priority[:, sites] < priority[owners, sites][None, :]
    return threats


def certificate(visible_threats):
    graph = csr_matrix(visible_threats, dtype=bool)
    return int(np.count_nonzero(maximum_bipartite_matching(graph, perm_type='column') >= 0))


def best_slates(scores, allowed, quotas):
    """Exact rowwise score maximization with stable ties by increasing listing ID."""
    if (allowed.sum(axis=1) < quotas).any():
        return None
    order = np.argsort(np.where(allowed, -np.asarray(scores, dtype=float), np.inf), axis=1, kind='stable')
    position = np.argsort(order, axis=1)
    return position < quotas[:, None]


def repair(scores, feasible, threats, quotas, budget):
    """Maximize additive scores at exact quotas and a specified certificate budget.

    Row inputs contain only outsiders. Returns None iff no slates pass the certificate.
    The positive-budget MIP uses arbitrary optimal tie breaking.
    """
    scores = np.asarray(scores, float)
    feasible = np.asarray(feasible, bool)
    threats = np.asarray(threats, bool)
    quotas = np.asarray(quotas, int)
    assert scores.shape == feasible.shape == threats.shape and quotas.shape == (len(feasible),)
    assert np.isfinite(scores[feasible]).all() and not (threats & ~feasible).any()
    assert ((quotas >= 0) & (quotas <= feasible.sum(axis=1))).all() and budget >= 0
    if budget == 0:
        return best_slates(scores, feasible & ~threats, quotas)
    unguarded = best_slates(scores, feasible, quotas)
    if certificate(unguarded & threats) <= budget:
        return unguarded
    n, m = feasible.shape
    h, j = np.where(feasible)
    edges = len(h)
    danger = np.flatnonzero(threats[h, j])
    threat_ids = np.arange(len(danger))
    nvar = edges + n + m
    row = np.r_[h, n + threat_ids, n + threat_ids, n + threat_ids,
                np.full(n + m, n + len(danger))]
    col = np.r_[np.arange(edges), danger, edges + h[danger], edges + n + j[danger],
                np.arange(edges, nvar)]
    val = np.r_[np.ones(edges + len(danger)), -np.ones(2 * len(danger)), np.ones(n + m)]
    matrix = coo_matrix((val, (row, col)), shape=(n + len(danger) + 1, nvar)).tocsr()
    lower = np.r_[quotas, np.full(len(danger) + 1, -np.inf)]
    upper = np.r_[quotas, np.zeros(len(danger)), budget]
    objective = np.r_[-scores[h, j], np.zeros(n + m)]
    result = milp(objective, integrality=np.ones(nvar), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, lower, upper), options={'mip_rel_gap': 0.0})
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(result.message)
    visible = np.zeros_like(feasible)
    visible[h, j] = result.x[:edges] > 0.5
    assert np.array_equal(visible.sum(axis=1), quotas)
    assert certificate(threats & visible) <= budget
    return visible


def repair_with_freezes(feasible, scores, initial_visible, incumbent, baseline, priority=None):
    """Preserve every quota; freeze forced outsiders at their full original visible lists.

    Returns final display, final fixed-set flags and number of nonempty rounds.
    """
    fixed = np.asarray(incumbent, bool).copy()
    quotas = initial_visible.sum(axis=1)
    assert initial_visible.shape == feasible.shape == scores.shape
    assert not (initial_visible & ~feasible).any()
    rounds = 0
    while True:
        threats = threat_mask(feasible, fixed, baseline, priority)
        allowed = feasible & ~threats
        deficient = ~fixed & (allowed.sum(axis=1) < quotas)
        if not deficient.any():
            result = initial_visible.copy()
            result[~fixed] = best_slates(scores[~fixed], allowed[~fixed], quotas[~fixed])
            assert np.array_equal(result.sum(axis=1), quotas)
            assert not (result & threats).any()
            return result, fixed, rounds
        fixed |= deficient
        rounds += 1
