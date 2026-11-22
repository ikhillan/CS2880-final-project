"""Coordinated menus using feasibility, saved recommendations and supplied scores.

No function accepts an Instance, realized preferences, employer priorities or
assignments. Supplying scores computed from such information would nevertheless
be leakage; the caller must record score provenance. Deterministic simulator
scores are a model-informed benchmark, not independently learned predictions.

Anchors preserve posting quotas, not individual assignments. Their maximum
matching certificate concerns the displayed graph, not the DA outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments_jobs.capacity_anchor import apply_anchor, anchor_assignment

DropOrder = Callable[[int, np.ndarray], Sequence[int]]


def _inputs(E, shown, s2p):
    E, shown = np.asarray(E, dtype=bool), np.asarray(shown, dtype=bool)
    s2p = np.asarray(s2p)
    if E.ndim != 2 or shown.shape != E.shape:
        raise ValueError("E and shown must have the same two-dimensional shape")
    if s2p.ndim != 1 or len(s2p) != E.shape[1] or not np.issubdtype(s2p.dtype, np.integer):
        raise ValueError("s2p must be one integer posting index per slot")
    if len(s2p) and (s2p.min() < 0 or not np.array_equal(np.unique(s2p), np.arange(s2p.max() + 1))):
        raise ValueError("posting indices must be contiguous from zero")
    if np.any(shown & ~E):
        raise ValueError("shown contains an infeasible pair")
    n_post = int(s2p.max()) + 1 if len(s2p) else 0
    E_post = np.zeros((E.shape[0], n_post), dtype=bool)
    shown_post = np.zeros_like(E_post)
    for slot, post in enumerate(s2p):
        E_post[:, post] |= E[:, slot]
        shown_post[:, post] |= shown[:, slot]
    if not np.array_equal(shown_post[:, s2p] & E, shown):
        raise ValueError("a displayed posting must expose all its feasible slots")
    return E, shown, s2p.astype(int, copy=False), E_post, shown_post


def _insert(E, shown_post, s2p, assignment, drop_order):
    out = shown_post.copy()
    changed = 0
    for i, slot in enumerate(assignment):
        if slot < 0:
            continue
        post = int(s2p[slot])
        current = np.flatnonzero(out[i])
        if out[i, post] or not len(current):
            continue
        order = list(drop_order(i, current.copy()))
        if not order or order[0] not in current:
            raise ValueError("drop_order must return a currently displayed posting first")
        out[i, int(order[0])] = False
        out[i, post] = True
        changed += 1
    return out[:, s2p] & E, changed


def original_anchor(E, shown, s2p, drop_order: DropOrder):
    """Reproduce the archived apply_anchor output, including skipped empty rows.

    The original solver uses BIG=10,000. Its lexicographic certificate applies
    to the archived small markets; do not extrapolate that constant to huge ones.
    """
    E, shown, s2p, _, _ = _inputs(E, shown, s2p)
    if not E.shape[1]:
        return shown.copy(), 0
    return apply_anchor(E, shown, s2p, drop_order)


def randomized_anchor_assignment(E, shown, seed: int = 0):
    """Same archived min-hidden-edge objective, with seeded row/slot tie order."""
    E, shown = np.asarray(E, dtype=bool), np.asarray(shown, dtype=bool)
    if E.ndim != 2 or shown.shape != E.shape or np.any(shown & ~E):
        raise ValueError("invalid assignment graph")
    rng = np.random.default_rng(seed)
    rows, cols = rng.permutation(E.shape[0]), rng.permutation(E.shape[1])
    permuted = anchor_assignment(E[np.ix_(rows, cols)], shown[np.ix_(rows, cols)])
    assignment = np.full(E.shape[0], -1, dtype=int)
    matched = permuted >= 0
    assignment[rows[matched]] = cols[permuted[matched]]
    return assignment


def permuted_anchor(E, shown, s2p, drop_order: DropOrder, seed: int = 0):
    """Tie sensitivity for the existing anchor; drops use original pupil IDs."""
    E, shown, s2p, _, shown_post = _inputs(E, shown, s2p)
    assignment = randomized_anchor_assignment(E, shown, seed)
    return _insert(E, shown_post, s2p, assignment, drop_order)


def normalize_score_ranks(E, s2p, score_post, scale: int = 1000):
    """Integer within-person dense-rank scores, 0..scale; tied scores stay tied.

    A row with only one distinct feasible score receives scale on every option.
    This is an explicit ordinal scoring convention, not an admission probability
    or an interpersonally calibrated utility. Infeasible scores are ignored.
    """
    if not isinstance(scale, int) or not 1 <= scale <= 1_000_000:
        raise ValueError("scale must be an integer in 1..1,000,000")
    E = np.asarray(E, dtype=bool)
    E, _, s2p, E_post, _ = _inputs(E, np.zeros_like(E), s2p)
    score = np.asarray(score_post, dtype=float)
    if score.shape != E_post.shape or not np.isfinite(score[E_post]).all():
        raise ValueError("provide a finite score for every feasible pupil-posting pair")
    ranks = np.zeros(E_post.shape, dtype=np.int64)
    for i in range(E.shape[0]):
        values = np.unique(score[i, E_post[i]])
        if not len(values):
            continue
        if len(values) == 1:
            ranks[i, E_post[i]] = scale
        else:
            position = np.searchsorted(values, score[i, E_post[i]])
            ranks[i, E_post[i]] = np.rint(scale * position / (len(values) - 1)).astype(np.int64)
    return ranks


def score_anchor_assignment(E, s2p, score_post, seed: int = 0):
    """Maximize cardinality first, then normalized supplied-score rank.

    Unlike the archived anchor, reuse of old menu edges is not a secondary
    objective here. A bounded dynamic penalty enforces the stated lexicography.
    """
    E = np.asarray(E, dtype=bool)
    E, _, s2p, _, _ = _inputs(E, np.zeros_like(E), s2p)
    rank = normalize_score_ranks(E, s2p, score_post)[:, s2p]
    p = min(E.shape)
    penalty = p * 1000 + 1
    cost = np.where(E, -rank, penalty)
    rng = np.random.default_rng(seed)
    rows, cols = rng.permutation(E.shape[0]), rng.permutation(E.shape[1])
    rr, cc = linear_sum_assignment(cost[np.ix_(rows, cols)])
    rr, cc = rows[rr], cols[cc]
    keep = E[rr, cc]
    assignment = np.full(E.shape[0], -1, dtype=int)
    assignment[rr[keep]] = cc[keep]
    return assignment


def score_anchor(E, shown, s2p, drop_order: DropOrder, score_post, seed: int = 0):
    """Insert an anchor optimized for explicitly supplied extra-information scores."""
    E, shown, s2p, _, shown_post = _inputs(E, shown, s2p)
    assignment = score_anchor_assignment(E, s2p, score_post, seed)
    return _insert(E, shown_post, s2p, assignment, drop_order)


def cached_rank_relevance(E, shown, s2p, cached_ranked_postings):
    """Ordinal relevance for balancing, using only saved ordered menu choices.

    The first saved displayed option receives k, then k-1, ..., 1. All
    unlisted options tie at zero: this is a censored-choice convention, not
    evidence that their true relevance is zero. Duplicate IDs are rejected.
    Every currently displayed posting must appear in its supplied ranking.
    """
    _, _, _, E_post, shown_post = _inputs(E, shown, s2p)
    if len(cached_ranked_postings) != len(E_post):
        raise ValueError("one cached ranking is required per pupil")
    relevance = np.zeros(E_post.shape, dtype=np.int64)
    for i, ranking in enumerate(cached_ranked_postings):
        ranking = list(ranking)
        if len(set(ranking)) != len(ranking):
            raise ValueError("duplicate cached posting IDs")
        if any(not isinstance(p, (int, np.integer)) or not 0 <= p < E_post.shape[1] for p in ranking):
            raise ValueError("invalid cached posting ID")
        ranked_shown = [int(p) for p in ranking if shown_post[i, p]]
        if set(ranked_shown) != set(np.flatnonzero(shown_post[i])):
            raise ValueError("cached ranking does not cover the displayed menu")
        for value, post in zip(range(len(ranked_shown), 0, -1), ranked_shown):
            relevance[i, post] = value
    return relevance


@dataclass
class _Edge:
    to: int
    rev: int
    cap: int
    cost: int


def _add(graph, source, target, cap, cost):
    forward = _Edge(target, len(graph[target]), int(cap), int(cost))
    reverse = _Edge(source, len(graph[source]), 0, -int(cost))
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def _send_flow(graph, source, sink, demand):
    """Exact-integer successive shortest paths with reduced-cost potentials."""
    n = len(graph)
    distance = [None] * n
    distance[source] = 0
    # Initial forward network is acyclic. Bellman--Ford also handles negative
    # relevance costs without floating-point tolerances or giant numeric bounds.
    for _ in range(n - 1):
        changed = False
        for u, edges in enumerate(graph):
            if distance[u] is None:
                continue
            for edge in edges:
                candidate = distance[u] + edge.cost
                if edge.cap and (distance[edge.to] is None or candidate < distance[edge.to]):
                    distance[edge.to] = candidate
                    changed = True
        if not changed:
            break
    potential = [0 if d is None else d for d in distance]
    sent = 0
    while sent < demand:
        distance, previous = [None] * n, [None] * n
        distance[source] = 0
        queue = [(0, source)]
        while queue:
            dist, u = heapq.heappop(queue)
            if distance[u] != dist:
                continue
            for idx, edge in enumerate(graph[u]):
                if not edge.cap:
                    continue
                reduced = edge.cost + potential[u] - potential[edge.to]
                if reduced < 0:
                    raise ArithmeticError("invalid min-cost-flow potential")
                candidate = dist + reduced
                if distance[edge.to] is None or candidate < distance[edge.to]:
                    distance[edge.to] = candidate
                    previous[edge.to] = (u, idx)
                    heapq.heappush(queue, (candidate, edge.to))
        if distance[sink] is None:
            raise ValueError("posting quotas are infeasible")
        for u, dist in enumerate(distance):
            if dist is not None:
                potential[u] += dist
        amount, v = demand - sent, sink
        while v != source:
            u, idx = previous[v]
            amount = min(amount, graph[u][idx].cap)
            v = u
        v = sink
        while v != source:
            u, idx = previous[v]
            edge = graph[u][idx]
            edge.cap -= amount
            graph[v][edge.rev].cap += amount
            v = u
        sent += amount


def balanced_menu(E, shown, s2p, relevance_post, seed: int = 0):
    """Minimize squared posting loads, then maximize supplied integer relevance.

    Feasibility and each person's exact original POSTING quota are fixed. A
    posting's load is its number of displayed menus, not its hire capacity.
    This is a full-menu intervention and may replace more than one option.
    It guarantees neither maximum matching capacity nor a DA protection result.

    With Q displayed pairs and score range R, multiplier Q*R+1 makes squared
    load lexicographically primary. Parallel unit arcs of costs 1,3,5,...
    represent each posting's convex load cost exactly. No new package is used.
    """
    E, shown, s2p, E_post, shown_post = _inputs(E, shown, s2p)
    relevance = np.asarray(relevance_post)
    if relevance.shape != E_post.shape or not np.issubdtype(relevance.dtype, np.integer):
        raise ValueError("relevance_post must be an integer pupil-by-posting matrix")
    quota = shown_post.sum(axis=1).astype(int)
    demand = int(quota.sum())
    if demand == 0:
        return np.zeros_like(E)
    scores = [int(v) for v in relevance[E_post]]
    lo, hi = min(scores), max(scores)
    multiplier = demand * (hi - lo) + 1
    n, p = E_post.shape
    source, sink = 0, n + p + 1
    graph = [[] for _ in range(sink + 1)]
    rng = np.random.default_rng(seed)
    row_order, post_order = rng.permutation(n), rng.permutation(p)
    row_node, post_node = np.empty(n, int), np.empty(p, int)
    row_node[row_order] = np.arange(1, n + 1)
    post_node[post_order] = np.arange(n + 1, n + p + 1)
    selected = []
    for i in row_order:
        _add(graph, source, int(row_node[i]), int(quota[i]), 0)
        if not quota[i]:
            continue
        for j in post_order:
            if E_post[i, j]:
                edge = _add(graph, int(row_node[i]), int(post_node[j]), 1,
                            -(int(relevance[i, j]) - lo))
                selected.append((int(i), int(j), edge))
    for j in post_order:
        possible = int((E_post[:, j] & (quota > 0)).sum())
        for load in range(1, possible + 1):
            _add(graph, int(post_node[j]), sink, 1, multiplier * (2 * load - 1))
    _send_flow(graph, source, sink, demand)
    out = np.zeros_like(E_post)
    for i, j, edge in selected:
        out[i, j] = edge.cap == 0
    if not np.array_equal(out.sum(axis=1), quota):
        raise ArithmeticError("flow failed to preserve posting quotas")
    return out[:, s2p] & E
