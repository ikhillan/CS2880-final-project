"""Instances, deferred acceptance and full-market stability checks (slides 3-5, 14).

Renters are the proposing side ("buyers"), listings the receiving side ("sellers"). Ranks are 0 = best.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NOT_EDGE = 1_000_000  # rank stored for a pair outside E
UNMATCHED = NOT_EDGE - 1  # sorts after every acceptable partner, as in the slides' ⊥


@dataclass
class Instance:
    """A market: E, strict rankings, firms and costs. E0 lives outside so exposures can share one instance."""
    rank_b: np.ndarray  # n_b x n_s, renter's rank of each listing (NOT_EDGE outside E)
    rank_s: np.ndarray  # n_b x n_s, listing's rank of each renter (NOT_EDGE outside E)
    firm_b: np.ndarray  # firm index of each renter
    firm_s: np.ndarray  # firm index of each listing
    cost: np.ndarray  # audit cost per firm
    firm_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.E = self.rank_b < NOT_EDGE
        if not np.array_equal(self.E, self.rank_s < NOT_EDGE):
            raise ValueError("rank_b and rank_s must have the same support")
        self.n_b, self.n_s = self.E.shape
        self.n_firms = len(self.cost)
        self.prefs_b = [np.argsort(row)[: int(k)].tolist() for row, k in zip(self.rank_b, self.E.sum(1))]
        self.rank_s_lists = self.rank_s.T.tolist()  # plain lists are much faster inside the DA loop
        self.list_len_b = self.E.sum(1)
        self.component_id, self.n_components = _components(self.E)

    def visible(self, audited: np.ndarray, e0: np.ndarray) -> np.ndarray:
        """E(A) = E0 ∪ {bs ∈ E : f(b) ∈ A or f(s) ∈ A}."""
        audited = np.asarray(audited, bool)
        return e0 | (self.E & (audited[self.firm_b][:, None] | audited[self.firm_s][None, :]))


def _components(E: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components of the bipartite graph E. Returns one component id per renter (-1 if isolated)."""
    n_b, n_s = E.shape
    parent = list(range(n_b + n_s))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for b, s in zip(*np.nonzero(E)):
        ra, rb = find(int(b)), find(n_b + int(s))
        if ra != rb:
            parent[ra] = rb
    ids, out = {}, np.full(n_b, -1)
    has_edge = E.any(1)
    for b in range(n_b):
        if has_edge[b]:
            out[b] = ids.setdefault(find(b), len(ids))
    return out, len(ids)


def deferred_acceptance(inst: Instance, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renter-proposing DA on the visible graph. Returns (match_b, match_s), with -1 for unmatched."""
    vis, prefs, rs = visible.tolist(), inst.prefs_b, inst.rank_s_lists
    match_b, match_s, next_i = [-1] * inst.n_b, [-1] * inst.n_s, [0] * inst.n_b
    free = list(range(inst.n_b))
    while free:
        b = free.pop()
        pl, row, i = prefs[b], vis[b], next_i[b]
        while i < len(pl):
            s = pl[i]
            i += 1
            if not row[s]:
                continue
            cur = match_s[s]
            if cur == -1:
                match_s[s], match_b[b] = b, s
                break
            if rs[s][b] < rs[s][cur]:
                match_s[s], match_b[b], match_b[cur] = b, s, -1
                free.append(cur)
                break
        next_i[b] = i
    return np.array(match_b, dtype=np.int32), np.array(match_s, dtype=np.int32)


def current_ranks(inst: Instance, match_b: np.ndarray, match_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rb = np.full(inst.n_b, UNMATCHED, dtype=np.int64)
    rs = np.full(inst.n_s, UNMATCHED, dtype=np.int64)
    bs = np.nonzero(match_b >= 0)[0]
    rb[bs] = inst.rank_b[bs, match_b[bs]]
    ss = np.nonzero(match_s >= 0)[0]
    rs[ss] = inst.rank_s[match_s[ss], ss]
    return rb, rs


def blocking_mask(inst: Instance, match_b: np.ndarray, match_s: np.ndarray) -> np.ndarray:
    """BP_E(M): pairs in E \\ M where both sides strictly prefer each other to their current partner."""
    rb, rs = current_ranks(inst, match_b, match_s)
    return inst.E & (inst.rank_b < rb[:, None]) & (inst.rank_s < rs[None, :])


def prefix_mask(inst: Instance, match_b: np.ndarray) -> np.ndarray:
    """Π(M) = {bs ∈ E : M(b) = ⊥ or s ⪰_b M(b)}. An unmatched renter contributes its whole list."""
    rb = np.full(inst.n_b, UNMATCHED, dtype=np.int64)
    bs = np.nonzero(match_b >= 0)[0]
    rb[bs] = inst.rank_b[bs, match_b[bs]]
    return inst.E & (inst.rank_b <= rb[:, None])


def is_full_market_stable(inst: Instance, match_b: np.ndarray, match_s: np.ndarray) -> bool:
    return not blocking_mask(inst, match_b, match_s).any()
