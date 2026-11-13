"""Exact offline disclosure planning with deferred-acceptance certificates.

All algorithmic claims concern explicit strict preferences and fixed portfolios.
Scipy's MILP solver is used only for the audit master; every proposed outcome
is checked by an independent deferred-acceptance implementation.

Only audit variables x are integer. For an optional edge between firms f,g,
0 <= y <= 1, y >= x_f, y >= x_g, and y <= x_f+x_g force y=0 when both
audits are zero and y=1 when either is one; initially disclosed edges fix y=1.
Thus continuous exposure variables are exact once x is binary. For each such
exposure vector, z_e <= y_e fixes prohibited stable-matching coordinates to
zero, defining an integral face of the strict stable-matching polytope; z may
therefore also remain continuous in the stable-support relaxation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import isfinite
from time import monotonic
from typing import Iterable
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def _run_milp(*args, **kwargs):
    """Set HiGHS to one thread; SciPy forwards this supported HiGHS option."""
    kwargs["options"] = {**kwargs.get("options", {}), "threads": 1}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*threads.*",
                                category=RuntimeWarning)
        return milp(*args, **kwargs)


@dataclass
class Market:
    buyer_prefs: tuple[tuple[int, ...], ...]
    seller_prefs: tuple[tuple[int, ...], ...]
    buyer_firms: tuple[int, ...]
    seller_firms: tuple[int, ...]
    costs: tuple[float, ...]
    initial_edges: frozenset[tuple[int, int]]
    name: str = "market"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.buyer_prefs = tuple(tuple(map(int, p)) for p in self.buyer_prefs)
        self.seller_prefs = tuple(tuple(map(int, p)) for p in self.seller_prefs)
        self.buyer_firms = tuple(map(int, self.buyer_firms))
        self.seller_firms = tuple(map(int, self.seller_firms))
        self.costs = tuple(map(float, self.costs))
        self.initial_edges = frozenset(tuple(map(int, e)) for e in self.initial_edges)
        self.n_buyers, self.n_sellers = len(self.buyer_prefs), len(self.seller_prefs)
        self.n_firms = len(self.costs)
        assert len(self.buyer_firms) == self.n_buyers
        assert len(self.seller_firms) == self.n_sellers
        assert all(isfinite(c) and c > 0 for c in self.costs)
        assert all(0 <= f < self.n_firms for f in self.buyer_firms + self.seller_firms)
        assert all(len(p) == len(set(p)) for p in self.buyer_prefs + self.seller_prefs)
        be = {(b, s) for b, p in enumerate(self.buyer_prefs) for s in p}
        se = {(b, s) for s, p in enumerate(self.seller_prefs) for b in p}
        assert be == se, "Acceptability must be mutual"
        assert all(0 <= b < self.n_buyers and 0 <= s < self.n_sellers for b, s in be)
        assert self.initial_edges <= be
        self.edges = tuple(sorted(be))
        self.edge_index = {e: k for k, e in enumerate(self.edges)}
        self.buyer_rank = tuple({s: k for k, s in enumerate(p)} for p in self.buyer_prefs)
        self.seller_rank = tuple({b: k for k, b in enumerate(p)} for p in self.seller_prefs)
        parent = list(range(self.n_buyers + self.n_sellers))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for b, s in self.edges:
            rb, rs = find(b), find(self.n_buyers + s)
            parent[rb] = rs
        groups = {}
        for k, (b, s) in enumerate(self.edges):
            groups.setdefault(find(b), []).append(k)
        self.component_edges = tuple(tuple(v) for v in groups.values())
        self.edge_component = {}
        self.component_firms = []
        for j, group in enumerate(self.component_edges):
            firms = set()
            for k in group:
                b, s = self.edges[k]
                self.edge_component[k] = j
                firms.update((self.buyer_firms[b], self.seller_firms[s]))
            self.component_firms.append(tuple(sorted(firms)))
        self.component_firms = tuple(self.component_firms)

    def to_dict(self):
        return dict(buyer_prefs=self.buyer_prefs, seller_prefs=self.seller_prefs,
                    buyer_firms=self.buyer_firms, seller_firms=self.seller_firms,
                    costs=self.costs, initial_edges=sorted(self.initial_edges),
                    name=self.name, metadata=self.metadata)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def cost(self, mask: int) -> float:
        return sum(c for f, c in enumerate(self.costs) if mask & (1 << f))

    def exposure(self, mask: int) -> tuple[bool, ...]:
        return tuple(e in self.initial_edges or
                     bool(mask & ((1 << self.buyer_firms[e[0]]) |
                                  (1 << self.seller_firms[e[1]])))
                     for e in self.edges)


@dataclass(frozen=True)
class DAOutcome:
    buyer_match: tuple[int, ...]
    seller_match: tuple[int, ...]
    proposals: tuple[int, ...]  # Full-market edge indices, chronological order.


def deferred_acceptance(market: Market, exposed: Iterable[bool]) -> DAOutcome:
    """Buyer-proposing DA; unmatched clients are represented by -1."""
    exposed = tuple(exposed)
    assert len(exposed) == len(market.edges)
    bm = [-1] * market.n_buyers
    sm = [-1] * market.n_sellers
    nxt = [0] * market.n_buyers
    free = deque(range(market.n_buyers))
    proposals = []
    while free:
        b = free.popleft()
        prefs = market.buyer_prefs[b]
        while nxt[b] < len(prefs):
            s = prefs[nxt[b]]
            nxt[b] += 1
            k = market.edge_index[b, s]
            if not exposed[k]:
                continue
            proposals.append(k)
            incumbent = sm[s]
            if incumbent == -1 or market.seller_rank[s][b] < market.seller_rank[s][incumbent]:
                if incumbent != -1:
                    bm[incumbent] = -1
                    free.append(incumbent)
                bm[b], sm[s] = s, b
                break
            # An immediate rejection permits the same buyer's next proposal.
    return DAOutcome(tuple(bm), tuple(sm), tuple(proposals))


def blocking_pairs(market: Market, out: DAOutcome) -> tuple[int, ...]:
    blocks = []
    for k, (b, s) in enumerate(market.edges):
        old_s, old_b = out.buyer_match[b], out.seller_match[s]
        if old_s == s:
            continue
        if (old_s == -1 or market.buyer_rank[b][s] < market.buyer_rank[b][old_s]) and (
            old_b == -1 or market.seller_rank[s][b] < market.seller_rank[s][old_b]
        ):
            blocks.append(k)
    return tuple(blocks)


def proposal_prefix(market: Market, out: DAOutcome, component: int | None = None) -> tuple[int, ...]:
    """All feasible edges weakly above each buyer's final assignment."""
    candidates = range(len(market.edges)) if component is None else market.component_edges[component]
    return tuple(k for k in candidates
                 if out.buyer_match[market.edges[k][0]] == -1 or
                 market.buyer_rank[market.edges[k][0]][market.edges[k][1]] <=
                 market.buyer_rank[market.edges[k][0]][out.buyer_match[market.edges[k][0]]])


def evaluate(market: Market, mask: int):
    exposure = market.exposure(mask)
    out = deferred_acceptance(market, exposure)
    return out, blocking_pairs(market, out)


def reverse_delete(market: Market, time_limit: float = float("inf")) -> dict:
    """Feasible upper bound; repeated one-deletion local search, not an approximation guarantee."""
    start = monotonic()
    mask = (1 << market.n_firms) - 1
    evaluations = 0
    changed = True
    while changed:
        changed = False
        for f in sorted(range(market.n_firms), key=lambda j: (-market.costs[j], j)):
            if monotonic() - start >= time_limit:
                return dict(mask=mask, cost=market.cost(mask), evaluations=evaluations)
            if not mask & (1 << f):
                continue
            candidate = mask & ~(1 << f)
            _, blocks = evaluate(market, candidate)
            evaluations += 1
            if not blocks:
                mask, changed = candidate, True
    return dict(mask=mask, cost=market.cost(mask), evaluations=evaluations)


def exhaustive_optimum(market: Market, max_firms: int = 14, time_limit: float = float("inf")) -> dict:
    """Independent enumeration oracle; inspect masks in increasing total cost."""
    assert market.n_firms <= max_firms
    start = monotonic()
    evaluations = 0
    for mask in sorted(range(1 << market.n_firms), key=lambda a: (market.cost(a), a)):
        if monotonic() - start >= time_limit:
            return dict(status="timeout", optimal=False, elapsed=monotonic()-start,
                        evaluations=evaluations, mask=None, cost=None)
        _, blocks = evaluate(market, mask)
        evaluations += 1
        if not blocks:
            return dict(status="optimal", optimal=True, elapsed=monotonic()-start,
                        evaluations=evaluations, mask=mask, cost=market.cost(mask))
    raise AssertionError("Full disclosure must be feasible")


class _Master:
    def __init__(self, market: Market, stable_support: bool):
        self.market = market
        p, m = market.n_firms, len(market.edges)
        self.p, self.m, self.nvars = p, m, p + m + (m if stable_support else 0)
        self.objective = np.zeros(self.nvars)
        self.objective[:p] = market.costs
        # For each integral exposure vector, prohibited z coordinates are zero;
        # the remaining stable-matching polytope is an integral face. Continuous
        # z therefore gives the exact stable-support projection in x/y.
        self.integrality = np.zeros(self.nvars)
        self.integrality[:p] = 1
        self.lower = np.zeros(self.nvars)
        self.upper = np.ones(self.nvars)
        self.rows = []
        self.row_lower = []
        self.row_upper = []
        self.cut_keys = set()
        for k, (b, s) in enumerate(market.edges):
            y = p + k
            f, g = market.buyer_firms[b], market.seller_firms[s]
            if (b, s) in market.initial_edges:
                self.lower[y] = 1
            else:
                for endpoint_firm in {f, g}:
                    self.add_row({endpoint_firm: 1, y: -1}, -np.inf, 0)
                row = {y: 1}
                for endpoint_firm in {f, g}:
                    row[endpoint_firm] = row.get(endpoint_firm, 0) - 1
                self.add_row(row, -np.inf, 0)
            if stable_support:
                self.add_row({p + m + k: 1, y: -1}, -np.inf, 0)
        if stable_support:
            z = lambda k: p + m + k
            for b, prefs in enumerate(market.buyer_prefs):
                self.add_row({z(market.edge_index[b, s]): 1 for s in prefs}, -np.inf, 1)
            for s, prefs in enumerate(market.seller_prefs):
                self.add_row({z(market.edge_index[b, s]): 1 for b in prefs}, -np.inf, 1)
            for k, (b, s) in enumerate(market.edges):
                row = {z(k): 1}
                for s2 in market.buyer_prefs[b][:market.buyer_rank[b][s]]:
                    row[z(market.edge_index[b, s2])] = 1
                for b2 in market.seller_prefs[s][:market.seller_rank[s][b]]:
                    row[z(market.edge_index[b2, s])] = 1
                self.add_row(row, 1, np.inf)

    def add_row(self, row, lower, upper):
        row = {int(k): float(v) for k, v in row.items() if v}
        if not row:
            assert lower <= 0 <= upper
            return
        self.rows.append(row)
        self.row_lower.append(float(lower))
        self.row_upper.append(float(upper))

    def nogood(self, variables: Iterable[int], values: Iterable[bool]) -> bool:
        pairs = dict(zip(variables, values))
        assert pairs, "An unstable market must admit an outcome-changing exposure"
        row = {k: -1 if v else 1 for k, v in pairs.items()}
        lower = 1 - sum(bool(v) for v in pairs.values())
        key = tuple(sorted(row.items())), lower
        if key in self.cut_keys:
            return False
        self.cut_keys.add(key)
        self.add_row(row, lower, np.inf)
        return True

    def solve(self, time_limit: float, upper_bound: float):
        rows = self.rows + [{f: c for f, c in enumerate(self.market.costs)}]
        lower = self.row_lower + [-np.inf]
        upper = self.row_upper + [upper_bound + 1e-7]
        ri, ci, values = [], [], []
        for i, row in enumerate(rows):
            for j, value in row.items():
                ri.append(i); ci.append(j); values.append(value)
        matrix = coo_matrix((values, (ri, ci)), shape=(len(rows), self.nvars)).tocsc()
        return _run_milp(self.objective, integrality=self.integrality,
                    bounds=Bounds(self.lower, self.upper),
                    constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
                    options={"time_limit": max(0.001, float(time_limit)), "mip_rel_gap": 0.0,
                             "presolve": True, "disp": False})


def solve_disclosure(market: Market, variant: str = "prefix", stable_support: bool = True,
                     time_limit: float = 60.0, warm_start: str = "reverse_delete",
                     max_rounds: int | None = None) -> dict:
    """Exact certificate-generation solver with valid timeout bounds.

    variant: 'assignment', 'component', or 'prefix'. Every learned constraint is
    necessary for success. optimal=True is returned only after a matching lower
    bound is established, or an optimal master solution passes the DA oracle.
    """
    assert variant in {"assignment", "component", "prefix"}
    assert warm_start in {"full", "reverse_delete"}
    start = monotonic()
    best_mask = (1 << market.n_firms) - 1
    best_cost = market.cost(best_mask)
    evaluations = 0
    if warm_start == "reverse_delete":
        warm = reverse_delete(market, time_limit=max(0.0, min(time_limit * 0.1, 2.0)))
        best_mask, best_cost = warm["mask"], warm["cost"]
        evaluations += warm["evaluations"]
    lower_bound = 0.0
    log = []
    master = _Master(market, stable_support)
    status, optimal = "timeout", False
    while monotonic() - start < time_limit:
        if max_rounds is not None and len(log) >= max_rounds:
            status = "round_limit"
            break
        if best_cost <= lower_bound + 1e-7:
            status, optimal = "optimal", True
            break
        remaining = time_limit - (monotonic() - start)
        solve_start = monotonic()
        result = master.solve(remaining, best_cost)
        solve_time = monotonic() - solve_start
        dual = getattr(result, "mip_dual_bound", None)
        bound_inconsistent = False
        if dual is not None and isfinite(float(dual)):
            if float(dual) > best_cost + 1e-6:
                bound_inconsistent = True
            else:
                lower_bound = max(lower_bound, min(float(dual), best_cost))
        if result.status == 2:
            raise AssertionError("Master infeasible despite a verified feasible upper bound")
        record = dict(round=len(log)+1, milp_status=int(result.status),
                      milp_time=solve_time, lower_bound=lower_bound,
                      upper_bound=best_cost, candidate_cost=None, blockers=None,
                      cuts_added=0, constraints=len(master.rows), variables=master.nvars,
                      bound_inconsistent=bound_inconsistent)
        if bound_inconsistent:
            log.append(record)
            status = "solver_failure"
            break
        if result.x is None:
            log.append(record)
            status = "timeout" if result.status == 1 else "solver_failure"
            break
        values = result.x[:market.n_firms]
        if np.max(np.abs(values - np.round(values)), initial=0) > 1e-5:
            log.append(record)
            status = "solver_failure"
            break
        candidate = sum(1 << f for f, v in enumerate(values) if v > 0.5)
        candidate_cost = market.cost(candidate)
        if result.status == 0:
            if candidate_cost > best_cost + 1e-6:
                record["bound_inconsistent"] = True
                log.append(record)
                status = "solver_failure"
                break
            lower_bound = max(lower_bound, min(candidate_cost, best_cost))
        out, blocks = evaluate(market, candidate)
        evaluations += 1
        record.update(candidate_cost=candidate_cost, blockers=len(blocks),
                      lower_bound=lower_bound)
        if not blocks:
            if candidate_cost < best_cost:
                best_mask, best_cost = candidate, candidate_cost
            record["upper_bound"] = best_cost
            log.append(record)
            if best_cost <= lower_bound + 1e-7:
                status, optimal = "optimal", True
            else:
                status = "timeout" if result.status == 1 else "solver_failure"
            break
        exposure = market.exposure(candidate)
        cuts_added = 0
        if variant == "assignment":
            cuts_added += master.nogood(range(market.n_firms),
                                       (bool(candidate & (1 << f)) for f in range(market.n_firms)))
        else:
            unstable_components = sorted({market.edge_component[k] for k in blocks})
            for comp in unstable_components:
                if variant == "component":
                    variables = market.component_firms[comp]
                    values = (bool(candidate & (1 << f)) for f in variables)
                else:
                    prefix = tuple(k for k in proposal_prefix(market, out, comp)
                                   if market.edges[k] not in market.initial_edges)
                    variables = (market.n_firms + k for k in prefix)
                    values = (exposure[k] for k in prefix)
                cuts_added += master.nogood(variables, values)
        assert cuts_added > 0, "No progress: returned assignment violates an existing certificate"
        record["cuts_added"] = cuts_added
        log.append(record)
        if result.status != 0:
            status = "timeout" if result.status == 1 else "solver_failure"
            break
    if status != "solver_failure" and best_cost <= lower_bound + 1e-7:
        status, optimal = "optimal", True
    # Never report an unchecked incumbent, including on timeout.
    assert not evaluate(market, best_mask)[1]
    return dict(name=market.name, variant=variant, stable_support=stable_support,
                status=status, optimal=optimal, mask=best_mask, cost=best_cost,
                lower_bound=lower_bound, upper_bound=best_cost,
                absolute_gap=max(0.0, best_cost-lower_bound),
                elapsed=monotonic()-start, rounds=len(log), cuts=len(master.cut_keys),
                evaluations=evaluations, log=log)


def vertex_cover_market(edges, n_vertices=None, name="vertex_cover_gadget") -> Market:
    edges = tuple(tuple(map(int, e)) for e in edges)
    if n_vertices is None:
        n_vertices = 1 + max((v for e in edges for v in e), default=-1)
    bp, sp, bf, sf, initial = [], [], [], [], []
    for j, (u, v) in enumerate(edges):
        b, t, s, r = 2*j, 2*j+1, 2*j, 2*j+1
        bp.extend(((s, r), (s,)))
        sp.extend(((b, t), (b,)))
        bf.extend((u, n_vertices+2*j))
        sf.extend((v, n_vertices+2*j+1))
        initial.extend(((b, r), (t, s)))
    return Market(bp, sp, bf, sf, [1]*(n_vertices+2*len(edges)), initial, name=name)


def nonmonotone_market() -> Market:
    return Market(((0, 1, 2), (1, 0, 2), (0, 2, 1)),
                  ((1, 2, 0), (0, 1, 2), (2, 0, 1)),
                  (0, 1, 2), (3, 4, 5), (1,)*6,
                  ((0, 1), (1, 0), (2, 2)), name="nonmonotone_three_by_three")


def support_lower_bound(market: Market, time_limit: float = 60.0) -> dict:
    """Optimize stable support only; its mask need not have a stable DA response."""
    start = monotonic()
    master = _Master(market, stable_support=True)
    if not market.edges:
        return dict(status="optimal", optimal=True, mask=0, cost=0.0,
                    lower_bound=0.0, elapsed=monotonic()-start, matching=[])
    result = master.solve(time_limit, sum(market.costs))
    dual = getattr(result, "mip_dual_bound", None)
    bound = max(0.0, float(dual)) if dual is not None and isfinite(float(dual)) else 0.0
    answer = dict(status="optimal" if result.status == 0 else "timeout" if result.status == 1 else "solver_failure",
                  optimal=result.status == 0, mask=None, cost=None, lower_bound=bound,
                  elapsed=monotonic()-start, matching=None)
    if result.x is not None:
        mask = sum(1 << f for f, value in enumerate(result.x[:market.n_firms]) if value > 0.5)
        cost = market.cost(mask)
        offset = market.n_firms + len(market.edges)
        support = result.x[offset:]
        integral_support = np.max(np.abs(support - np.round(support)), initial=0) <= 1e-5
        matching = ([market.edges[k] for k in range(len(market.edges)) if support[k] > 0.5]
                    if integral_support else None)
        answer.update(mask=mask, cost=cost, matching=matching,
                      support_integral=bool(integral_support),
                      support_values=[float(v) for v in support])
        if bound > cost + 1e-6:
            answer.update(status="solver_failure", optimal=False, lower_bound=0.0,
                          bound_inconsistent=True)
        elif result.status == 0:
            answer["lower_bound"] = cost
    return answer


def prefix_cover(market: Market, time_limit: float = 60.0) -> dict:
    """Minimize audit cost subject to exposing the full-market DA proposal prefix.

    This is exact weighted vertex cover for a sufficient certificate, not an
    approximation guarantee for minimum stable disclosure. If the cover solve
    times out, a verified feasible cover and its MILP bound are still returned.
    """
    start = monotonic()
    full = deferred_acceptance(market, (True,) * len(market.edges))
    required = [k for k in proposal_prefix(market, full)
                if market.edges[k] not in market.initial_edges]
    interactions = sorted({tuple(sorted({market.buyer_firms[market.edges[k][0]],
                                         market.seller_firms[market.edges[k][1]]}))
                           for k in required})
    best_mask = (1 << market.n_firms) - 1
    best_cost = market.cost(best_mask)
    if not interactions:
        best_mask, best_cost = 0, 0.0
        result_status, lower, optimal = 0, 0.0, True
    else:
        rows, columns, values = [], [], []
        for j, firms in enumerate(interactions):
            for f in firms:
                rows.append(j); columns.append(f); values.append(1.0)
        matrix = coo_matrix((values, (rows, columns)),
                            shape=(len(interactions), market.n_firms)).tocsc()
        result = _run_milp(np.array(market.costs), integrality=np.ones(market.n_firms),
                      bounds=Bounds(np.zeros(market.n_firms), np.ones(market.n_firms)),
                      constraints=LinearConstraint(matrix, np.ones(len(interactions)),
                                                   np.full(len(interactions), np.inf)),
                      options={"time_limit": max(0.001, time_limit), "mip_rel_gap": 0.0,
                               "presolve": True, "disp": False})
        result_status = int(result.status)
        dual = getattr(result, "mip_dual_bound", None)
        lower = max(0.0, float(dual)) if dual is not None and isfinite(float(dual)) else 0.0
        optimal = result.status == 0
        if result.x is not None:
            candidate = sum(1 << f for f, value in enumerate(result.x) if value > 0.5)
            assert all(any(candidate & (1 << f) for f in firms) for firms in interactions)
            best_mask, best_cost = candidate, market.cost(candidate)
        if lower > best_cost + 1e-6:
            result_status, optimal, lower = 4, False, 0.0
        elif optimal:
            lower = best_cost
    response, blocks = evaluate(market, best_mask)
    assert response.buyer_match == full.buyer_match, "Full-DA prefix certificate violated"
    assert not blocks
    return dict(name=market.name, status="optimal" if optimal else "timeout" if result_status == 1 else "solver_failure",
                optimal=optimal, mask=best_mask, cost=best_cost, lower_bound=lower,
                elapsed=monotonic()-start, prefix_edges=len(required),
                cover_constraints=len(interactions), matching=response.buyer_match,
                bound_scope="prefix_cover_problem")


def switch_market(edges, n_vertices=None, name="side_exclusive_switch") -> Market:
    """K2,2 switch construction: optimum audit cost equals vertex-cover size.

    Every original vertex firm represents buyers only. All seller firms are
    private singletons; the useful seller firm is an edge-specific fallback.
    """
    edges = tuple(tuple(map(int, e)) for e in edges)
    if n_vertices is None:
        n_vertices = 1 + max((v for e in edges for v in e), default=-1)
    bp, sp, bf, sf, initial = [], [], [], [], []
    for j, (u, v) in enumerate(edges):
        p, q, x, y = 2*j, 2*j+1, 2*j, 2*j+1
        bp.extend(((y, x), (x, y)))
        sp.extend(((p, q), (q, p)))
        bf.extend((u, v))
        sf.extend((n_vertices+2*j, n_vertices+2*j+1))
        initial.extend(((p, x), (q, x)))
    return Market(bp, sp, bf, sf, [1]*(n_vertices+2*len(edges)), initial,
                  name=name, metadata={"source_graph": edges, "n_vertices": n_vertices})


def response_gap_market(n: int, name: str | None = None) -> Market:
    """Family with stable-support optimum 2 and stable-DA optimum n+2."""
    assert n >= 1
    m = n + 3  # This many force edges for EACH of the two shared buyer firms.
    bp, sp, bf, sf, initial = [], [], [], [], []
    next_firm = 2 + n  # Shared A=0, B=1; private C_i=2+i.
    for i in range(n):
        a, b, c = 3*i, 3*i+1, 3*i+2
        x, y, z = 3*i, 3*i+1, 3*i+2
        bp.extend(((x, y, z), (y, x, z), (x, z, y)))
        sp.extend(((b, c, a), (a, b, c), (c, a, b)))
        bf.extend((0, 1, 2+i))
        sf.extend((next_firm, next_firm+1, next_firm+2))
        next_firm += 3
        initial.extend(((a, y), (b, x), (c, z)))
    for shared_firm in (0, 1):
        for j in range(m):
            b, s = len(bp), len(sp)
            bp.append((s,)); sp.append((b,))
            bf.append(shared_firm); sf.append(next_firm)
            next_firm += 1
    return Market(bp, sp, bf, sf, [1]*next_firm, initial,
                  name=name or f"response_gap_{n}",
                  metadata={"n_blocks": n, "forcing_edges_per_shared_firm": m,
                            "theoretical_support_optimum": 2,
                            "theoretical_response_optimum": n+2})
