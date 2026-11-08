"""Audit-selection methods (slides 6, 13-19).

Every method returns a boolean audit vector whose DA outcome has been checked against the full market. The exact method is
logic-based Benders: a HiGHS master picks the cheapest audits that support a stable matching, DA checks them, and each failure
adds a cut. Three cut families are available: "assignment", "component" and "prefix".
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import highspy
import numpy as np

from .matching import Instance, blocking_mask, deferred_acceptance, prefix_mask


class Oracle:
    """DA plus the full-market check for one (instance, E0), memoized on the audit vector with a memory limit."""

    def __init__(self, inst: Instance, e0: np.ndarray, cache_mb: int = 200):
        if (e0 & ~inst.E).any():
            raise ValueError("E0 must be a subset of E")
        self.inst, self.e0 = inst, e0
        self.calls = 0
        self._cache: dict[bytes, tuple] = {}
        self._max_entries = max(8, cache_mb * 10**6 // (inst.n_b * inst.n_s + 8 * (inst.n_b + inst.n_s)))

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = np.asarray(x, bool).tobytes()
        hit = self._cache.get(key)
        if hit is None:
            self.calls += 1
            mb, ms = deferred_acceptance(self.inst, self.inst.visible(x, self.e0))
            hit = (mb, ms, blocking_mask(self.inst, mb, ms))
            if len(self._cache) >= self._max_entries:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = hit
        return hit

    def stable(self, x: np.ndarray) -> bool:
        return not self.evaluate(x)[2].any()


@dataclass
class PlanResult:
    method: str
    x: np.ndarray
    cost: int
    certified: bool  # True only if this method proves optimality
    seconds: float
    lower_bound: int | None = None
    support_cost: int | None = None
    master_solves: int = 0
    n_cuts: int = 0
    da_calls: int = 0
    timed_out: bool = False
    extra: dict = field(default_factory=dict)


def _cost(inst: Instance, x: np.ndarray) -> int:
    return int(inst.cost[np.asarray(x, bool)].sum())


# ------------------------------------------------------------------------------------------- baselines
def no_audit(inst: Instance, oracle: Oracle) -> PlanResult:
    x = np.zeros(inst.n_firms, bool)
    return PlanResult("no_audit", x, 0, False, 0.0, da_calls=1, extra={"stable": oracle.stable(x)})


def full_disclosure(inst: Instance, oracle: Oracle) -> PlanResult:
    x = np.ones(inst.n_firms, bool)
    if not oracle.stable(x):
        raise AssertionError("DA on E must be stable on E")
    return PlanResult("full_disclosure", x, _cost(inst, x), False, 0.0, da_calls=1)


def reverse_delete(inst: Instance, oracle: Oracle, order: np.ndarray | None = None) -> PlanResult:
    """Start with every firm. Keep a deletion only if DA still passes.

    Default order: firms opening the fewest hidden pairs per unit cost go first, so large portfolios are kept when possible.
    """
    t0, calls0 = time.perf_counter(), oracle.calls
    x = np.ones(inst.n_firms, bool)
    if order is None:
        hidden = inst.E & ~oracle.e0
        opened = np.bincount(inst.firm_b, weights=hidden.sum(1), minlength=inst.n_firms)
        opened += np.bincount(inst.firm_s, weights=hidden.sum(0), minlength=inst.n_firms)
        order = np.argsort(opened / inst.cost, kind="stable")
    for f in order:
        x[f] = False
        if not oracle.stable(x):
            x[f] = True
    return PlanResult("reverse_delete", x.copy(), _cost(inst, x), False, time.perf_counter() - t0,
                      da_calls=oracle.calls - calls0)


def blocker_greedy(inst: Instance, oracle: Oracle) -> PlanResult:
    """Add the audit with the greatest immediate blocker reduction per unit cost.

    When no single audit reduces blockers, add the firm that opens the most hidden pairs per unit cost, so the loop always ends.
    """
    t0, calls0 = time.perf_counter(), oracle.calls
    x = np.zeros(inst.n_firms, bool)
    while True:
        blockers = int(oracle.evaluate(x)[2].sum())
        if blockers == 0:
            break
        vis, best, best_gain, best_open = inst.visible(x, oracle.e0), None, 0.0, None
        for f in np.nonzero(~x)[0]:
            y = x.copy()
            y[f] = True
            gain = (blockers - int(oracle.evaluate(y)[2].sum())) / inst.cost[f]
            opened = (inst.visible(y, oracle.e0) & ~vis).sum() / inst.cost[f]
            if best is None or (gain, opened) > (best_gain, best_open):
                best, best_gain, best_open = f, gain, opened
        x[best] = True
    return PlanResult("blocker_greedy", x, _cost(inst, x), False, time.perf_counter() - t0, da_calls=oracle.calls - calls0)


def exhaustive(inst: Instance, oracle: Oracle, max_firms: int = 20) -> PlanResult | None:
    """Cheapest passing audit vector by enumeration. Returns None when there are too many firms."""
    if inst.n_firms > max_firms:
        return None
    t0, calls0 = time.perf_counter(), oracle.calls
    vectors = sorted(itertools.product((False, True), repeat=inst.n_firms), key=lambda v: int(inst.cost[list(v)].sum()))
    for v in vectors:
        x = np.array(v, bool)
        if oracle.stable(x):
            return PlanResult("exhaustive", x, _cost(inst, x), True, time.perf_counter() - t0, da_calls=oracle.calls - calls0)
    raise AssertionError("auditing every firm always passes")


# ------------------------------------------------------------------------------------------- HiGHS
class _Mip:
    """Thin wrapper on one HiGHS model. One thread and a fixed seed, as in Appendix 12."""

    def __init__(self, cost, integer, lower, upper, rows, seed=0):
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("threads", 1)
        self.h.setOptionValue("random_seed", int(seed))
        self.h.setOptionValue("mip_abs_gap", 0.999)  # objective coefficients are integers
        self.h.setOptionValue("mip_rel_gap", 0.0)
        lp = highspy.HighsLp()
        n = len(cost)
        lp.num_col_, lp.num_row_ = n, len(rows)
        lp.col_cost_ = np.asarray(cost, float)
        lp.col_lower_, lp.col_upper_ = np.asarray(lower, float), np.asarray(upper, float)
        lp.row_lower_ = np.array([r[0] for r in rows], float)
        lp.row_upper_ = np.array([r[1] for r in rows], float)
        start, index, value = [0], [], []
        for _, _, idx, val in rows:
            index.extend(idx)
            value.extend(val)
            start.append(len(index))
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_ = np.array(start, np.int32)
        lp.a_matrix_.index_ = np.array(index, np.int32)
        lp.a_matrix_.value_ = np.array(value, float)
        lp.integrality_ = [highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous for i in integer]
        self.h.passModel(lp)

    def add_row(self, lb, ub, idx, val):
        self.h.addRow(float(lb), float(ub), len(idx), np.array(idx, np.int32), np.array(val, float))

    def solve(self, time_limit: float) -> tuple[str, np.ndarray | None, float]:
        """Returns (status, column values or None, valid lower bound on the objective)."""
        self.h.setOptionValue("time_limit", float(self.h.getRunTime() + max(time_limit, 0.01)))
        self.h.run()
        status = self.h.getModelStatus()
        info = self.h.getInfo()
        bound = info.mip_dual_bound
        bound = -math.inf if bound is None or not math.isfinite(bound) else float(bound)
        if status == highspy.HighsModelStatus.kOptimal:
            return "optimal", np.array(self.h.getSolution().col_value), max(bound, -math.inf)
        if status in (highspy.HighsModelStatus.kInfeasible, highspy.HighsModelStatus.kUnboundedOrInfeasible):
            return "infeasible", None, math.inf  # the objective is bounded below, so this always means infeasible
        return "timeout", None, bound


class SupportMaster:
    """The support master of Appendix 2 plus learned cuts.

    Columns: x_f (binary), v_e (hidden pairs only), z_e (every pair in E).
    """

    def __init__(self, inst: Instance, e0: np.ndarray, seed: int = 0, support: bool = True):
        self.inst, self.e0 = inst, e0
        self.edges = [(int(b), int(s)) for b, s in zip(*np.nonzero(inst.E))]
        self.hidden = [e for e in self.edges if not e0[e]]
        F = inst.n_firms
        self.vi = {e: F + i for i, e in enumerate(self.hidden)}
        z0 = F + len(self.hidden)
        self.zi = {e: z0 + i for i, e in enumerate(self.edges)}
        n = z0 + len(self.edges)
        cost = np.zeros(n)
        cost[:F] = inst.cost
        lower, upper = np.zeros(n), np.ones(n)
        integer = np.zeros(n, bool)
        integer[:F] = True
        rows = []
        for e, v in self.vi.items():  # the status of each hidden pair follows the audits
            fb, fs = inst.firm_b[e[0]], inst.firm_s[e[1]]
            rows.append((0.0, math.inf, [v, int(fb)], [1.0, -1.0]))
            rows.append((0.0, math.inf, [v, int(fs)], [1.0, -1.0]))
            rows.append((-math.inf, 0.0, [v, int(fb), int(fs)] if fb != fs else [v, int(fb)],
                         [1.0, -1.0, -1.0] if fb != fs else [1.0, -1.0]))
        if support:
            by_b, by_s = {}, {}
            for e in self.edges:
                by_b.setdefault(e[0], []).append(self.zi[e])
                by_s.setdefault(e[1], []).append(self.zi[e])
            for group in (*by_b.values(), *by_s.values()):
                rows.append((-math.inf, 1.0, group, [1.0] * len(group)))
            for b, s in self.edges:  # stable-matching polytope: bs does not block
                idx = [self.zi[(b, s)]]
                idx += [self.zi[(b, int(t))] for t in np.nonzero(inst.rank_b[b] < inst.rank_b[b, s])[0]]
                idx += [self.zi[(int(a), s)] for a in np.nonzero(inst.rank_s[:, s] < inst.rank_s[b, s])[0]]
                rows.append((1.0, math.inf, idx, [1.0] * len(idx)))
            for e, v in self.vi.items():  # a matching may only use visible pairs
                rows.append((-math.inf, 0.0, [self.zi[e], v], [1.0, -1.0]))
        else:
            upper[z0:] = 0.0
        self.F = F
        self.mip = _Mip(cost, integer, lower, upper, rows, seed)

    def solve(self, time_limit: float):
        status, sol, bound = self.mip.solve(time_limit)
        x = None if sol is None else sol[: self.F] > 0.5
        return status, x, bound

    def add_status_cut(self, edges, vis) -> None:
        """Σ_{e hidden now} v_e + Σ_{e open now} (1 - v_e) >= 1."""
        idx, val, opened = [], [], 0
        for e in edges:
            idx.append(self.vi[e])
            if vis[e]:
                val.append(-1.0)
                opened += 1
            else:
                val.append(1.0)
        self.mip.add_row(1 - opened, math.inf, idx, val)

    def add_nogood(self, firms, x) -> None:
        """Change at least one of these firm decisions."""
        idx, val, on = [], [], 0
        for f in firms:
            idx.append(int(f))
            if x[f]:
                val.append(-1.0)
                on += 1
            else:
                val.append(1.0)
        self.mip.add_row(1 - on, math.inf, idx, val)

    def add_budget(self, budget: float) -> None:
        idx = list(range(self.F))
        self.mip.add_row(-math.inf, float(budget), idx, [float(c) for c in self.inst.cost])


def _failed_components(inst: Instance, blockers: np.ndarray) -> list[int]:
    return sorted({int(inst.component_id[b]) for b in np.nonzero(blockers.any(1))[0]})


def _add_cuts(master: SupportMaster, inst: Instance, oracle: Oracle, x: np.ndarray, cuts: str) -> int:
    match_b, _, blockers = oracle.evaluate(x)
    if cuts == "assignment":
        master.add_nogood(range(inst.n_firms), x)
        return 1
    vis = inst.visible(x, oracle.e0)
    prefix = prefix_mask(inst, match_b) & ~oracle.e0
    added = 0
    for c in _failed_components(inst, blockers):
        renters = inst.component_id == c
        if cuts == "component":
            listings = inst.E[renters].any(0)
            firms = sorted(set(inst.firm_b[renters].tolist()) | set(inst.firm_s[listings].tolist()))
            master.add_nogood(firms, x)
        else:
            mask = np.zeros_like(prefix)
            mask[renters] = prefix[renters]
            edges = [(int(b), int(s)) for b, s in zip(*np.nonzero(mask))]
            if not edges:
                raise AssertionError("a blocking pair lies in the prefix, so the failed prefix has a hidden pair")
            master.add_status_cut(edges, vis)
        added += 1
    return added


def exact(inst: Instance, oracle: Oracle, cuts: str = "prefix", time_cap: float = 15.0, seed: int = 0,
          incumbent: PlanResult | None = None, support: bool = True) -> PlanResult:
    """Cheapest audits such that DA is full-market stable. A timeout returns a verified plan and a gap."""
    t0, calls0 = time.perf_counter(), oracle.calls
    best = incumbent or reverse_delete(inst, oracle)
    best_x, upper = best.x.copy(), best.cost
    master = SupportMaster(inst, oracle.e0, seed, support)
    lower, support_cost, solves, n_cuts, timed_out, certified = 0, None, 0, 0, False, False
    master_infeasible = False
    while True:
        if lower >= upper:
            certified = True
            break
        remaining = time_cap - (time.perf_counter() - t0)
        if remaining <= 0:
            timed_out = True
            break
        status, x, bound = master.solve(remaining)
        solves += 1
        if status == "infeasible":  # valid cuts never remove a passing plan, so this signals numerics: do not certify
            master_infeasible = True
            break
        if math.isfinite(bound):
            lower = max(lower, math.ceil(bound - 1e-6))
        if status == "timeout":
            timed_out = True
            break
        cost = _cost(inst, x)
        lower = max(lower, cost)
        if solves == 1 and support:
            support_cost = cost
        if oracle.stable(x):
            if cost < upper:
                best_x, upper = x.copy(), cost
            certified = True
            lower = upper
            break
        n_cuts += _add_cuts(master, inst, oracle, x, cuts)
    if not oracle.stable(best_x):
        raise AssertionError("returned plan failed the full-market check")
    return PlanResult(f"exact_{cuts}", best_x, upper, certified, time.perf_counter() - t0, lower_bound=int(min(lower, upper)),
                      support_cost=support_cost, master_solves=solves, n_cuts=n_cuts, da_calls=oracle.calls - calls0,
                      timed_out=timed_out, extra={"master_infeasible": master_infeasible})


def support_cost(inst: Instance, e0: np.ndarray, time_cap: float = 60.0) -> int | None:
    """C_support: the cheapest plan making some full-market stable matching available."""
    status, x, _ = SupportMaster(inst, e0).solve(time_cap)
    return None if status != "optimal" else _cost(inst, x)


def enumerate_optimal_plans(inst: Instance, oracle: Oracle, optimum: int, cap: int = 200, time_cap: float = 30.0,
                            seed: int = 0) -> tuple[list[np.ndarray], bool]:
    """All verified plans of cost `optimum`, up to `cap`. The flag is True when the list is known to be complete."""
    t0 = time.perf_counter()
    master = SupportMaster(inst, oracle.e0, seed)
    master.add_budget(optimum)
    plans: list[np.ndarray] = []
    while len(plans) < cap:
        remaining = time_cap - (time.perf_counter() - t0)
        if remaining <= 0:
            return plans, False
        status, x, _ = master.solve(remaining)
        if status == "infeasible":
            return plans, True
        if status == "timeout":
            return plans, False
        if oracle.stable(x):
            plans.append(x.copy())
            master.add_nogood(range(inst.n_firms), x)
        else:
            _add_cuts(master, inst, oracle, x, "prefix")
    return plans, False


def prefix_cover(inst: Instance, oracle: Oracle) -> PlanResult:
    """Expose every hidden pair in the prefix of DA(E): a minimum-cost vertex cover on endpoint firms."""
    t0, calls0 = time.perf_counter(), oracle.calls
    match_b, _, _ = oracle.evaluate(np.ones(inst.n_firms, bool))
    need = prefix_mask(inst, match_b) & ~oracle.e0
    pairs = {(int(inst.firm_b[b]), int(inst.firm_s[s])) for b, s in zip(*np.nonzero(need))}
    rows = [(1.0, math.inf, [a, c] if a != c else [a], [1.0, 1.0] if a != c else [1.0]) for a, c in sorted(pairs)]
    x = np.zeros(inst.n_firms, bool)
    if rows:
        mip = _Mip(inst.cost, np.ones(inst.n_firms, bool), np.zeros(inst.n_firms), np.ones(inst.n_firms), rows)
        status, sol, _ = mip.solve(60.0)
        if status != "optimal":
            raise RuntimeError("the prefix-cover vertex cover did not solve")
        x = sol > 0.5
    if not oracle.stable(x):
        raise AssertionError("exposing the DA(E) prefix must replay DA(E)")
    return PlanResult("prefix_cover", x, _cost(inst, x), False, time.perf_counter() - t0, da_calls=oracle.calls - calls0)
