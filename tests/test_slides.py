"""Every worked example in full-paper-slides.pdf, plus exact-vs-enumeration checks on random markets."""
import numpy as np
import pytest

from hiddenoptions.audit import (Oracle, blocker_greedy, enumerate_optimal_plans, exact, exhaustive, full_disclosure,
                                 prefix_cover, reverse_delete, support_cost)
from hiddenoptions.matching import NOT_EDGE, Instance, blocking_mask, deferred_acceptance, prefix_mask


class Builder:
    """Build instances from named preference lists, as the slides write them."""

    def __init__(self):
        self.b, self.s, self.pb, self.ps, self.fb, self.fs = {}, {}, {}, {}, {}, {}
        self.e0, self.firms = [], {}

    def firm(self, name):
        return self.firms.setdefault(name, len(self.firms))

    def renter(self, name, prefs, firm=None):
        self.b[name] = len(self.b)
        self.pb[name], self.fb[name] = prefs, self.firm(firm or f"r:{name}")

    def listing(self, name, prefs, firm=None):
        self.s[name] = len(self.s)
        self.ps[name], self.fs[name] = prefs, self.firm(firm or f"l:{name}")

    def build(self):
        rb = np.full((len(self.b), len(self.s)), NOT_EDGE)
        rs = rb.copy()
        for n, prefs in self.pb.items():
            for i, s in enumerate(prefs):
                rb[self.b[n], self.s[s]] = i
        for n, prefs in self.ps.items():
            for i, b in enumerate(prefs):
                rs[self.b[b], self.s[n]] = i
        inst = Instance(rb, rs, np.array([self.fb[n] for n in self.b]), np.array([self.fs[n] for n in self.s]),
                        np.ones(len(self.firms), int), list(self.firms))
        e0 = np.zeros_like(inst.E)
        for b, s in self.e0:
            e0[self.b[b], self.s[s]] = True
        return inst, e0

    def audit(self, *names):
        x = np.zeros(len(self.firms), bool)
        for n in names:
            x[self.firms[n]] = True
        return x


def add_copy(B, i=0, fa=None, fb=None):
    """Slide 7: a,b,c and x,y,z, with E0 = {ay, bx, cz}."""
    a, b, c, x, y, z = [f"{n}{i}" for n in "abcxyz"]
    B.renter(a, [x, y, z], fa)
    B.renter(b, [y, x, z], fb)
    B.renter(c, [x, z, y])
    B.listing(x, [b, c, a])
    B.listing(y, [a, b, c])
    B.listing(z, [c, a, b])
    B.e0 += [(a, y), (b, x), (c, z)]
    return a, b, c, x, y, z


def pairs(B, inst, mb):
    names_s = {v: k for k, v in B.s.items()}
    return {(n, names_s[int(mb[i])]) for n, i in B.b.items() if mb[i] >= 0}


def test_slide7_success_is_not_monotone():
    B = Builder()
    a, b, c, x, y, z = add_copy(B)
    inst, e0 = B.build()
    o = Oracle(inst, e0)
    # No audits: stable.
    assert pairs(B, inst, o.evaluate(B.audit())[0]) == {(a, y), (b, x), (c, z)}
    assert o.stable(B.audit())
    # Audit a, b: both get first choices and cx blocks.
    ab = B.audit(f"r:{a}", f"r:{b}")
    mb, ms, blockers = o.evaluate(ab)
    assert pairs(B, inst, mb) == {(a, x), (b, y), (c, z)}
    assert blockers[B.b[c], B.s[x]] and blockers.sum() == 1
    # Also audit c: back to the original stable matching.
    abc = B.audit(f"r:{a}", f"r:{b}", f"r:{c}")
    assert pairs(B, inst, o.evaluate(abc)[0]) == {(a, y), (b, x), (c, z)}
    assert o.stable(abc)


@pytest.mark.parametrize("cuts", ["assignment", "component", "prefix"])
@pytest.mark.parametrize("N", [1, 2, 3])
def test_slide9_two_audits_versus_n_plus_two(N, cuts):
    B = Builder()
    for i in range(N):
        add_copy(B, i, "FA", "FB")
    for fam in ("FA", "FB"):
        for j in range(N + 3):  # each shared firm also has separate hidden one-edge markets
            B.renter(f"{fam}b{j}", [f"{fam}s{j}"], fam)
            B.listing(f"{fam}s{j}", [f"{fam}b{j}"])
    inst, e0 = B.build()
    assert support_cost(inst, e0) == 2
    r = exact(inst, Oracle(inst, e0), cuts=cuts, time_cap=30)
    assert r.certified and r.cost == N + 2 and r.support_cost == 2
    assert r.x[B.firms["FA"]] and r.x[B.firms["FB"]]
    if cuts == "prefix":
        assert r.master_solves <= 2  # slide 16: prefix cuts need at most two master solves


def test_slide11_two_by_two_switch():
    def make():
        B = Builder()
        B.renter("p", ["y", "x"], "Fu")
        B.renter("q", ["x", "y"], "Fv")
        B.listing("x", ["p", "q"])
        B.listing("y", ["q", "p"])
        B.e0 += [("p", "x"), ("q", "x")]
        return B
    B = make()
    inst, e0 = B.build()
    o = Oracle(inst, e0)
    none = B.audit()
    assert pairs(B, inst, o.evaluate(none)[0]) == {("p", "x")}
    assert o.evaluate(none)[2][B.b["q"], B.s["y"]]  # qy blocks
    fu, fv, fy = B.audit("Fu"), B.audit("Fv"), B.audit("l:y")
    assert pairs(B, inst, o.evaluate(fu)[0]) == {("p", "y"), ("q", "x")} and o.stable(fu)
    assert pairs(B, inst, o.evaluate(fv)[0]) == {("p", "x"), ("q", "y")} and o.stable(fv)
    assert pairs(B, inst, o.evaluate(fy)[0]) == {("p", "y"), ("q", "x")} and o.stable(fy)
    assert exact(inst, o).cost == 1


def test_slide14_prefix_contains_hidden_better_options():
    B = Builder()
    B.renter("b", ["x", "y", "z"])
    B.renter("c", ["x"])
    B.listing("x", ["c", "b"])
    B.listing("y", ["b"])
    B.listing("z", ["b"])
    B.e0 += [("b", "y"), ("c", "x")]
    inst, e0 = B.build()
    mb, _ = deferred_acceptance(inst, e0)
    p = prefix_mask(inst, mb)
    assert p[B.b["b"], B.s["x"]] and p[B.b["b"], B.s["y"]] and not p[B.b["b"], B.s["z"]]


def test_appendix1_gap_one_client_per_firm():
    N = 3
    B = Builder()
    for i in range(N):
        a, b, c, x, y, z = [f"{n}{i}" for n in "abcxyz"]
        B.renter(a, [x, y, z])
        B.renter(b, [y, x, z])
        B.renter(c, [x, z, y])
        B.listing(x, [b, c, a])
        B.listing(y, [a, b, c])
        B.listing(z, [c, a, b])
        B.e0 += [(a, x), (a, y), (a, z), (b, x), (b, y), (b, z), (c, z)]
    B.renter("iso", ["t"])
    B.listing("t", ["iso"])
    inst, e0 = B.build()
    assert support_cost(inst, e0) == 1
    r = exact(inst, Oracle(inst, e0))
    assert r.certified and r.cost == N + 1


def test_appendix6_prefix_cover_can_overpay():
    N = 4
    B = Builder()
    for i in range(N):
        a, b, s, r = f"a{i}", f"b{i}", f"s{i}", f"r{i}"
        B.renter(a, [s, r])
        B.renter(b, [s, r])
        B.listing(s, [a, b])
        B.listing(r, [b, a])
        B.e0 += [(a, s), (b, r)]
    B.renter("p", ["y", "x"], "Fu")
    B.renter("q", ["x", "y"], "Fv")
    B.listing("x", ["p", "q"])
    B.listing("y", ["q", "p"])
    B.e0 += [("p", "x"), ("q", "x")]
    inst, e0 = B.build()
    o = Oracle(inst, e0)
    assert exact(inst, o).cost == 1
    assert prefix_cover(inst, o).cost == N + 1


def random_market(n, n_firms, rng, density=0.7, exposure=0.35, common_priority=False):
    E = rng.random((n, n)) < density
    rb = np.full((n, n), NOT_EDGE)
    rs = rb.copy()
    priority = rng.permutation(n)
    for b in range(n):
        idx = np.nonzero(E[b])[0]
        rb[b, idx] = rng.permutation(len(idx))
    for s in range(n):
        idx = np.nonzero(E[:, s])[0]
        rs[idx, s] = np.argsort(np.argsort(priority[idx])) if common_priority else rng.permutation(len(idx))
    half = n_firms // 2
    inst = Instance(rb, rs, rng.integers(0, half, n), rng.integers(half, n_firms, n), np.ones(n_firms, int))
    return inst, E & (rng.random((n, n)) < exposure)


@pytest.mark.parametrize("seed", range(25))
def test_exact_matches_enumeration(seed):
    rng = np.random.default_rng(seed)
    inst, e0 = random_market(int(rng.integers(6, 15)), 10, rng)
    o = Oracle(inst, e0)
    truth = exhaustive(inst, o)
    costs = {cuts: exact(inst, o, cuts=cuts, time_cap=60) for cuts in ("assignment", "component", "prefix")}
    for r in costs.values():
        assert r.certified and r.cost == truth.cost and o.stable(r.x)
    assert support_cost(inst, e0) <= truth.cost
    for heuristic in (reverse_delete, blocker_greedy, prefix_cover, full_disclosure):
        h = heuristic(inst, o)
        assert h.cost >= truth.cost and o.stable(h.x)


@pytest.mark.parametrize("seed", range(6))
def test_enumerated_plans_are_all_the_cheapest_plans(seed):
    rng = np.random.default_rng(100 + seed)
    inst, e0 = random_market(8, 8, rng)
    o = Oracle(inst, e0)
    optimum = exhaustive(inst, o).cost
    plans, complete = enumerate_optimal_plans(inst, o, optimum, cap=500)
    found = {p.tobytes() for p in plans}
    brute = set()
    for m in range(2 ** 8):
        x = np.array([(m >> i) & 1 for i in range(8)], bool)
        if x.sum() == optimum and o.stable(x):
            brute.add(x.tobytes())
    assert complete and found == brute


@pytest.mark.parametrize("seed", range(10))
def test_appendix13_common_priority_removes_the_gap(seed):
    rng = np.random.default_rng(200 + seed)
    inst, e0 = random_market(10, 10, rng, common_priority=True)
    r = exact(inst, Oracle(inst, e0), time_cap=60)
    assert r.certified and support_cost(inst, e0) == r.cost


def test_no_support_model_still_exact():
    rng = np.random.default_rng(7)
    inst, e0 = random_market(8, 8, rng)
    o = Oracle(inst, e0)
    assert exact(inst, o, support=False, time_cap=60).cost == exhaustive(inst, o).cost
