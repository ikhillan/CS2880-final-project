"""Renter protection (Todd's protocol, section 2): the constructions from the protocol and the paper audit, plus brute force."""
import itertools

import numpy as np
import pytest

from hiddenoptions.audit import (Oracle, ProtectedOracle, exact, prefix_cover, protection_sets, renter_ranks, reverse_delete,
                                 support_cost)
from hiddenoptions.matching import NOT_EDGE, Instance
from tests.test_slides import Builder, random_market


def shared_broker_family(N: int):
    """Protocol: 'A market where protecting renters changes the cost from one audit to N'."""
    B = Builder()
    for i in range(N):
        a, b, x, y = f"a{i}", f"b{i}", f"x{i}", f"y{i}"
        B.renter(a, [x, y])
        B.renter(b, [y, x])
        B.listing(x, [b, a], "X")  # every x_i uses the same brokerage X
        B.listing(y, [a, b])
        B.e0 += [(a, x), (a, y)]
    return B


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_shared_broker_family(N):
    B = shared_broker_family(N)
    inst, e0 = B.build()
    o = Oracle(inst, e0)
    assert support_cost(inst, e0) == 1
    ordinary = exact(inst, o, time_cap=60)
    assert ordinary.certified and ordinary.cost == 1
    everyone = np.ones(inst.n_b, bool)
    protected, unavoidable = protection_sets(inst, o, everyone)
    assert not unavoidable.any()  # with every edge visible, both renters of each copy get their first choice
    r = exact(inst, ProtectedOracle(o, protected, "all"), time_cap=60)
    assert r.certified and r.cost == N and r.method == "exact_prefix_protect_all"
    for t in range(N + 1):  # protect t renters of type a
        group = np.zeros(inst.n_b, bool)
        group[[B.b[f"a{i}"] for i in range(t)]] = True
        p, _ = protection_sets(inst, o, group)
        assert exact(inst, ProtectedOracle(o, p, "group"), time_cap=60).cost == min(N, 1 + t)


def ring(n: int, padding: int = 0) -> tuple[Instance, np.ndarray]:
    """The paper audit's sparse ring (odd n), optionally padded with four filler listings per core proposer."""
    bp, sp, visible = [], [], set()
    for i in range(n):
        own, one, two = i, (i + 1) % n, (i + 2) % n
        extras = tuple(range(n + padding * i, n + padding * (i + 1)))
        bp.append(((two, one, own) if i == 0 else (two, own, one)) + extras)
        sp.append(((i - 1) % n, i, (i - 2) % n))
        visible.add((i, own))
        if i:
            visible.add((i, one))
        visible.update((i, z) for z in extras[: 4 if i == 0 else 3])
    for z in range(n, n * (1 + padding)):
        bp.append((z,))
        sp.append((z, (z - n) // padding))
        visible.add((z, z))
    side = len(bp)
    rb = np.full((side, side), NOT_EDGE)
    rs = rb.copy()
    for b, prefs in enumerate(bp):
        for k, s in enumerate(prefs):
            rb[b, s] = k
    for s, prefs in enumerate(sp):
        for k, b in enumerate(prefs):
            rs[b, s] = k
    inst = Instance(rb, rs, np.arange(side), np.arange(side, 2 * side), np.ones(2 * side, int))
    e0 = np.zeros_like(inst.E)
    for b, s in visible:
        e0[b, s] = True
    return inst, e0


@pytest.mark.parametrize("n", [3, 5, 7])
@pytest.mark.parametrize("padding", [0, 4])
def test_sparse_ring(n, padding):
    inst, e0 = ring(n, padding)
    o = Oracle(inst, e0)
    assert int(o.evaluate(np.zeros(inst.n_firms, bool))[2].sum()) == 1  # exactly one initial blocking pair
    assert exact(inst, o, time_cap=60).cost == 1
    protected, unavoidable = protection_sets(inst, o, np.ones(inst.n_b, bool))
    assert not unavoidable.any()
    r = exact(inst, ProtectedOracle(o, protected, "all"), time_cap=120)
    assert r.certified and r.cost == (n + 1) // 2
    assert r.support_cost == (n + 1) // 2  # the protected support bound is already exact on this family


def brute_force(inst, oracle):
    best = None
    for v in itertools.product((False, True), repeat=inst.n_firms):
        x = np.array(v, bool)
        c = int(inst.cost[x].sum())
        if (best is None or c < best) and oracle.stable(x):
            best = c
    return best


@pytest.mark.parametrize("seed", range(20))
def test_protected_exact_matches_brute_force(seed):
    rng = np.random.default_rng(300 + seed)
    inst, e0 = random_market(int(rng.integers(5, 10)), 8, rng)
    o = Oracle(inst, e0)
    lower = rng.random(inst.n_b) < 0.5
    costs = {}
    for label, group in (("lower", lower), ("all", np.ones(inst.n_b, bool))):
        p, unavoidable = protection_sets(inst, o, group)
        po = ProtectedOracle(o, p, label)
        r = exact(inst, po, time_cap=60)
        assert r.certified and r.cost == brute_force(inst, po)
        # The losses of a protected plan are exactly the unavoidable set.
        lost = group & (renter_ranks(inst, o.evaluate(r.x)[0]) > renter_ranks(inst, o.evaluate(np.zeros(inst.n_firms, bool))[0]))
        assert np.array_equal(lost, unavoidable)
        for h in (reverse_delete(inst, po), prefix_cover(inst, po)):
            assert po.stable(h.x) and h.cost >= r.cost
        costs[label] = r.cost
    ordinary = exact(inst, o, time_cap=60).cost
    assert support_cost(inst, e0) <= ordinary <= costs["lower"] <= costs["all"] <= inst.n_firms
