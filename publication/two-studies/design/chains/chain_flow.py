"""Vacancy chains under fixed priorities. Rows of contracts are in strict preference order."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order


@dataclass
class Contracts:
    at: np.ndarray
    j: np.ndarray
    pr: np.ndarray
    cap: np.ndarray
    protected: np.ndarray  # protected program index or -1 for each applicant

    def __post_init__(self):
        self.n = len(self.at) - 1
        self.m = len(self.cap)
        self.i = np.repeat(np.arange(self.n), np.diff(self.at))
        self.edges = np.arange(len(self.j))

    def limit(self, assignment):
        return np.where(assignment >= 0, assignment, self.at[1:])

    def occupancy(self, assignment):
        return np.bincount(self.j[assignment[assignment >= 0]], minlength=self.m)

    def nonblocking(self, assignment, shown):
        """Inspect every displayed pair and every holder under strict responsive priorities."""
        held = assignment[assignment >= 0]
        assert np.all(self.i[held] == np.flatnonzero(assignment >= 0))
        assert np.all(shown[held])
        count = self.occupancy(assignment)
        assert np.all(count <= self.cap)
        worst = np.full(self.m, -np.inf)
        np.maximum.at(worst, self.j[held], self.pr[held])
        better = shown & (self.edges < self.limit(assignment)[self.i])
        dst = self.j[better]
        assert np.all((count[dst] == self.cap[dst]) & (self.pr[better] >= worst[dst]))

    def edited_menu(self, witness, shown):
        changed = shown.copy()
        edit = witness[(witness >= 0) & ~shown[np.maximum(witness, 0)]]
        discretionary = shown & (self.j != self.protected[self.i])
        last = np.full(self.n, -1)
        np.maximum.at(last, self.i[discretionary], self.edges[discretionary])
        dropped = last[self.i[edit]]
        assert np.all(dropped >= 0)
        changed[dropped], changed[edit] = False, True
        assert np.array_equal(
            np.bincount(self.i[shown], minlength=self.n),
            np.bincount(self.i[changed], minlength=self.n),
        )
        assert np.all(changed[shown & (self.j == self.protected[self.i])])
        return changed, len(edit)


def augment(a, old, eligible):
    """Integral residual flow from the baseline. Return a primal assignment and verify a dual cut."""
    incumbent = old >= 0
    held = old[incumbent]
    assert eligible[held].all()
    forward = eligible.copy()
    forward[held] = False
    edges = a.edges[forward]
    free = np.flatnonzero(~incumbent)
    remaining = a.cap - a.occupancy(old)
    vacancies = np.flatnonzero(remaining > 0)
    source = a.n + a.m
    sink = source + 1
    ones = len(free) + len(edges) + len(held)
    u = np.r_[np.full(len(free), source), a.i[edges], a.n + a.j[held], a.n + vacancies]
    v = np.r_[free, a.n + a.j[edges], a.i[held], np.full(len(vacancies), sink)]
    capacity = np.r_[np.ones(ones, dtype=np.int32), remaining[vacancies]].astype(
        np.int32
    )
    graph = csr_matrix((capacity, (u, v)), shape=(sink + 1, sink + 1))
    assert graph.nnz == len(u), "repeated applicant-program contract"
    result = maximum_flow(graph, source, sink)
    residual = graph - result.flow
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, source, return_predecessors=False)
    left = np.zeros(sink + 1, dtype=bool)
    left[reachable] = True
    assert not left[sink]
    assert int(capacity[left[u] & ~left[v]].sum()) == result.flow_value
    out = old.copy()
    if len(edges):
        carrying = np.asarray(result.flow[a.i[edges], a.n + a.j[edges]]).reshape(-1) > 0
        new = edges[carrying]
        assert np.unique(a.i[new]).size == len(new)
        out[a.i[new]] = new
    assert np.all(a.limit(out) <= a.limit(old))
    assert np.all(a.occupancy(out) >= a.occupancy(old))
    assert np.all(a.occupancy(out) <= a.cap)
    assert (out >= 0).sum() - incumbent.sum() == result.flow_value
    return out, int(result.flow_value)


def decompose_chains(a, old, witness):
    """Pair arrivals with departures at each program, discarding any remaining pure cycles."""
    from collections import defaultdict

    moved = np.flatnonzero(witness != old)
    departures = defaultdict(list)
    for person in moved[old[moved] >= 0]:
        departures[a.j[old[person]]].append(person)
    following = {}
    for person in moved:
        destination = a.j[witness[person]]
        group = departures[destination]
        if group:
            following[person] = group.pop()
    assert not any(departures.values())
    # Each incumbent departure has exactly one predecessor. Entrants have none.
    entrants = moved[old[moved] < 0]
    active = set()
    lengths = []
    for entrant in entrants:
        person = entrant
        length = 0
        while person is not None:
            assert person not in active
            active.add(person)
            length += 1
            person = following.get(person)
        lengths.append(length)
    clean = old.copy()
    active = list(active)
    clean[active] = witness[active]
    assert np.array_equal(a.occupancy(clean), a.occupancy(witness))
    return clean, lengths


def evaluate(a, old, shown, da, top=None):
    """When top is set, only a previously hidden entry must be in the first top original options."""
    a.nonblocking(old, shown)
    held = old[old >= 0]
    better = a.edges < a.limit(old)[a.i]
    allowable = better.copy()
    if top is not None:
        allowable &= shown | ((a.edges - a.at[a.i]) < top)
    old_claims = shown & better
    cutoff = np.full(a.m, np.inf)
    np.minimum.at(cutoff, a.j[old_claims], a.pr[old_claims])
    occupancy = a.occupancy(old)
    assert np.all(occupancy[np.isfinite(cutoff)] == a.cap[np.isfinite(cutoff)])
    discretionary = shown & (a.j != a.protected[a.i])
    has_swap = np.bincount(a.i[discretionary], minlength=a.n) > 0
    movable = allowable & (shown | has_swap[a.i])
    protected = movable & (a.pr <= cutoff[a.j])
    protected[held] = True
    candidates = allowable.copy()
    candidates[held] = True
    direct = movable & (old[a.i] < 0) & (occupancy[a.j] < a.cap[a.j])
    direct[held] = True
    direct_w, direct_gain = augment(a, old, direct)
    a.nonblocking(direct_w, shown | np.isin(a.edges, direct_w[direct_w >= 0]))
    w, certified = augment(a, old, protected)
    w, lengths = decompose_chains(a, old, w)
    edited, n_edits = a.edited_menu(w, shown)
    # Check even the old options that will be dropped, in addition to all new contracts.
    a.nonblocking(w, shown | edited)
    actual = da(edited)
    a.nonblocking(actual, edited)
    assert np.all(a.limit(actual) <= a.limit(w))
    assert np.array_equal(actual >= 0, w >= 0)
    _, upper = augment(a, old, candidates)
    assert direct_gain <= certified <= upper
    return dict(
        applicants=a.n,
        placed=int(len(held)),
        direct=direct_gain,
        chain=certified,
        upper=upper,
        edited=n_edits,
        paths_with_incumbents=sum(length > 1 for length in lengths),
        longest_path=max(lengths, default=0),
        movers_witness=int(np.sum((old >= 0) & (w != old))),
        movers_da=int(np.sum((old >= 0) & (actual != old))),
        entrants_at_witness_option=int(np.sum((old < 0) & (w >= 0) & (actual == w))),
        tight=int(certified == upper),
    )


def chile(repo, early, late, years, top=None):
    import sys

    sys.path.insert(0, str(repo / "publication/research/policy/equity"))
    from transitions import Market

    for year in years:
        directory = early if year <= 2023 else late
        m = Market(directory / f"{year}.bin")
        a = Contracts(m.at, m.j, m.pr, m.cap, m.ret)
        for policy in ("blind", "rank"):
            shown = np.fromfile(
                directory / f"{year}-{policy}.mask", dtype=np.uint8
            ).astype(bool)
            partners = np.fromfile(directory / f"{year}-{policy}.partners", dtype="<i4")
            hits = partners[a.i] == a.j
            old = np.full(a.n, -1, dtype=int)
            old[a.i[hits]] = a.edges[hits]
            assert hits.sum() == (partners >= 0).sum()
            assert np.array_equal(old, m.solve(shown))
            yield dict(
                year=year, policy=policy, **evaluate(a, old, shown, m.solve, top)
            )


def housing(repo, top=None):
    import sys

    sys.path.insert(0, str(repo))
    from experiments.config import expand_labeled, load_config
    from experiments.pilot_outcomes import (
        CONFIG,
        DEFAULT_CACHE,
        load_pilot_cache,
        pilot_displays,
    )
    from hiddenoptions.markets import Catalog, build_market, InsufficientSupply
    from hiddenoptions.matching import deferred_acceptance

    catalog, cache = Catalog(), load_pilot_cache(DEFAULT_CACHE)
    built = 0
    for spec, design in expand_labeled(load_config(CONFIG)):
        try:
            m = build_market(spec, catalog)
        except InsufficientSupply:
            assert spec.borough == "Staten Island"
            continue
        built += 1
        inst = m.inst
        i, j = np.nonzero(inst.E)
        order = np.lexsort((inst.rank_b[i, j], i))
        i, j = i[order], j[order]
        at = np.r_[0, np.cumsum(np.bincount(i, minlength=inst.n_b))]
        a = Contracts(
            at, j, inst.rank_s[i, j], np.ones(inst.n_s, int), np.full(inst.n_b, -1)
        )

        def da(shown):
            display = np.zeros_like(inst.E)
            display[i, j] = shown
            dest, _ = deferred_acceptance(inst, display)
            out = np.full(a.n, -1)
            found = dest[i] == j
            out[i[found]] = a.edges[found]
            return out

        for policy, mask in zip(("uniform", "haiku"), pilot_displays(m, cache)):
            shown = mask[i, j]
            yield dict(
                design=design,
                borough=spec.borough,
                month=spec.month,
                seed=spec.seed,
                policy=policy,
                **evaluate(a, da(shown), shown, da, top),
            )
    assert built == 480


if __name__ == "__main__":
    import argparse
    import csv
    from pathlib import Path

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("study", choices=["chile", "housing"])
    for name in ("repo", "early", "late", "out"):
        cli.add_argument("--" + name, type=Path, required=name in ("repo", "out"))
    cli.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    cli.add_argument("--top", type=int)
    args = cli.parse_args()
    stream = (
        chile(args.repo.resolve(), args.early, args.late, args.years, args.top)
        if args.study == "chile"
        else housing(args.repo.resolve(), args.top)
    )
    with args.out.open("w") as f:
        writer = None
        for row in stream:
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=row.keys(), lineterminator="\n")
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            if args.study == "chile":
                print(row, flush=True)
