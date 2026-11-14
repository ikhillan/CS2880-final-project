"""Trace distinct other applicants behind covered full-list partners followed by nonplacement.

The path witness requires a stable full and a stable restricted assignment. Capacitated
programs use a partial bijection between departing and incoming holders after cancelling stayers.
"""
import argparse
import csv
import importlib.util
from collections import Counter
from pathlib import Path
import sys

import numpy as np


def displacements(full, restricted, visible, priority_full, priority_new, rank_full, rank_new, subset):
    """Return path lengths for full partners displayed but not obtained by unmatched applicants.

    At a program, pair its departing and arriving holders after cancelling all common holders.
    If departures exceed arrivals, unmatched departure slots are last. Whenever a departing
    holder prefers return and has the edge on display, the program is full, so each such
    departing holder always has a paired incoming applicant of higher priority.
    """
    n = len(full)
    starts = (full >= 0) & (restricted < 0) & visible
    moved = full != restricted
    leaving = np.flatnonzero(moved & (full >= 0))
    entering = np.flatnonzero(moved & (restricted >= 0))
    leaving = leaving[np.lexsort((priority_full[leaving], full[leaving]))]
    entering = entering[np.lexsort((priority_new[entering], restricted[entering]))]
    m = max(full.max(initial=-1), restricted.max(initial=-1)) + 1
    n_left = np.bincount(full[leaving], minlength=m)
    n_joined = np.bincount(restricted[entering], minlength=m)
    first_left = np.cumsum(n_left) - n_left
    first_joined = np.cumsum(n_joined) - n_joined
    within = np.arange(len(leaving)) - first_left[full[leaving]]
    paired = within < n_joined[full[leaving]]
    next_holder = np.full(n, -1, dtype=int)
    next_holder[leaving[paired]] = entering[(first_joined[full[leaving]] + within)[paired]]
    visited = set()
    lengths = Counter()
    selected_lengths = Counter()
    for i in np.flatnonzero(starts):
        at = i
        length = 0
        while visible[at]:
            if at in visited:
                raise AssertionError("Disjoint paths must not meet or cycle")
            visited.add(at)
            new = next_holder[at]
            assert new >= 0 and full[at] == restricted[new]
            assert priority_new[new] < priority_full[at]
            assert full[new] >= 0 and rank_new[new] > rank_full[new]
            length += 1
            at = new
        assert at not in visited
        visited.add(at)
        lengths[length] += 1
        if subset[i]:
            selected_lengths[length] += 1
    return lengths, selected_lengths


def school_vectors(market, initial_program, new_program, mask):
    full = np.full(market.n, -1, dtype=int)
    new = np.full(market.n, -1, dtype=int)
    for program, out in [(initial_program, full), (new_program, new)]:
        ids = np.flatnonzero(market.j == program[market.i])
        assert len(ids) == np.count_nonzero(program >= 0)
        out[market.i[ids]] = ids
    return np.where(full >= 0, mask[full], False), market.pr[full], market.pr[new], full, new


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def school(args):
    paired = load_module("paired_school", args.school_reader)
    for year in (2022, 2023):
        m, masks, partners = paired.read_cohort(year, args.school_numeric, args.repo / "publication")
        full, a, b = [partners[k] for k in ("full", "blind", "rank")]
        visible, p0, p1, r0, r1 = school_vectors(m, full, b, masks["rank"])
        subset = (a >= 0) & ~paired.has_course(m, masks["blind"], full)
        yield "Chile", str(year), *displacements(full, b, visible, p0, p1, r0, r1, subset)


def housing(args):
    sys.path.insert(0, str(args.repo.resolve()))
    from experiments.config import expand_labeled, load_config
    from experiments.pilot_outcomes import DEFAULT_CACHE, CONFIG, load_pilot_cache, pilot_displays
    from hiddenoptions.markets import Catalog, InsufficientSupply, build_market
    from hiddenoptions.matching import deferred_acceptance
    cache, catalog = load_pilot_cache(DEFAULT_CACHE), Catalog()
    for spec, design in expand_labeled(load_config(CONFIG)):
        try:
            market = build_market(spec, catalog)
        except InsufficientSupply:
            continue
        inst = market.inst
        uniform, haiku = pilot_displays(market, cache)
        full, a, b = [deferred_acceptance(inst, menu)[0] for menu in (inst.E, uniform, haiku)]
        i = np.arange(inst.n_b)
        visible = (full >= 0) & haiku[i,full]
        subset = (a >= 0) & ~uniform[i,full]
        yield "NYC", design, *displacements(full, b, visible, inst.rank_s[i,full], inst.rank_s[i,b],
                                               inst.rank_b[i,full], inst.rank_b[i,b], subset)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--school-reader", type=Path, required=True)
    p.add_argument("--school-numeric", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    table = {}
    for gen in (school, housing):
        for site, panel, lengths, selected in gen(args):
            for group, hist in (("covered_but_unplaced", lengths), ("gained_coverage_lost_placement", selected)):
                table.setdefault((site, panel, group), Counter()).update(hist)
    with args.out.open("w") as f:
        w = csv.writer(f)
        w.writerow(["setting","panel","subset","program_steps","applicants"])
        for key, hist in sorted(table.items()):
            for hops, num in sorted(hist.items()):
                w.writerow([*key,hops,num])
            print(*key, "total", sum(hist.values()), "direct", hist[1], "longest", max(hist, default=0))


if __name__ == "__main__":
    main()
