"""Replay one-slot referrals on the archived pooled-priority school choice markets."""
import argparse
import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def edge_assignments(market, partners):
    selected = np.flatnonzero(market.j == partners[market.i])
    result = np.full(market.n, -1, dtype=int)
    result[market.i[selected]] = selected
    if len(selected) != (partners >= 0).sum() or not np.array_equal(result >= 0, partners >= 0):
        raise ValueError("A replay partner has an absent or ambiguous application edge")
    return result


def panel(market, stem, policy, flow):
    show = np.fromfile(stem.parent / f"{stem.name}-{policy}.mask", dtype=np.uint8).astype(bool)
    partners = np.fromfile(stem.parent / f"{stem.name}-{policy}.partners", dtype="<i4")
    start = edge_assignments(market, partners)
    if not np.array_equal(start, market.solve(show)):
        raise ValueError("Baseline school assignments did not reproduce")
    active = start >= 0
    seats = market.cap - np.bincount(partners[active], minlength=market.m)
    remaining = ~active[market.i] & (seats[market.j] > 0)
    assert not (remaining & show).any()
    frame = SimpleNamespace(a=market.i, school=market.j, n=market.n, m=market.m)
    for label, eligible in (("top5", remaining & (market.cols < 5)), ("any", remaining)):
        edges = np.flatnonzero(eligible)
        chosen, cut = flow.match_vacancies(frame, seats, edges, 2880781)
        people = market.i[chosen]
        revised = show.copy()
        # Keep the guaranteed continuity edge, which is outside the five-slot quota.
        candidates = np.flatnonzero(show & (market.j != market.ret[market.i]))
        worst = np.full(market.n, -1, int)
        np.maximum.at(worst, market.i[candidates], candidates)
        assert (worst[people] >= 0).all() and (market.degree[people] > 5).all()
        revised[worst[people]] = False
        revised[chosen] = True
        assert np.array_equal(np.bincount(market.i[revised], minlength=market.n),
                              np.bincount(market.i[show], minlength=market.n))
        assert np.all(revised[market.j == market.ret[market.i]])
        after = market.solve(revised)
        selected = np.zeros(market.n, bool)
        selected[people] = True
        assert np.array_equal(after >= 0, active | selected)
        assert (after[active] <= start[active]).all() and np.array_equal(after[people], chosen)
        assert cut["cut"] == len(chosen)
        yield dict(year=stem.name, policy=policy, rank_limit=label, applicants=market.n,
                   initially_placed=int(active.sum()), vacant_seats=int(seats.sum()),
                   unmatched_with_option=cut["eligible"], eligible_edges=len(edges),
                   reachable_vacancies=cut["reachable_seats"], max_extension=len(chosen),
                   post_referral_placed=int((after >= 0).sum()), named_referrals_verified=len(people),
                   incumbent_rank_losses=int(np.count_nonzero(after[active] > start[active])),
                   minimum_cut=cut["cut"])


def main(args):
    load_market = load_module(args.published / "research/policy/equity/transitions.py").Market
    flow = load_module(args.published / "research/policy/design/repair.py")
    records = []
    for year in ("2022", "2023"):
        stem = args.numeric / year
        market = load_market(stem.with_suffix(".bin"))
        for policy in ("blind", "rank"):
            records.extend(panel(market, stem, policy, flow))
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", type=Path, required=True, help="archived publication directory")
    parser.add_argument("--numeric", type=Path, required=True, help="private binaries, masks and partner vectors")
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
