"""Joint applicant transitions for the two frozen Chilean school menus. Emits aggregate counts."""
import argparse
import importlib.util
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def import_market(publication):
    path = publication / "research/policy/equity/transitions.py"
    spec = importlib.util.spec_from_file_location("school_equity_transitions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Market


def has_course(market, display, target):
    """An outside outcome can never be shown. Match the public course within each pupil's row."""
    hits = display & (market.j == target[market.i]) & (target[market.i] >= 0)
    present = np.zeros(market.n, dtype=bool)
    present[market.i[hits]] = True
    return present


def tabulate(numerator_a, numerator_b, placed_a, placed_b, group):
    dc = numerator_b.astype(np.int8) - numerator_a
    dy = placed_b.astype(np.int8) - placed_a
    joint = np.bincount(3 * (dc[group] + 1) + (dy[group] + 1), minlength=9).reshape(3, 3)
    assert joint.sum() == group.sum()
    assert joint[2].sum() - joint[0].sum() == numerator_b[group].sum() - numerator_a[group].sum()
    assert joint[:, 2].sum() - joint[:, 0].sum() == placed_b[group].sum() - placed_a[group].sum()
    return joint


def read_cohort(year, numeric, publication):
    Market = import_market(publication)
    market = Market(numeric / f"{year}.bin")
    masks = {}
    partners = {}
    for policy in ("full", "blind", "rank"):
        raw = np.fromfile(numeric / f"{year}-{policy}.mask", dtype=np.uint8)
        if len(raw) != market.ne or not np.isin(raw, [0, 1]).all():
            raise ValueError(f"{year}/{policy}: invalid edge mask")
        masks[policy] = raw.astype(bool)
        chosen = market.solve(masks[policy])
        programs = np.full(market.n, -1, np.int32)
        matched = chosen >= 0
        programs[matched] = market.j[chosen[matched]]
        archived = np.fromfile(numeric / f"{year}-{policy}.partners", dtype="<i4")
        np.testing.assert_array_equal(programs, archived)
        partners[policy] = programs
    assert masks["full"].all()
    for policy in ("blind", "rank"):
        lengths = np.bincount(market.i[masks[policy]], minlength=market.n)
        np.testing.assert_array_equal(lengths, np.minimum(market.degree, 5) + (market.ret >= 0))
        assert has_course(market, masks[policy], market.ret)[market.ret >= 0].all()
    return market, masks, partners


def assess(year, numeric, publication):
    market, masks, partners = read_cohort(year, numeric, publication)
    unchanged = np.ones(market.n, dtype=bool)
    unchanged[market.i[masks["blind"] != masks["rank"]]] = False
    assert unchanged[market.degree <= 5].all()
    groups = {
        "all": np.ones(market.n, dtype=bool),
        "same_menu": unchanged,
        "changed_menu": ~unchanged,
        "short_list": market.degree <= 5,
        "unchanged_long_list": unchanged & (market.degree > 5),
        "designated": market.ses == 1,
        "other": market.ses == 0,
    }
    before, after = (partners[arm] >= 0 for arm in ("blind", "rank"))
    joint_rows, levels = [], []
    for name, reference in (("full_DA", partners["full"]), ("recorded_offer", market.off)):
        blind, rank = (has_course(market, masks[arm], reference) for arm in ("blind", "rank"))
        assert (blind[unchanged] == rank[unchanged]).all()
        for group, population in groups.items():
            base = {"year": year, "reference": name, "group": group}
            joint = tabulate(blind, rank, before, after, population)
            for row, col in product(range(3), range(3)):
                joint_rows.append({**base, "delta_coverage": row - 1, "delta_placement": col - 1,
                                   "count": int(joint[row, col])})
            levels.append({**base, "applicants": int(population.sum()),
                           "with_reference": int(((reference >= 0) & population).sum()),
                           "blind_shown": int((blind & population).sum()),
                           "rank_shown": int((rank & population).sum()),
                           "blind_placed": int((before & population).sum()),
                           "rank_placed": int((after & population).sum())})
    # Validate coverage, same-menu partition and gross placement flows against prior
    # summaries prepared from per-applicant assignments independently of this table.
    prior = pd.read_csv(publication / f"research/policy/equity/{year}-cells.csv.gz")
    prior = prior[(prior.mechanism == "pooled") & (prior.policy == "rank")].sum(numeric_only=True)
    observed = next(r for r in levels if r["reference"] == "recorded_offer" and r["group"] == "all")
    for key, previous in (("applicants", "n"), ("rank_placed", "placed"),
                          ("blind_placed", "placed_base"), ("rank_shown", "covered")):
        assert observed[key] == prior[previous]
    assert (~unchanged).sum() == prior.changed
    assert (after & ~before).sum() == prior.gain
    assert (~after & before).sum() == prior.loss
    arrived = []
    for name, reference in (("full_DA", partners["full"]), ("recorded_offer", market.off)):
        blind_shown = has_course(market, masks["blind"], reference)
        rank_shown = has_course(market, masks["rank"], reference)
        newly_shown = ~blind_shown & rank_shown
        received_blind = (reference >= 0) & (partners["blind"] == reference)
        received_rank = (reference >= 0) & (partners["rank"] == reference)
        assert (blind_shown[received_blind]).all() and (rank_shown[received_rank]).all()
        assert not received_blind[newly_shown].any()
        arrived.append({"year": year, "reference": name, "applicants": market.n,
                        "newly_shown": int(newly_shown.sum()),
                        "newly_shown_receive_named": int((newly_shown & received_rank).sum()),
                        "newly_shown_receive_other": int((newly_shown & after & ~received_rank).sum()),
                        "newly_shown_receive_none": int((newly_shown & ~after).sum()),
                        "blind_receive_named": int(received_blind.sum()),
                        "rank_receive_named": int(received_rank.sum())})
    return joint_rows, levels, arrived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True,
                        help="Archived publication directory, including research/policy/equity")
    parser.add_argument("--numeric", type=Path, required=True,
                        help="Private school input binaries and exported frozen five-choice menus")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    joint_rows, level_rows, named_rows = [], [], []
    for year in (2022, 2023):
        joint, levels, named = assess(year, args.numeric, args.publication)
        joint_rows.extend(joint)
        level_rows.extend(levels)
        named_rows.extend(named)
    pd.DataFrame(joint_rows).to_csv(args.out / "school-joint.csv", index=False)
    pd.DataFrame(level_rows).to_csv(args.out / "school-levels.csv", index=False)
    pd.DataFrame(named_rows).to_csv(args.out / "school-named.csv", index=False)


if __name__ == "__main__":
    main()
