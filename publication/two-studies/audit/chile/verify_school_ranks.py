"""Independently rerun DA and check school rank curves using binary input row indices."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def verify(year, folder, load, curves, changes):
    market = load(folder / f"{year}.bin")
    outcomes, masks = {}, {}
    for policy in ("blind", "rank", "full"):
        masks[policy] = np.fromfile(folder / f"{year}-{policy}.mask", np.uint8).astype(bool)
        outcomes[policy] = market.solve(masks[policy])
        expected_courses = np.fromfile(folder / f"{year}-{policy}.partners", "<i4")
        got_courses = np.full(market.n, -1, dtype=int)
        placed = outcomes[policy] >= 0
        got_courses[placed] = market.j[outcomes[policy][placed]]
        np.testing.assert_array_equal(got_courses, expected_courses)
    changed = np.zeros(market.n, bool)
    changed[market.i[masks["blind"] != masks["rank"]]] = True
    full = outcomes["full"]
    full_programs = np.where(full >= 0, market.j[np.maximum(full,0)], -1)
    agreement = full_programs == market.off
    groups = {"all": np.ones(market.n,bool), "same_menu": ~changed, "changed_menu": changed,
              "full_replay_agrees": agreement, "full_replay_differs": ~agreement}
    checked = 0
    for key, keep in groups.items():
        stats = changes.loc[(year, key)]
        cdf = curves.loc[(year, key)]
        assert len(cdf) == max(np.diff(market.at)) and (cdf.index == np.arange(1,len(cdf)+1)).all()
        coverage, places, scores = {}, {}, {}
        for policy in ("blind","rank"):
            sol = outcomes[policy]
            has = sol >= 0
            ranks = sol[has] - market.at[:-1][has] + 1
            selected = keep[has]
            appended = market.j[sol[has]] == market.app[has]
            for accepted, label in ((selected,policy),(selected & ~appended,"submitted_"+policy)):
                cum = np.cumsum(np.bincount(ranks[accepted],minlength=len(cdf)+1)[1:])
                np.testing.assert_array_equal(cdf[label].to_numpy(),cum)
            scores[policy] = np.where(has,sol-market.at[:-1],max(np.diff(market.at))+1)
            coverage[policy] = np.sum(keep[market.i] & masks[policy] & (market.off[market.i] == market.j))
            places[policy] = has
        assert coverage["rank"]-coverage["blind"] == stats.historical_coverage_delta
        assert np.sum(keep & places["rank"])-np.sum(keep & places["blind"]) == stats.placed_delta
        both = places["blind"] & places["rank"]
        better = scores["rank"] < scores["blind"]
        worse = scores["rank"] > scores["blind"]
        assert np.sum(better & keep) == stats.better_including_gains
        assert np.sum(worse & keep) == stats.worse_including_losses
        assert np.sum(better & both & keep) == stats.improved_both
        assert np.sum(worse & both & keep) == stats.worsened_both
        assert np.sum(keep & ~better & ~worse) == stats.same_outcome
        checked += len(cdf)
    return dict(year=year,serial_batched_assignment_checks=int(3*market.n),
                verified_group_threshold_rows=int(checked),verified_transition_rows=len(groups))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("publication","early-numeric","late-numeric","out"):
        p.add_argument("--"+arg,required=True,type=Path)
    args = p.parse_args()
    sys.path.insert(0,str((args.publication/"research/policy/equity").resolve()))
    from transitions import Market
    d = pd.read_csv(args.out/"school-rank-changes.csv").set_index(["year","group"])
    c = pd.read_csv(args.out/"school-rank-thresholds.csv").set_index(["year","group","k"]).sort_index()
    checks = [verify(y,args.early_numeric if y <= 2023 else args.late_numeric,Market,c,d)
              for y in (2022,2023,2024,2025)]
    (args.out/"school-rank-verification.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(checks)


if __name__ == "__main__":
    main()
