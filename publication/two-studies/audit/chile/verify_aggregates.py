"""Check school audit aggregates and comparator arithmetic without microdata."""
import argparse
from pathlib import Path
import pandas as pd


def run(path):
    changes=pd.read_csv(path/"school-rank-changes.csv").set_index(["year","group"])
    curves=pd.read_csv(path/"school-rank-thresholds.csv").set_index(["year","group","k"]).sort_index()
    cap=pd.read_csv(path/"school-capacity-bracket.csv").set_index(["year","scenario"])
    sim=pd.read_csv(path/"school-simulated-menu-batches.csv").set_index(["year","policy","lottery_draws","batch"]).sort_index()
    one=pd.read_csv(path/"school-simulated-menu.csv").set_index(["year","policy","lottery_draws"]).sort_index()
    pd.testing.assert_frame_equal(one,sim.xs(0,level="batch").sort_index())
    for year in (2022,2023,2024,2025):
        all_rows=changes.loc[(year,"all")]
        for a,b in (("same_menu","changed_menu"),("full_replay_agrees","full_replay_differs")):
            assert (changes.loc[(year,a)]+changes.loc[(year,b)]==all_rows).all()
        assert all_rows.placed_delta==all_rows.gain-all_rows.loss
        assert all_rows.same_outcome+all_rows.better_including_gains+all_rows.worse_including_losses==all_rows.applicants
        selected=curves.loc[(year,"all")]
        assert (selected["rank"]-selected["blind"]==selected.delta).all()
        assert (selected.submitted_rank-selected.submitted_blind==selected.submitted_delta).all()
        assert selected.delta.iloc[-1]==all_rows.placed_delta
        base=cap.loc[(year,"baseline")]
        assert base.full_exact==changes.loc[(year,"full_replay_agrees")].applicants
        assert (base.placed_delta,base.hist_delta,base.first_delta)==(
            all_rows.placed_delta,all_rows.historical_coverage_delta,selected.loc[1,"submitted_delta"])
        expanded=cap.loc[(year,"sum_frozen_menus")]
        rescored=cap.loc[(year,"sum_rescored_menus")]
        assert expanded.full_exact==rescored.full_exact
        assert expanded.full_matched==rescored.full_matched
        assert expanded.blind_rows_vs_base==expanded.rank_rows_vs_base==0
        assert expanded.hist_delta==base.hist_delta
        for row in (base,expanded,rescored):
            assert row.placed_delta==row.placed_rank-row.placed_blind==row.gains-row.losses
        if year<=2023:
            for batch in range(8):
                blind=sim.loc[(year,"blind",0,batch)]
                rank=sim.loc[(year,"rank",0,batch)]
                assert (rank.placed-blind.placed,rank.historical_shown-blind.historical_shown,
                        rank.full_shown-blind.full_shown,rank.submitted_first-blind.submitted_first)==(
                    base.placed_delta,base.hist_delta,base.full_shown_delta,base.first_delta)
                for d in (16,32,64):
                    row=sim.loc[(year,"simulated",d,batch)]
                    assert row.submitted_first<=row.placed<=row.applicants
                    assert row.full_named<=row.full_shown<=base.full_matched
                    assert 0<=row.zero_cut_ties<=row.cut_ties<=row.long_rows
                    assert row.selected_zero_long_edges<=5*row.long_rows
    print("Verified four rank cohorts, 12 capacity rows, and two eight-batch lottery simulation controls.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    run(parser.parse_args().directory)
