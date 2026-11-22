"""Cross-check the empirical tables from aggregate files alone."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1] / "evidence"


def school_checks():
    joint = pd.read_csv(BASE / "school-joint.csv")
    levels = pd.read_csv(BASE / "school-levels.csv")
    named = pd.read_csv(BASE / "school-named.csv").set_index(["year", "reference"])
    bench = pd.read_csv(BASE / "school-benchmarks.csv").set_index(["year", "policy"])
    capacity = pd.read_csv(BASE / "school-capacity.csv").set_index(["year", "policy"])
    referrals = pd.read_csv(BASE / "school-referrals.csv").set_index(["year", "policy", "rank_limit"])
    for row in levels.itertuples(index=False):
        cell = joint[(joint.year == row.year) & (joint.reference == row.reference) & (joint.group == row.group)]
        assert len(cell) == 9 and cell["count"].sum() == row.applicants
        assert cell.delta_coverage @ cell["count"] == row.rank_shown - row.blind_shown
        assert cell.delta_placement @ cell["count"] == row.rank_placed - row.blind_placed
        if row.group == "all":
            gain = cell[(cell.delta_coverage == 1)]["count"].sum()
            rec = named.loc[row.year, row.reference]
            assert gain == rec.newly_shown
            assert gain == rec.newly_shown_receive_named + rec.newly_shown_receive_other + rec.newly_shown_receive_none
    for year in (2022, 2023):
        controls = pd.read_csv(BASE / f"school-controls-{year}.csv")
        flow = pd.read_csv(BASE / f"school-independent-{year}.csv").set_index(["policy", "scope"])
        for pol, rows in controls.groupby("policy"):
            rec = bench.loc[year, pol]
            for col, field in [("covered_full", "full_covered"), ("covered_recorded", "official_covered"), ("placed", "placed")]:
                assert len(rows) == rec.menu_draws
                np.testing.assert_allclose(rows[col].mean(), rec[field], rtol=0, atol=1e-8)
        for pol in ("blind", "rank"):
            cap = capacity.loc[year, pol]
            assert cap.visible_capacity == flow.loc[(pol, "display"), "maximum"]
            assert cap.placed + cap.shortfall == cap.visible_capacity
            for limit in ("top5", "any"):
                extension = referrals.loc[(year, pol, limit)]
                other = flow.loc[(pol, limit)]
                assert extension.max_extension == extension.minimum_cut == other.maximum == other.named
                assert extension.incumbent_rank_losses == other.worse == 0
                assert extension.initially_placed + extension.max_extension == other.after == extension.post_referral_placed


def rental_checks():
    joint = pd.read_csv(BASE / "joint.csv")
    named = pd.read_csv(BASE / "named-partners.csv")
    referrals = pd.read_csv(BASE / "rental-referrals.csv")
    paths = pd.read_csv(BASE / "displacement-paths.csv")
    for rec in named.itertuples(index=False):
        panel = joint[(joint.design == rec.design) & (joint.group == rec.group)]
        assert panel["count"].sum() == rec.n
        assert panel.loc[panel.delta_coverage == 1, "count"].sum() == rec.newly_shown
        assert rec.newly_shown == rec.newly_shown_named + rec.newly_shown_other + rec.newly_shown_unplaced
        assert rec.named_haiku - rec.named_uniform == rec.named_gain - rec.named_loss
        if rec.group == "all":
            old = referrals[(referrals.design == rec.design) & (referrals.rank_limit == "any")].set_index("policy")
            assert panel.delta_placement @ panel["count"] == old.loc["haiku","baseline_placed"] - old.loc["uniform","baseline_placed"]
            special = panel[(panel.delta_coverage == 1) & (panel.delta_placement == -1)]["count"].sum()
            traced = paths[(paths.setting == "NYC") & (paths.panel == rec.design) & (paths.subset == "gained_coverage_lost_placement")]
            assert traced.applicants.sum() == special
    assert json.loads((BASE / "path-exhaustive.json").read_text())["stable_outcome_pairs"] == 82689654


if __name__ == "__main__":
    school_checks()
    rental_checks()
    print("All school benchmarks, flow checks, joint margins and rental transitions reconciled.")
