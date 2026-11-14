"""Reconcile the common benchmark, frozen-year test, policy mixes and stability screen."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT.parent / "evidence"


def verify_later_years():
    raw = pd.read_csv(EVIDENCE / "later-school-results.csv")
    archive = json.loads((EVIDENCE / "later-cohorts.json").read_text())
    frozen = "122413f014977a69ddd213e41b87a3979b4b979dd9414c97b7c4a1fe91ae4f75"
    assert archive["frozen_fit_sha256"] == frozen
    fields = [c for c in raw if c not in ("year", "group", "policy")]
    for (year, policy), panel in raw.groupby(["year", "policy"]):
        by_group = panel.set_index("group")[fields]
        np.testing.assert_array_equal(by_group.loc["all"], by_group.loc["priority"] + by_group.loc["other"])
    all_years = raw[raw.group == "all"].set_index(["year","policy"])[fields]
    delta = all_years.xs("rank",level="policy").subtract(all_years.xs("blind",level="policy"))
    np.testing.assert_array_equal(delta.official_shown,[1735,1324])
    np.testing.assert_array_equal(delta.placed,[-481,-321])
    assert int(delta.lost_same_menu.sum()) == 748
    earlier = pd.read_csv(EVIDENCE / "school-levels.csv")
    earlier = earlier[(earlier.group=="all") & (earlier.reference=="recorded_offer")]
    assert earlier.applicants.sum()+all_years.xs("rank",level="policy").n.sum()==2044649
    assert (earlier.rank_shown-earlier.blind_shown).sum()+delta.official_shown.sum()==5552
    assert (earlier.rank_placed-earlier.blind_placed).sum()+delta.placed.sum()==-1787


def verify_rollout():
    school = pd.read_csv(ROOT / "chile/school-rollout-summary.csv")
    for y,cohort in school.groupby("year"):
        rows=cohort.set_index(["uptake","metric"])
        assert rows.loc[(.5,"full_covered_minus_linear"),"count"] == 0
        d=rows.loc[(.5,"placed_delta"),"count"]-.5*rows.loc[(1.,"placed_delta"),"count"]
        np.testing.assert_allclose(d,rows.loc[(.5,"placed_minus_linear"),"count"])
        np.testing.assert_allclose(rows.loc[(1.,"placed_delta"),"count"],-468 if y==2022 else -517)
    nyc = pd.read_csv(ROOT / "nyc/rental-rollout.csv")
    for _,cohort in nyc.groupby("design"):
        placed = cohort[cohort.metric=="placed"].set_index("p")
        cov=cohort[cohort.metric=="covered"].set_index("p")
        np.testing.assert_allclose(cov.loc[.5,"interpolation_residual_pp"],0,atol=1e-10)
        np.testing.assert_allclose(placed.loc[.5,"interpolation_residual_pp"],
                                   placed.loc[.5,"delta_pp"]-.5*placed.loc[1.,"delta_pp"])
    cert = pd.read_csv(ROOT / "theory/rollout-certificates.csv")
    assert (cert.baseline_placed == cert.screened_placed).all()
    for side in ("gains","losses"):
        assert (cert[side]<=cert.bound).all() and (cert.bound<=cert.flagged).all()
    assert (cert.screened_changes+cert.flagged==cert.changed).all()
    chile=cert[cert.setting=="Chile"]
    assert chile.full_safe_improving_rows.sum()==22 and chile.screened_changes.sum()==27851
    prefix = pd.read_csv(ROOT/"theory/prefix-obstruction.csv")
    assert prefix[prefix.setting=="NYC"].covered_but_unplaced.sum()==0
    assert prefix[prefix.setting=="Chile"].covered_but_unplaced.sum()==246
    assert (prefix[prefix.setting=="Chile"].covered_but_unplaced == prefix[prefix.setting=="Chile"].traced_paths).all()
    lottery = pd.read_csv(ROOT/"chile/school-lottery-summary.csv")
    randomized = lottery[(lottery.lottery!="observed") & (lottery.outcome=="placed")]
    assert randomized.draws.sum()==128 and (randomized.draws==randomized.joint_sign_draws).all()
    quotas=pd.read_csv(ROOT/"chile/school-budget-check.csv")
    assert len(quotas)==10 and (quotas.rank_placed<quotas.blind_placed).all()
    assert (quotas.rank_full_covered>quotas.blind_full_covered).all()
    assert (quotas.rank_recorded_covered>quotas.blind_recorded_covered).all()


def verify_common():
    common=pd.read_csv(EVIDENCE / "common-benchmark.csv")
    assert len(common)==4 and (common.draws==48).all()
    assert (common.delta_coverage_pp>0).all() and (common.delta_placement_pp<0).all()
    school=pd.read_csv(EVIDENCE / "school-benchmarks.csv").set_index(["year","policy"])
    for row in common.itertuples():
        np.testing.assert_allclose(row.first_placed-row.uniform_placed,row.delta_placement_pp)
        if row.site=="Chile":
            first,uniform = (school.loc[(int(row.panel),policy)] for policy in ("first","uniform"))
            np.testing.assert_allclose([row.first_placed,row.uniform_placed],
                                       [first.placed_per100,uniform.placed_per100])


if __name__ == "__main__":
    verify_later_years()
    verify_common()
    verify_rollout()
    print("Later cohorts, common contrasts, rollout checks, and assignment-preserving screen agree.")
