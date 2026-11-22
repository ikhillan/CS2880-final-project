"""Report every pre-specified 2024 crossover contrast and validate the old policy endpoints."""
import argparse
import json
import hashlib
from pathlib import Path

import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",required=True,type=Path)
    p.add_argument("--publication",required=True,type=Path)
    args=p.parse_args()
    d=args.directory
    obs=pd.read_csv(d/"school-simulated-menu-2024-batches.csv")
    assert set(obs.year)=={2024} and set(obs.batch)==set(range(8)) and len(obs)==40
    original=pd.read_csv(args.publication/"two-studies/evidence/later-school-results.csv")
    original=original.query('year==2024 and group=="all"').set_index("policy")
    ranks=pd.read_csv(d/"school-rank-thresholds.csv").query('year==2024 and group=="all" and k==1').iloc[0]
    vars=["historical_shown","placed","full_shown","full_named","submitted_first"]
    cumulative=[]
    for batch in range(8):
        part=obs.query('batch==@batch')
        for pol in ("blind","rank"):
            old=part.query('policy==@pol').iloc[0]
            assert old.placed==original.loc[pol,"placed"]
            assert old.full_shown==original.loc[pol,"full_partner_shown"]
            assert old.historical_shown==original.loc[pol,"official_shown"]
            assert old.submitted_first==ranks["submitted_"+pol]
        rank=part.query('policy=="rank"').iloc[0]
        follow=part.query('policy=="simulated"')
        assert list(follow.lottery_draws)==[16,32,64]
        for _,row in follow.iterrows():
            assert row.submitted_first<=row.placed<=row.applicants
            assert row.full_named<=row.full_shown<=original.loc["full","placed"]
            assert 0 <= row.zero_cut_ties <= row.cut_ties <= row.long_rows
            cumulative.append(dict(year=2024,batch=batch,draws=row.lottery_draws,
                       **{"delta_"+v:int(row[v]-rank[v]) for v in vars}))
    pdf=pd.DataFrame(cumulative)
    pdf.to_csv(d/"school-simulated-menu-2024-differences.csv",index=False)
    plan=hashlib.sha256((d/"2024-COMPARATOR-PLAN.md").read_bytes()).hexdigest()
    code=hashlib.sha256((d/"school_simulated_menu.cpp").read_bytes()).hexdigest()
    assert plan=="32a8761d7fc6bc3dabc9d1eac3b4a586d5e25e96218d3d5354bf01a11a9971dd"
    assert code=="465cb66ce430dc611e1136b0b2482630c297a6d01910414f73ec961ca77d1582"
    rec=dict(local_plan_sha256=plan,policy_code_sha256=code,checked_baseline_rows=16,reported_policy_rows=24)
    (d/"school-simulated-menu-2024-check.json").write_text(json.dumps(rec,indent=2)+"\n")
    print(pdf.groupby("draws")[[c for c in pdf if c.startswith("delta_")]].agg(["min","mean","max"]).to_string())


if __name__=="__main__":
    main()
