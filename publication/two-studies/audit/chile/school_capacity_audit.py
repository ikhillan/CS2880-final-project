"""Check overlapping versus additive incumbent capacity, with frozen and rescored displays.

The current year historical partner is only used for scoring results. Every
2019 coefficient and prior is fixed. Capacity, squared pressure, and for the
rank-informed model its rank-pressure interaction change in the rescored arm.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from school_ranks import RELEASES, read_public


def reconstruct_scores(market, new_capacity, school, numeric, scratch, year):
    fit = json.loads((school / "fit-and-inputs.json").read_text())["models"]
    with (numeric / f"{year}.bin").open("rb") as file:
        file.seek(-16 * market.ne,2)
        base_scores = {name:np.fromfile(file,count=market.ne,dtype="<f8") for name in ("blind","rank")}
    demand = np.bincount(market.j,minlength=market.m)
    change_logcap = np.log1p(new_capacity.astype(float)) - np.log1p(market.cap.astype(float))
    old_pressure = np.log((1+demand)/(1+market.cap))
    new_pressure = np.log((1+demand)/(1+new_capacity))
    delta_squared = new_pressure**2 - old_pressure**2
    delta_pressure = new_pressure - old_pressure
    with (scratch/"scores.bin").open("wb") as dest:
        for name, baseline in base_scores.items():
            model = fit[name]
            slope = dict(zip(model["features"],np.array(model["coef"][1:])/model["sd"]))
            corrected = baseline + (slope["log_capacity"]*change_logcap+slope["pressure_sq"]*delta_squared)[market.j]
            if name=="rank":
                corrected += slope["rank_pressure"]*np.log1p(market.cols)*delta_pressure[market.j]
            assert np.isfinite(corrected).all()
            corrected.astype("<f8").tofile(dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("publication","early-archives","late-archives","early-numeric","late-numeric","out"):
        p.add_argument("--"+arg,required=True,type=Path)
    args=p.parse_args()
    school=args.publication/"research/observed/school"
    sys.path.insert(0,str((args.publication/"research/policy/equity").resolve()))
    from transitions import Market
    args.out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    with TemporaryDirectory() as tmp:
        scratch=Path(tmp)
        exe=scratch/"replay"
        subprocess.run(["c++","-O3","-std=c++17","-I"+str(school),
                        str(Path(__file__).with_suffix(".cpp")),"-o",exe],check=True)
        for year,filename in RELEASES.items():
            numeric=args.early_numeric if year<=2023 else args.late_numeric
            archives=args.early_archives if year<=2023 else args.late_archives
            m=Market(numeric/f"{year}.bin")
            incumbent=np.bincount(m.ret[m.ret>=0],minlength=m.m)
            archive=archives/filename
            vacancies=read_public(archive,"A1",["vacantes"]).vacantes.to_numpy()
            np.testing.assert_array_equal(np.maximum(incumbent,vacancies),m.cap)
            capacity=incumbent+vacancies
            assert np.all(capacity>=m.cap)
            capacity.astype("<i4").tofile(scratch/"cap.bin")
            reconstruct_scores(m,capacity,school,numeric,scratch,year)
            subprocess.run([exe,numeric/str(year),scratch/"cap.bin",scratch/"scores.bin",
                            str(year),scratch/"result.csv"],check=True)
            yearrows=pd.read_csv(scratch/"result.csv").to_dict("records")
            # Independently rerun the fixed menus and full reference under the
            # expanded capacity using batched Python DA, checking aggregate totals.
            m.cap=capacity
            fixed=yearrows[1]
            full=m.solve(np.ones(m.ne,bool))
            fp=np.where(full>=0,m.j[np.maximum(full,0)],-1)
            assert int((fp==m.off).sum())==fixed["full_exact"]
            assert int((full>=0).sum())==fixed["full_matched"]
            cover=[]
            for pol in ("blind","rank"):
                mask=np.fromfile(numeric/f"{year}-{pol}.mask",np.uint8).astype(bool)
                s=m.solve(mask)
                assert int((s>=0).sum())==fixed["placed_"+pol]
                cover.append(int(mask[full[full>=0]].sum()))
            assert cover[1]-cover[0]==fixed["full_shown_delta"]
            rows.extend(yearrows)
            print(year,[(r["scenario"],r["full_exact"],r["hist_delta"],r["full_shown_delta"],r["placed_delta"]) for r in yearrows],flush=True)
    with args.out.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0],lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    main()
