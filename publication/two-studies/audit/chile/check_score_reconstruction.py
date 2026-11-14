"""Recompute every shifted logit from raw features and check outcomes for direct scoring."""
import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import numpy as np
import pandas as pd
from school_ranks import RELEASES
from school_capacity_audit import reconstruct_scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("publication","training-archives","early-archives","late-archives","early-numeric","late-numeric","out","published"):
        p.add_argument("--"+arg,required=True,type=Path)
    p.add_argument("--direct",type=Path,required=True)
    args=p.parse_args()
    school=args.publication/"research/observed/school"
    sys.path.insert(0,str(school.resolve()))
    sys.path.insert(0,str((args.publication/"research/policy/equity").resolve()))
    import prepare
    from transitions import Market
    fit=json.loads((school/"fit-and-inputs.json").read_text())["models"]
    ak,_,y,_,d,_=prepare.prepare(args.training_archives,2019)
    edit=d["tier"]!=0
    j=d["j"][edit]
    rate=y[edit].mean()
    ak["win"]=np.bincount(j,weights=y[edit],minlength=len(ak))
    ak["n"]=np.bincount(j,minlength=len(ak))
    prepare.FILES.update(RELEASES)
    prepare.URLS.update(RELEASES)
    reference=pd.read_csv(args.published)
    info=[]
    direct_tables=[]
    with TemporaryDirectory() as tmp:
        scratch=Path(tmp)
        exe=scratch/"replay"
        subprocess.run(["c++","-std=c++17","-O3","-I"+str(school),Path(__file__).with_name("school_capacity_audit.cpp"),"-o",exe], check=True)
        for year in RELEASES:
            numeric=args.early_numeric if year<=2023 else args.late_numeric
            archives=args.early_archives if year<=2023 else args.late_archives
            market=Market(numeric/f"{year}.bin")
            catalog,pos,_,x,raw,_=prepare.prepare(archives,year)
            nload=np.bincount(raw["j"],minlength=raw["m"])
            vacancies=prepare.read(archives,year,"A1",["vacantes"]).vacantes.to_numpy(dtype=int)
            newcap=vacancies+np.bincount(raw["ret"][raw["ret"]>=0],minlength=len(catalog))
            counts=catalog.merge(ak,how="left",on=prepare.AKEY,validate="one_to_one").fillna(0)
            j=raw["j"]
            prior=((counts.win+20*rate)/(counts.n+20)).to_numpy()[j]
            x["log_odds_prior"]=np.log(prior/(1-prior))
            x["prior_sq"]=x.log_odds_prior.to_numpy()**2
            results={}
            reconstruct_scores(market,newcap,school,numeric,scratch,year)
            with (scratch/"scores.bin").open("rb") as src:
                expanded={pol: np.fromfile(src,"<f8",count=market.ne) for pol in ("blind","rank")}
            for scenario,capacity in (("old",raw["cap"]),("sum",newcap)):
                features=x.copy()
                pressure=np.log((1+nload[j])/(1+capacity[j]))
                features["log_capacity"]=np.log1p(capacity[j])
                features["pressure_sq"]=pressure**2
                for name in ("blind","rank"):
                    if name=="rank":
                        features["log_rank"]=np.log(pos)
                        features["rank1"]=(pos==1).astype(float)
                        features["rank_pressure"]=np.log(pos)*pressure
                    assert features.columns.tolist()==fit[name]["features"]
                    results[(scenario,name)]=prepare.predict(features.to_numpy(),fit[name])
            with (numeric/f"{year}.bin").open("rb") as src:
                src.seek(-16*market.ne,2)
                for pol in ("blind","rank"):
                    np.testing.assert_array_equal(results[("old",pol)],np.fromfile(src,"<f8",count=market.ne))
            newcap.astype("<i4").tofile(scratch/"cap.bin")
            with (scratch/"direct.bin").open("wb") as dst:
                for pol in ("blind","rank"): results[("sum",pol)].astype("<f8").tofile(dst)
            subprocess.run([exe,numeric/str(year),scratch/"cap.bin",scratch/"direct.bin",str(year),scratch/"direct.csv"],check=True)
            result=pd.read_csv(scratch/"direct.csv")
            direct_tables.append(result)
            target=reference.loc[reference.year.eq(year)].reset_index(drop=True)
            col_changes=[c for c in target if not result[c].equals(target[c])]
            entry=dict(year=year,maximum_blind_logit_error=float(np.max(np.abs(results[("sum","blind")]-expanded["blind"]))),
                       maximum_rank_logit_error=float(np.max(np.abs(results[("sum","rank")]-expanded["rank"]))),
                       differing_aggregate_columns=col_changes)
            assert set(col_changes) <= {"edited_vs_each_other","blind_rows_vs_base","rank_rows_vs_base"}
            assert entry["maximum_blind_logit_error"] < 1e-12 and entry["maximum_rank_logit_error"] < 1e-12
            info.append(entry)
            print(entry,flush=True)
    pd.concat(direct_tables,ignore_index=True).to_csv(args.direct,index=False)
    args.out.write_text(json.dumps(info,indent=2)+"\n")


if __name__=="__main__":
    main()
