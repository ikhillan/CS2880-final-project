"""Cross-section benchmark. Select top-five from fresh DA lottery draws; evaluate at the held-out order."""
import argparse
import hashlib
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import pandas as pd


def late_tiers(archives, numeric, publication, year):
    from school_ranks import read_public
    source=next(s for s in json.loads((publication/"two-studies/robustness/sources.json").read_text())
                if str(year) in s["filename"])
    archive=archives/source["filename"]
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=source["sha256"]:
        raise ValueError("public release digest mismatch")
    key=["rbd","cod_curso"]
    flags=["prioridad_matriculado","prioridad_hermano","prioridad_hijo_funcionario","prioridad_exalumno"]
    catalog=pd.MultiIndex.from_frame(read_public(archive,"A1",key)[key])
    pupils=pd.Index(read_public(archive,"B1",["mrun"]).mrun).sort_values()
    apps=read_public(archive,"C1",["mrun","preferencia_postulante",*key,*flags])
    assert not apps.duplicated(["mrun","preferencia_postulante"]).any()
    secured=apps[apps[flags[0]].eq(1)].set_index("mrun").preferencia_postulante
    assert secured.index.is_unique and catalog.is_unique and pupils.is_unique
    maxr=apps.mrun.map(secured)
    apps=apps[maxr.isna() | apps.preferencia_postulante.le(maxr)].sort_values(["mrun","preferencia_postulante"])
    owners=pupils.get_indexer(apps.mrun)
    programs=catalog.get_indexer(pd.MultiIndex.from_frame(apps[key]))
    tiers=np.select([apps[k].eq(1).to_numpy() for k in flags],np.arange(4),default=4).astype(np.uint8)
    with (numeric/f"{year}.bin").open("rb") as file:
        n,m,ne=np.fromfile(file,"<i4",3)
        assert (n,m,ne)==(len(pupils),len(catalog),len(tiers))
        offset=np.fromfile(file,"<i4",n+1)
        np.testing.assert_array_equal(owners,np.repeat(np.arange(n),np.diff(offset)))
        np.testing.assert_array_equal(programs,np.fromfile(file,"<i4",ne))
    return tiers.tobytes()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("publication","out"):
        p.add_argument("--"+arg,required=True,type=Path)
    for arg in ("early-archives","early-numeric","late-archives","late-numeric"):
        p.add_argument("--"+arg,type=Path)
    p.add_argument("--years",nargs="+",type=int,choices=[2022,2023,2024],default=[2022,2023])
    p.add_argument("--batches",type=int,default=8)
    args=p.parse_args()
    if args.batches<1: p.error("--batches must be positive")
    if len(set(args.years))!=len(args.years): p.error("repeat year")
    if any(y<2024 for y in args.years) and (args.early_archives is None or args.early_numeric is None):
        p.error("2022 and 2023 require --early-archives and --early-numeric")
    if 2024 in args.years and (args.late_archives is None or args.late_numeric is None):
        p.error("2024 requires --late-archives and --late-numeric")
    pdir=args.publication
    school=pdir/"research/observed/school"
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory() as tmp:
        temp=Path(tmp)
        exe=temp/"simulated_menu"
        subprocess.run(["c++","-O3","-std=c++17","-I"+str(school),
                       Path(__file__).with_name("school_simulated_menu.cpp"),"-o",exe],check=True)
        def one(year):
            outputs=[]
            command=[sys.executable,pdir/"two-studies/robustness/chile/school_priority_tiers.py",
                     "--school",school,"--archives",args.early_archives,"--numeric",args.early_numeric,
                     "--cohorts",pdir/"research/policy/equity/cohorts.json","--year",str(year)]
            # The stream contains only the four public priority flags collapsed to
            # five ordered categories. No realized tie value is passed via stdin.
            tiers=late_tiers(args.late_archives,args.late_numeric,pdir,year) if year==2024 else subprocess.check_output(command)
            numeric=args.late_numeric if year==2024 else args.early_numeric
            for batch in range(args.batches):
                output=temp/f"{year}-{batch}.csv"
                subprocess.run([exe,numeric,str(year),output,str(batch)],input=tiers,check=True)
                observed=pd.read_csv(output).assign(batch=batch)
                outputs.append(observed)
            return pd.concat(outputs,ignore_index=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            result=pd.concat(list(pool.map(one,args.years)),ignore_index=True)
        result[result.batch==0].drop(columns="batch").to_csv(args.out,index=False)
        result.to_csv(args.out.with_name(args.out.stem+"-batches.csv"),index=False)
        print(result.query('policy=="simulated" and lottery_draws==64').to_string(index=False))


if __name__=="__main__":
    main()
