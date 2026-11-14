"""Use public archive tables directly to validate the later blind/rank contrasts.

Shares only grouped counts. Requires bsdtar with RAR support, NumPy and pandas.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


def table(archive, label, fields):
    members = subprocess.check_output(["bsdtar", "-tf", archive], text=True).splitlines()
    member, = [p for p in members if Path(p).name.startswith(label + "_") and p.endswith(".csv")]
    data = subprocess.check_output(["bsdtar", "-xOf", archive, member])
    return pd.read_csv(io.BytesIO(data), sep=";", encoding="utf-8-sig", usecols=fields)


def verify(year, archives, numeric, published):
    archive = archives / f"SAE_{year}.rar"
    applicants = table(archive, "B1", ["mrun","prioritario","cod_nivel"]).set_index("mrun").sort_index()
    offers = table(archive, "D1", ["mrun","rbd_admitido","cod_curso_admitido","cod_nivel"]).set_index("mrun")
    assert applicants.index.is_unique and offers.index.is_unique
    assert applicants.index.symmetric_difference(offers.index).empty
    offers = offers.loc[applicants.index]
    assert applicants.cod_nivel.equals(offers.cod_nivel)
    schools = table(archive, "A1", ["rbd","cod_curso"])
    apps = table(archive, "C1", ["mrun","rbd","cod_curso","preferencia_postulante","prioridad_matriculado","loteria_original"])
    cutoffs = apps.loc[apps.prioridad_matriculado.eq(1)].set_index("mrun").preferencia_postulante
    assert cutoffs.index.is_unique
    limits = apps.mrun.map(cutoffs)
    apps = apps.loc[limits.isna() | apps.preferencia_postulante.le(limits)].sort_values(["mrun","preferencia_postulante"])
    kids = applicants.index.get_indexer(apps.mrun)
    ids = pd.MultiIndex.from_frame(schools)
    courses = ids.get_indexer(pd.MultiIndex.from_frame(apps[["rbd","cod_curso"]]))
    assert ids.is_unique and np.all(courses >= 0)
    assert not apps.duplicated(["mrun","rbd","cod_curso"]).any()
    # The official course must occur in the A1 lookup even when C1 omits it.
    given = offers[["rbd_admitido","cod_curso_admitido"]].apply(pd.to_numeric,errors="coerce")
    off = ids.get_indexer(pd.MultiIndex.from_frame(given))
    assert np.array_equal(off >= 0, given.rbd_admitido.notna() & given.rbd_admitido.ge(0))
    with (numeric/f"{year}.bin").open("rb") as f:
        n,m,ne=map(int,np.fromfile(f,dtype="<i4",count=3))
        offsets=np.fromfile(f,dtype="<i4",count=n+1)
        np.testing.assert_array_equal(offsets[1:],np.cumsum(np.bincount(kids,minlength=n)))
        np.testing.assert_array_equal(np.fromfile(f,dtype="<i4",count=ne),courses)
        f.seek((ne+m)*4,1)
        np.testing.assert_array_equal(np.fromfile(f,dtype="<i4",count=n),off)
        np.testing.assert_array_equal(np.fromfile(f,dtype="<i4",count=n),applicants.prioritario)
    same = apps.rbd.to_numpy() == given.rbd_admitido.to_numpy()[kids]
    same &= apps.cod_curso.to_numpy() == given.cod_curso_admitido.to_numpy()[kids]
    result = []
    for pol in ["full","blind","rank"]:
        mask = np.fromfile(numeric/f"{year}-{pol}.mask",dtype=np.uint8)
        z=np.fromfile(numeric/f"{year}-{pol}.partners",dtype="<i4")
        assert np.isin(mask,[0,1]).all() and len(mask)==len(apps) and len(z)==len(applicants)
        covered=np.bincount(kids[(mask==1)&same],minlength=n)
        assert (covered<=1).all()
        for group, select in [("all",np.ones(n,bool)),("priority",applicants.prioritario.eq(1).to_numpy()),("other",applicants.prioritario.eq(0).to_numpy())]:
            counts=dict(n=int(select.sum()), historical_offered=int(np.sum((off>=0)&select)),
                        official_shown=int(covered[select].sum()), placed=int(np.sum((z>=0)&select)),
                        agreement=int(np.sum((off==z)&select)))
            expected=published.loc[(year,group,pol)]
            assert all(counts[k]==expected[k] for k in counts), (year,group,pol,counts)
            result.append(dict(year=year,group=group,policy=pol,**counts))
    return dict(year=year,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),eligible_applications=len(apps),
                repeated_published_orders_within_course=int(apps.duplicated(["rbd","cod_curso","loteria_original"]).sum()),
                missing_published_orders=int(apps.loteria_original.isna().sum()),direct_counts=result)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ["archives","numeric","paper","out"]:
        parser.add_argument("--"+arg, type=Path, required=True)
    args=parser.parse_args()
    fit=args.paper.parent/"research/observed/school/fit-and-inputs.json"
    frozen_hash=hashlib.sha256(fit.read_bytes()).hexdigest()
    assert frozen_hash==json.loads((args.paper.parent/"research/policy/equity/cohorts.json").read_text())["frozen_fit_sha256"]
    assert frozen_hash==json.loads((args.paper/"evidence/later-cohorts.json").read_text())["frozen_fit_sha256"]
    reference=pd.read_csv(args.paper/"evidence/later-school-results.csv").set_index(["year","group","policy"])
    r={"fit_sha256":frozen_hash,"years":[verify(y,args.archives,args.numeric,reference) for y in (2024,2025)]}
    args.out.write_text(json.dumps(r,indent=2)+"\n")
    print("Verified 18 year/group/policy rows.")
