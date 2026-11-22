"""Build private grade, contract and family records from public A1, B1, C1 and F1 only."""
import argparse
import hashlib
import io
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

def table(archive, members, prefix, cols):
    name=next(n for n in members if Path(n).name.startswith(prefix+"_") and n.endswith(".csv"))
    stream=subprocess.check_output(["bsdtar","-xOf",archive,name])
    return pd.read_csv(io.BytesIO(stream),sep=";",usecols=cols,encoding="utf-8-sig",low_memory=False)

def main(args):
    members=subprocess.check_output(["bsdtar","-tf",args.archive],text=True).splitlines()
    key=["rbd","cod_curso"]
    capacities=["vacantes_regular","vacantes_prioritarios","vacantes_pie","vacantes_alta_exigencia_t","vacantes_alta_exigencia_r"]
    flags=["prioridad_matriculado","prioridad_hermano","prioridad_hijo_funcionario","prioridad_exalumno","es_pie"]
    a=table(args.archive,members,"A1",[*key,*capacities,"vacantes","cod_nivel"])
    b=table(args.archive,members,"B1",["mrun","cod_nivel","prioritario","alto_rendimiento"]).sort_values("mrun")
    c=table(args.archive,members,"C1",[*key,*flags,"preferencia_postulante","loteria_original","orden_alta_exigencia_transicion","orden_pie","mrun"])
    f=table(args.archive,members,"F1",["mrun_1","mrun_2","es_hermano","mismo_nivel","postula_en_bloque"])
    # The frozen menu market stopped at the protected current course; retain that same envelope.
    cut=c[c.prioridad_matriculado.eq(1)].set_index("mrun").preferencia_postulante
    limits=c.mrun.map(cut)
    if not args.all_reports:
        c=c[limits.isna()|c.preferencia_postulante.le(limits)]
    c=c.sort_values(["mrun","preferencia_postulante"])
    ai=pd.MultiIndex.from_frame(a[key]); bi=pd.Index(b.mrun); assert ai.is_unique and bi.is_unique
    owners=bi.get_indexer(c.mrun); courses=ai.get_indexer(pd.MultiIndex.from_frame(c[key]))
    with args.numeric.open("rb") as base:
        n,m,ne=np.fromfile(base,"<i4",3)
        assert (n,m,ne)==(len(b),len(a),len(c))
        at=np.fromfile(base,"<i4",n+1); np.testing.assert_array_equal(owners,np.repeat(np.arange(n),np.diff(at)))
        np.testing.assert_array_equal(courses,np.fromfile(base,"<i4",ne))
    np.testing.assert_array_equal(a[capacities].sum(axis=1),a.vacantes)
    np.testing.assert_array_equal(a.cod_nivel.to_numpy()[courses],b.cod_nivel.to_numpy()[owners])
    assert c.loteria_original.notna().all()
    assert not (a.vacantes_alta_exigencia_r.gt(0)&a.vacantes_alta_exigencia_t.gt(0)).any()
    start=bi.get_indexer(f.mrun_1);end=bi.get_indexer(f.mrun_2)
    assert min(start.min(),end.min())>=0
    high=b.cod_nivel.to_numpy()[start];low=b.cod_nivel.to_numpy()[end]
    forward=(high>low) & (f.es_hermano.eq(1)|f.postula_en_bloque.eq(1)).to_numpy()
    assert ((high==low)==f.mismo_nivel.to_numpy()).all() and (high>=low).all()
    normflag=sum(c[k].eq(1).to_numpy(np.uint8)<<i for i,k in enumerate(flags))
    prec=['loteria_original','orden_pie','orden_alta_exigencia_transicion']
    values=[pd.to_numeric(c[k],errors='coerce').fillna(-1).to_numpy() for k in prec]
    for arr in values: assert np.isfinite(arr).all() and np.array_equal(arr,arr.astype('int32'))
    with args.out.open("wb") as target:
        def dump(v,t="<i4"): np.asarray(v,dtype=t).tofile(target)
        dump([n,m,ne,forward.sum()]);dump(a.rbd);dump(a.cod_nivel);dump(a[capacities].to_numpy())
        dump(b.alto_rendimiento,"u1");dump(normflag,"u1")
        for arr in values: dump(arr)
        dump(np.column_stack([start[forward],end[forward],f.es_hermano[forward],f.postula_en_bloque[forward]]))
    print(args.archive.name,'retained',len(c),'same_grade_links',sum(high==low),'higher_grade_links',forward.sum(),'special rank range',[(int(x.min()),int(x.max())) for x in values],
          'source sha256',hashlib.sha256(args.archive.read_bytes()).hexdigest(),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("archive","numeric","out"): p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--all-reports",action="store_true",help="Align to the raw full-list market, retaining entries below continuity.")
    main(p.parse_args())
