"""Independently join predictions back to the unmodified D1 initial offer. Aggregate output only."""
import argparse
import csv
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def main(args):
    sys.path.insert(0,str(args.publication/'two-studies/behavior/field'))
    from round_link import read_file, RELEASES
    archive=args.archives/RELEASES[args.year]
    members=subprocess.check_output(['bsdtar','-tf',archive],text=True).splitlines()
    schools=read_file(archive,members,'A1',['rbd','cod_curso'])
    pupils=read_file(archive,members,'B1',['mrun','prioritario']).sort_values('mrun')
    d1=read_file(archive,members,'D1',['mrun','rbd_admitido','cod_curso_admitido']).set_index('mrun')
    assert pupils.mrun.is_unique and d1.index.is_unique and set(d1.index)==set(pupils.mrun)
    d1=d1.reindex(pupils.mrun).replace(r'^\s*$',np.nan,regex=True).apply(pd.to_numeric,errors='raise')
    assert d1.rbd_admitido.isna().equals(d1.cod_curso_admitido.isna())
    key=pd.MultiIndex.from_frame(d1.fillna(-1).astype('int64'))
    official=pd.MultiIndex.from_frame(schools[['rbd','cod_curso']]).get_indexer(key)
    assert np.array_equal(official<0,d1.rbd_admitido.isna())
    proj=np.fromfile(args.programs,dtype='<i4')
    n=len(official)
    assert len(proj)==n and proj.min()>=-1 and proj.max()<len(schools)
    c=read_file(archive,members,'C1',['mrun','rbd','cod_curso','preferencia_postulante','prioridad_matriculado'])
    edges=c.merge(pupils[['mrun']].assign(person=np.arange(n)),on='mrun',how='left',validate='many_to_one')
    idx=edges.person.to_numpy()
    position=edges.preferencia_postulante.to_numpy()
    consent=edges.prioridad_matriculado.eq(1).to_numpy()
    cutoff=np.full(n,np.iinfo(np.int32).max);cutoff[idx[consent]]=position[consent]
    beyond=position>cutoff[idx]
    prog=pd.MultiIndex.from_frame(schools[['rbd','cod_curso']]).get_indexer(pd.MultiIndex.from_frame(edges[['rbd','cod_curso']]))
    too_low=np.zeros(n,dtype=bool);too_low[idx[beyond]]=True
    official_below=np.zeros(n,dtype=bool)
    official_below[idx[beyond&(official[idx]==prog)]]=True
    family=read_file(archive,members,'F1',['mrun_1','mrun_2','postula_en_bloque','mismo_nivel'])
    younger=set(family.loc[family.postula_en_bloque.eq(1)&family.mismo_nivel.eq(0),'mrun_2'])
    tab=[]
    for name,mask in [('all',np.ones(n,dtype=bool)),('submitted_below_continuity',too_low),
                      ('official_below_continuity',official_below),('younger_block',pupils.mrun.isin(younger).to_numpy()),
                      ('no_official_offer',official<0),('official_offer',official>=0)]:
        cols={'year':args.year,'report':args.report,'group':name,'applicants':int(mask.sum())}
        for field,occur in [('same_course_or_outside',proj==official),('same_offer_status',(proj>=0)==(official>=0)),('official_only',(proj<0)&(official>=0)),('replay_only',(proj>=0)&(official<0)),('both_placed_different_course',(proj>=0)&(official>=0)&(proj!=official))]:
            cols[field]=int((mask&occur).sum())
        assert cols['same_course_or_outside']+sum(cols[k] for k in ('official_only','replay_only','both_placed_different_course'))==cols['applicants']
        tab.append(cols)
    with args.out.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=tab[0],lineterminator='\n')
        writer.writeheader();writer.writerows(tab)
    print(tab[0])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',choices=['policy_pool','all_submitted'],required=True)
    p.add_argument('--year',type=int,required=True)
    for k in ('publication','archives','programs','out'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
