"""Make a full C1 report market for historical offer validation. It contains no offer or enrollment labels."""
import argparse
import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

def prepare(args):
    sys.path.insert(0, str(args.publication/'two-studies/behavior/field'))
    from round_link import read_file, RELEASES
    archive=args.archives/RELEASES[args.year]
    files=subprocess.check_output(['bsdtar','-tf',archive],text=True).splitlines()
    a=read_file(archive,files,'A1',['rbd','cod_curso','vacantes','cod_nivel'])
    b=read_file(archive,files,'B1',['mrun','cod_nivel','prioritario']).sort_values('mrun')
    flags=['prioridad_matriculado','prioridad_hermano','prioridad_hijo_funcionario','prioridad_exalumno']
    c=read_file(archive,files,'C1',['mrun','rbd','cod_curso','preferencia_postulante','agregada_por_continuidad','loteria_original',*flags]).sort_values(['mrun','preferencia_postulante'])
    ids=pd.Index(b.mrun)
    courses=pd.MultiIndex.from_frame(a[['rbd','cod_curso']]); prog=courses.get_indexer(pd.MultiIndex.from_frame(c[['rbd','cod_curso']]))
    own=ids.get_indexer(c.mrun)
    n,m,ne=len(b),len(a),len(c)
    assert min(prog.min(),own.min())>=0 and ids.is_unique and courses.is_unique
    counts=np.bincount(own,minlength=n)
    at=np.r_[0,np.cumsum(counts)]
    sec=c[flags[0]].eq(1).to_numpy()
    fallback=np.full(n,-1,dtype='<i4'); fallback[own[sec]]=prog[sec]
    appended=np.full(n,-1,dtype='<i4');extra=c.agregada_por_continuidad.eq(1);appended[own[extra]]=prog[extra]
    tier=np.select([c[k].eq(1).to_numpy() for k in flags],range(4),4)
    order=np.lexsort((own,c.loteria_original.to_numpy(),tier,prog))
    prio=np.empty(ne,dtype='<i4');prio[order]=np.arange(ne)
    capacity=np.maximum(a.vacantes,np.bincount(prog[sec],minlength=m))
    keep=np.ones(ne,dtype=bool)
    sec_pos=np.full(n,ne);sec_pos[own[sec]]=np.flatnonzero(sec)
    keep &= np.arange(ne)<=sec_pos[own]
    with args.frozen.open('rb') as f:
        num=np.fromfile(f,'<i4',3)
        assert num.tolist()==[n,m,int(keep.sum())]
        offsets=np.fromfile(f,'<i4',n+1);np.testing.assert_array_equal(np.diff(offsets),np.bincount(own[keep],minlength=n))
        np.testing.assert_array_equal(prog[keep],np.fromfile(f,'<i4',num[2]))
    with args.out.open('wb') as f:
        for array in ([n,m,ne],at,prog,prio,capacity,np.full(n,-1),b.prioritario,b.cod_nivel,fallback,appended,counts-(fallback>=0)):
            np.asarray(array,dtype='<i4').tofile(f)
        np.zeros(ne,dtype='<f8').tofile(f);np.zeros(ne,dtype='<f8').tofile(f)
    print(args.year,ne,'original entries;',int((~keep).sum()),'below a recorded continuity course;',len(np.unique(own[~keep])),'affected reports')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archives','publication','frozen','out'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--year',type=int,required=True)
    prepare(p.parse_args())
