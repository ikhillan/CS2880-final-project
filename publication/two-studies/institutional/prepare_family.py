"""Build private school/family groups for the lottery sensitivity from A1, B1, F1."""
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def main(args):
    sys.path.insert(0,str(args.publication/'two-studies/behavior/field'))
    from round_link import RELEASES, read_file
    path=args.archives/RELEASES[args.year]
    names=subprocess.check_output(['bsdtar','-tf',path],text=True).splitlines()
    b=read_file(path,names,'B1',['mrun']).sort_values('mrun')
    a=read_file(path,names,'A1',['rbd'])
    f=read_file(path,names,'F1',['mrun_1','mrun_2','es_hermano','postula_en_bloque'])
    import pandas as pd
    pupils=pd.Index(b.mrun)
    x=pupils.get_indexer(f.mrun_1);y=pupils.get_indexer(f.mrun_2)
    assert pupils.is_unique and min(x.min(),y.min())>=0
    # F1 includes links by shared guardian as well as blood relationships.
    # Using every F1 relationship makes the assumed grouping explicit.
    graph=coo_matrix((np.ones(len(f),np.int8),(x,y)),shape=(len(b),len(b))).tocsr()
    nfamilies,family=connected_components(graph,directed=False)
    with args.numeric.open('rb') as stream:
        n,m,_=np.fromfile(stream,'<i4',3)
    assert (n,m)==(len(pupils),len(a))
    groups=np.arange(len(a)) if args.course_specific else a.rbd
    with args.out.open('wb') as stream:
        for values in ([n,m],family,groups):
            np.array(values,dtype='<i4').tofile(stream)
    print(args.year,'applicants',n,'public F1 components',nfamilies,'reported links',len(f),
          'linked by neither blood nor block flag',int(((f.es_hermano==0)&(f.postula_en_bloque==0)).sum()))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--year',type=int,required=True)
    p.add_argument('--course-specific',action='store_true',help='Order families independently at each course.')
    for name in ('archives','numeric','out','publication'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
