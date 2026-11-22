"""Link regular-round applicant identifiers to April schools, without providing labels to the engine."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

def prepare(args):
    sys.path.insert(0,str(args.publication/'two-studies/behavior/field'))
    from round_link import read_file
    from link_register import enrollment_register
    members=subprocess.check_output(['bsdtar','-tf',args.archive],text=True).splitlines()
    pupils=read_file(args.archive,members,'B1',['mrun']).mrun.sort_values()
    schools=read_file(args.archive,members,'A1',['rbd']).rbd
    register,_=enrollment_register(args.register,pupils,args.year+1)
    labels=register.RBD.reindex(pd.Index(pupils)).fillna(-1)
    with args.out.open('wb') as f:
        schools.to_numpy('<i4').tofile(f);labels.to_numpy('<i4').tofile(f)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('publication','archive','register','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--year',type=int,required=True)
    prepare(p.parse_args())
