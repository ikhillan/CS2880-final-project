"""Build capacity-only sensitivity, leaving frozen scores and all realized priorities unchanged."""
import argparse
from pathlib import Path
import numpy as np
from prepare import read,AKEY
from verify import read_numeric

def main(a):
    for year in (2020,2021):
        prep=read(a.data,year,'A1',AKEY+['vacantes','vacantes_regular'])
        at,j,prior,cap,off,ret,deg,blind,rank=read_numeric(a.scratch/f'{year}.bin')
        rc=np.bincount(ret[ret>=0],minlength=len(prep))
        assert np.array_equal(cap,np.maximum(rc,prep.vacantes))
        for name,v in [('regular',np.maximum(rc,prep.vacantes_regular)),('raw',prep.vacantes)]:
            np.asarray(v,dtype='<i4').tofile(a.scratch/f'{year}-{name}.bin')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);main(p.parse_args())
