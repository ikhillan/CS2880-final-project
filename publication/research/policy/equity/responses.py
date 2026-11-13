"""Coverage of accepted initial offers and released post-waitlist destinations, fixed menus only."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from transitions import Market
from prepare_later import FILES,URLS

def main(args):
    sys.path.insert(0,str(args.school.resolve()))
    import prepare
    prepare.FILES.update(FILES);prepare.URLS.update(URLS)
    rows=[]
    for yr in [2022,2023]:
        m=Market(args.numeric/f'{yr}.bin')
        programs=prepare.read(args.raw,yr,'A1',prepare.AKEY)
        applicants=prepare.read(args.raw,yr,'B1',['mrun']).sort_values('mrun').mrun.to_numpy()
        resp=prepare.read(args.raw,yr,'D1',['mrun','rbd_admitido','cod_curso_admitido','rbd_admitido_post_resp',
                          'cod_curso_admitido_post_resp','respuesta_postulante']).sort_values('mrun')
        assert np.array_equal(applicants,resp.mrun.to_numpy())
        lookup=pd.MultiIndex.from_frame(programs)
        ds={}
        for ref,suf in [('initial',''),('post_waitlist','_post_resp')]:
            keys=resp[['rbd_admitido'+suf,'cod_curso_admitido'+suf]].apply(pd.to_numeric,errors='coerce').fillna(-1).astype('int64')
            ds[ref]=lookup.get_indexer(pd.MultiIndex.from_frame(keys));assert np.array_equal(ds[ref]<0,keys.iloc[:,0].to_numpy()<0)
        assert np.array_equal(ds['initial'],m.off)
        response=pd.to_numeric(resp.respuesta_postulante,errors='coerce').fillna(-1).astype(int).to_numpy()
        assert set(np.unique(response))<={-1,1,2,3,4,5,6,7}
        for pol in ['full','blind','rank']:
            shown=np.fromfile(args.numeric/f'{yr}-{pol}.mask',dtype=np.uint8).astype(bool)
            for ref,target in ds.items():
                hit=(target[m.i]==m.j)&shown
                for label,choose in [('all',np.ones(m.n,bool)),('explicit_accept',np.isin(response,[1,2]))]+[
                                       (str(code),response==code) for code in np.unique(response)]:
                    den=choose&(target>=0)
                    rows.append(dict(year=yr,policy=pol,reference=ref,initial_response=label,group_size=int(choose.sum()),
                                     applicants_with_target=int(den.sum()),target_covered=int((hit&choose[m.i]).sum())))
    pd.DataFrame(rows).to_csv(args.out,index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('school','raw','numeric','out'):p.add_argument(n,type=Path)
    main(p.parse_args())
