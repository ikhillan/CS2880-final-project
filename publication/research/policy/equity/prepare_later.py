"""Reuse the 2019-trained coefficients and the 2019 program priors for later application years."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

FILES={2022:'SAE_2022.rar',2023:'SAE-2023.rar'}
URLS={2022:'https://datosabiertos.mineduc.cl/wp-content/uploads/2023/03/SAE_2022.rar',
      2023:'https://datosabiertos.mineduc.cl/wp-content/uploads/2024/04/SAE-2023.rar'}

def run(args):
    sys.path.insert(0,str(args.school.resolve()))
    import prepare
    prepare.FILES.update(FILES);prepare.URLS.update(URLS)
    record=json.loads((args.school/'fit-and-inputs.json').read_text())
    reg=record['models'];args.numeric.mkdir(parents=True,exist_ok=True)
    ak,_,y,_,d,_=prepare.prepare(args.old,2019);edit=d['tier']!=0;j=d['j']
    history=ak.copy()
    history['win']=np.bincount(j[edit],weights=y[edit],minlength=len(ak))
    history['n']=np.bincount(j[edit],minlength=len(ak));base=y[edit].mean()
    info=[]
    for yr in [2020,2021,2022,2023]:
        data=args.old if yr<2022 else args.new
        a=prepare.read(data,yr,'A1',prepare.AKEY+['vacantes_prioritarios'])
        np.asarray(a.vacantes_prioritarios,dtype='<i4').tofile(args.numeric/f'{yr}-reserves.bin')
        if yr<2022:continue
        ak,pos,y,x,d,entry=prepare.prepare(data,yr);j=d['j']
        h=ak.merge(history,how='left',on=prepare.AKEY,validate='one_to_one').fillna(0)
        prior=((h.win+20*base)/(h.n+20)).to_numpy()[j]
        x['log_odds_prior']=np.log(prior/(1-prior));x['prior_sq']=x.log_odds_prior.to_numpy()**2
        for name in ['blind','rank']:
            if name=='rank':
                x['log_rank']=np.log(pos)
                x['rank1']=(pos==1)*1.
                x['rank_pressure']=x.log_rank*np.log((1+np.bincount(j,minlength=d['m'])[j])/(1+d['cap'][j]))
            assert x.columns.tolist()==reg[name]['features']
            d[name]=prepare.predict(x.to_numpy(),reg[name])
        i=d['i'];n=d['n'];m=d['m']
        offset=np.r_[0,np.cumsum(np.bincount(i,minlength=n))];assert offset[-1]==len(j)
        with (args.numeric/f'{yr}.bin').open('wb') as f:
            for a in [np.array([n,m,len(j)]),offset,j,d['priority'],d['cap'],d['off'],d['sen'],d['g'],d['ret'],d['appended'],d['degree']]:
                np.asarray(a,dtype='<i4').tofile(f)
            for name in ['blind','rank']:np.asarray(d[name],dtype='<f8').tofile(f)
        info.append(entry);print(yr,entry['applicants'],entry['included_edges'],flush=True)
    out={'frozen_fit_sha256':hashlib.sha256((args.school/'fit-and-inputs.json').read_bytes()).hexdigest(),'sources':info}
    args.record.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['school','old','new','numeric','record']:p.add_argument(name,type=Path)
    run(p.parse_args())
