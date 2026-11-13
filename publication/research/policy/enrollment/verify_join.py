"""Independently replay the enrollment join via raw program-key merges and row ranks."""
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('inputs','enrollment','features','fit','expected'):p.add_argument(n,type=Path)
    a=p.parse_args()
    apps=pd.read_csv(next((a.inputs/'2026').glob('C_POST*.csv')),sep=';',usecols=['MRUN','TIPO_PREF','COD_CARRERA_PREF','ORDEN_PREF','ESTADO_PREF'])
    reg=pd.read_csv(next(a.enrollment.glob('D_MAT*.csv')),sep=';',usecols=['MRUN','CODIGO_CARRERA'])
    apps=apps[apps.TIPO_PREF=='REGULAR'];people=np.sort(apps.MRUN.unique())
    all_reg=reg.merge(pd.DataFrame({'MRUN':people}),on='MRUN',how='inner',validate='one_to_one')
    # Unchanged public status graph was independently proven equal to positive score support.
    e=apps[apps.ESTADO_PREF.isin([24,25,26])].sort_values(['MRUN','ORDEN_PREF']).reset_index(drop=True)
    e['person']=np.searchsorted(people,e.MRUN)
    e['degree']=e.groupby('MRUN').MRUN.transform('size')
    join=all_reg.merge(e[['MRUN','COD_CARRERA_PREF']].reset_index(),left_on=['MRUN','CODIGO_CARRERA'],right_on=['MRUN','COD_CARRERA_PREF'],how='inner',validate='one_to_one')
    result={'registered_in_regular':len(all_reg),'within_support':len(join)}
    published=pd.read_csv(a.expected/'enrollment.csv').query('group=="all_registrants"').set_index('policy')
    col=np.load(a.features/'test-col.npy')
    for model in ['blind','informed']:
        with np.load(a.fit/f'{model}-ensemble.npz') as scores:e['score']=scores['p'][e.person,col]
        e['tie']=np.random.default_rng(118844).permutation(len(e))
        d=e.sort_values(['person','score','tie'],ascending=[True,False,True])
        pos=d.groupby('person',sort=False).cumcount()
        got=d.loc[pos<5,['MRUN','COD_CARRERA_PREF']]
        matches=all_reg.merge(got,left_on=['MRUN','CODIGO_CARRERA'],right_on=['MRUN','COD_CARRERA_PREF'],how='inner',validate='one_to_one')
        assert published.loc[model,'covered']==len(matches)
        result[model]=len(matches)
    print(json.dumps(result,indent=2))
