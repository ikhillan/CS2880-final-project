"""Independently rebuild development n=64 scarcity results from both raw input sources."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
import json
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

root = Path(__file__).resolve().parent
data = root / 'data'
geo = pd.read_csv(data / 'acs' / 'tract_to_puma_2020.txt', dtype=str)
county = {'005':'Bronx', '047':'Brooklyn', '061':'Manhattan', '081':'Queens', '085':'Staten Island'}
geo = geo[(geo.STATEFP=='36') & geo.COUNTYFP.isin(county)]
areas = geo.drop_duplicates(['PUMA5CE', 'COUNTYFP'])
assert not areas.PUMA5CE.duplicated().any()
puma_borough = dict(zip(areas.PUMA5CE, areas.COUNTYFP.map(county)))
with ZipFile(data / 'acs' / 'csv_hny.zip') as z, z.open('psam_h36.csv') as f:
    h=pd.read_csv(f,usecols=['SERIALNO','PUMA','TEN','TYPEHUGQ','NP','MV','WGTP','ADJINC','HINCP'],
                  dtype={'PUMA':str,'SERIALNO':str})
h['borough']=h.PUMA.map(puma_borough)
h=h[(h.TYPEHUGQ==1)&(h.TEN==3)&(h.WGTP>0)&(h.ADJINC>0)&(h.NP>0)&h.borough.notna()].dropna().copy()
h['microincome']=(h.HINCP*h.ADJINC).astype('int64')
incomes=h.groupby('microincome').WGTP.sum().sort_index()
cutoff=int(incomes.index[np.searchsorted(incomes.cumsum().values,incomes.sum()/2)])
assert (len(h),h.WGTP.sum(),cutoff)==(71908,2184936,64435014000)
h=h[h.MV==1].copy()
assert (len(h),h.WGTP.sum())==(10302,299382)
h=h[h.SERIALNO.map(lambda sid:int(sha256(('ACS2020-2024|'+sid).encode()).hexdigest()[:16],16)%10<7)].copy()
h['group']=np.where(h.microincome<=cutoff,'lower','higher')
h['budget']=h.microincome.clip(lower=0)/1e6*333.918/313.689/30.
h['minbeds']=((h.NP-1)//2).astype(int)
reference=pd.read_csv(root/'local-results'/'degree_cells.csv')
reference=reference[(reference['n']==64)&reference['available']].set_index(['month','borough','group','price_floor_july'])
result=[]
for month,cpi in {'2026-06':333.952,'2026-07':333.918,'2026-08':334.980}.items():
    ls=pd.read_csv(data/'firstmover'/f'{month}.csv',dtype=str,keep_default_na=False)
    ls=ls[ls.state.str.strip().eq('NY') & ls.borough.str.strip().isin(county.values())].copy()
    ls['borough']=ls.borough.str.strip()
    ls['rent']=pd.to_numeric(ls.price,errors='coerce')*(333.918/cpi)
    ls['beds']=pd.to_numeric(ls.bedrooms,errors='coerce')
    ls=ls[(ls.rent>0)&ls.id.str.strip().ne('')&np.isfinite(ls.rent)&ls.beds.ge(0)&ls.beds.mod(1).eq(0)]
    ls=ls.sort_values(['created_at_utc','id'],kind='stable').drop_duplicates('id',keep='first')
    for col in ['street','unit']:
        ls[col]=ls[col].str.upper().str.replace('[^A-Z0-9]','',regex=True)
    keyrows=ls.street.ne('') & ~ls.unit.isin(['','NA','NONE','UNKNOWN','NULL'])
    ls=ls.loc[~(keyrows & ls.duplicated(['borough','zip_code','street','unit']))]
    for floor in [0,1000]:
        for b,g in ls[ls.rent>=floor].groupby('borough'):
            L=len(g)
            if L<64:continue
            for inc,households in h[h.borough.eq(b)].groupby('group'):
                D=np.empty(len(households),dtype=int)
                for minbeds in np.unique(households.minbeds):
                    sel=households.minbeds.eq(minbeds)
                    ordered=np.sort(g.loc[g.beds>=minbeds,'rent'].to_numpy())
                    D[sel]=np.searchsorted(ordered,households.loc[sel,'budget'],side='right')
                p=np.vstack([D==0,hypergeom.pmf(0,L,D,64),hypergeom.sf(5,L,D,64),hypergeom.sf(1,L,D,64)])
                rates=np.average(p,axis=1,weights=households.WGTP)
                ref=reference.loc[month,b,inc,floor]
                assert L==ref.catalogue_L and households.WGTP.sum()==ref.donor_weight
                assert np.allclose(rates,ref[['full_p0','draw_p0','draw_p6plus','draw_p2plus']].to_numpy(float),rtol=5e-12,atol=2e-16),(month,b,inc,floor,rates,ref)
                result.append(dict(month=month,borough=b,group=inc,floor=floor,w=int(households.WGTP.sum()),fullp0=rates[0],p0=rates[1],p6=rates[2],p2=rates[3]))
r=pd.DataFrame(result)
print('Compared',len(r),'borough-month-income-floor cells to independent raw-data/Scipy replay.')
print(r[(r.group=='lower') & (r.floor==0)].to_string(index=False,float_format=lambda x:f'{x:.11g}'))
print(r[(r.group=='lower') & (r.floor==0)].groupby('borough')[['fullp0','p0','p6','p2']].mean().to_string(float_format=lambda x:f'{x:.11g}'))
print(h.groupby(['borough','group']).WGTP.agg(['count','sum']).to_string())
