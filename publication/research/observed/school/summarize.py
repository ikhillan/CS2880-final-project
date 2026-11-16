"""Aggregate policy replay with Monte Carlo error for the random comparator."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import t

def main(a):
    x=pd.concat([pd.read_csv(f) for f in a.files],ignore_index=True)
    assert x[['year','k','policy','group','draw']].duplicated().sum()==0
    big=x[x.group=='all']
    for (yr,k,pol),f in big.groupby(['year','k','policy']):
        assert len(f)==(48 if pol=='uniform' else 1)
        assert np.array_equal(f.n.to_numpy(),x[(x.group.str.startswith('grade'))&(x.year==yr)&(x.k==k)&(x.policy==pol)].groupby('draw').n.sum().to_numpy())
        assert (f[['full_covered','official_covered']].to_numpy()<=f[['full_assigned','official_assigned']].to_numpy()).all()
    x.to_csv(a.out/'counts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    denom={'placed':'n','on_reported':'n','full_covered':'full_assigned','official_covered':'official_assigned','identical_official':'n'}
    rows=[]
    for (year,k,group),frame in x.groupby(['year','k','group']):
        df=frame.groupby('policy').mean(numeric_only=True)
        for target,den in denom.items():
            numer=frame[target]
            frac=100*numer/frame[den].replace(0,np.nan)
            if k==0:
                rows.append(dict(year=year,k=k,group=group,comparison='full',metric=target,rate=frac.iloc[0],delta=np.nan,mc95=np.nan,n=int(df.loc['full','n'])))
                continue
            u=frac[frame.policy=='uniform'];ck=t.ppf(.975,len(u)-1)*u.std(ddof=1)/np.sqrt(len(u))
            for name in ('blind','rank','first','uniform'):
                val=frac[frame.policy==name].mean()
                rows.append(dict(year=year,k=k,group=group,comparison=name,metric=target,rate=val,delta=val-u.mean(),mc95=ck,n=int(df.loc[name,'n'])))
            r=100*df.loc['rank',target]/df.loc['rank',den];b=100*df.loc['blind',target]/df.loc['blind',den]
            rows.append(dict(year=year,k=k,group=group,comparison='rank_minus_blind',metric=target,rate=np.nan,delta=r-b,mc95=np.nan,n=int(df.loc['rank','n'])))
    pd.DataFrame(rows).to_csv(a.out/'summary.csv',index=False,float_format='%.9g')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('files',nargs='+',type=Path);main(p.parse_args())
