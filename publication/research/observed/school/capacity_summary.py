"""Summarize the two recorded follow-up capacity variants without altering frozen scores."""
import argparse
from pathlib import Path
import pandas as pd
def main(a):
    counts=[];rows=[]
    for year in (2020,2021):
        for capacity in ('raw','regular'):
            df=pd.read_csv(a.scratch/f'{year}-{capacity}.csv').assign(capacity=capacity)
            assert len(df)==20*(1+3+48)
            counts.append(df)
            measures=df.query('k==5 and group=="all"').groupby('policy').mean(numeric_only=True)
            for policy in ('blind','rank','uniform','first'):
                m=measures.loc[policy]
                rows.append(dict(year=year,capacity=capacity,policy=policy,placement=100*m.placed/m.n,observed_coverage=100*m.official_covered/m.official_assigned,replay_coverage=100*m.full_covered/m.full_assigned))
    pd.concat(counts,ignore_index=True).to_csv(a.out/'capacity-counts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(rows).to_csv(a.out/'capacity-summary.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scratch',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
