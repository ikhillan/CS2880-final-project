"""Keep original pre-outcome training features; change only the training labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

def make(args):
    sys.path.insert(0,str(args.previous.resolve()))
    from run import Market
    m=Market(args.inputs,2025,positive_scores=True)
    applicant=pd.read_csv(next((args.inputs/'2025').glob('C_POST*.csv')),sep=';',usecols=['MRUN','TIPO_PREF'])
    people=pd.Index(np.sort(applicant.loc[applicant.TIPO_PREF=='REGULAR','MRUN'].unique()))
    reg=pd.read_csv(next(args.enrollment.glob('*.csv')),sep=';',usecols=['MRUN','CODIGO_CARRERA']).drop_duplicates()
    registered=reg.MRUN.nunique();ambiguous=reg.MRUN.duplicated(keep=False)
    unique=reg.loc[~ambiguous];i=people.get_indexer(unique.MRUN);j=pd.Index(m.program).get_indexer(unique.CODIGO_CARRERA)
    keep=(i>=0)&(j>=0)
    e=pd.Index(m.a.astype(np.int64)*m.m+m.school).get_indexer(i[keep].astype(np.int64)*m.m+j[keep])
    good=e>=0
    col=np.argsort(np.random.default_rng(13312+2025).random((m.n,20)),axis=1)[m.a,m.pos]
    y=np.full(m.n,20,np.uint8);y[i[keep][good]]=col[e[good]]
    original=args.features/'features2025.npz'
    with np.load(original) as z:
        expected=np.full(m.n,20,np.uint8);got=m.recorded>=0;expected[got]=col[m.recorded[got]]
        assert np.array_equal(expected,z['target'])
        assert np.all(z['present'][m.a,col])
        args.out.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(args.out/'features2025.npz',features=z['features'],present=z['present'],target=y)
    h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    rec=dict(original_features_sha=h(original),enrollment_training_sha=h(args.out/'features2025.npz'),
             raw_unique_registrants=registered,ambiguous_registrants=int(reg[ambiguous].MRUN.nunique()),train_n=m.n,
             positive_enrolled_target=int((y<20).sum()),outside_target=int((y==20).sum()),changed_labels=int((y!=expected).sum()))
    (args.out/'train-labels.json').write_text(json.dumps(rec,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('previous','inputs','enrollment','features','out'):p.add_argument(name,type=Path)
    make(p.parse_args())
