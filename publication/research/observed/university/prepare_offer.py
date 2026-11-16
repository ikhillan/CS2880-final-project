"""Export only pre-allocation numeric features; no identifiers or test labels leave this preparation."""
import argparse
import json
from pathlib import Path
import numpy as np
from run import Market,history

def make(root,out):
    out.mkdir(parents=True,exist_ok=True)
    prev=Market(root,2024,positive_scores=True);metadata={}
    for yr in (2025,2026):
        train=yr==2025
        m=Market(root,yr,positive_scores=True)
        table,gcut=history(prev);ht=table.reindex(m.program)
        missing=~np.isfinite(ht.cut);cut=np.where(missing,gcut,ht.cut)
        pressure=ht.pressure.fillna(table.pressure.median()).to_numpy()
        feat=np.column_stack(((m.score-650)/100,(m.score-cut[m.school])/80,pressure[m.school]/3,
                              np.log1p(m.cap[m.school])/5,np.asarray(missing)[m.school],np.log(m.r)/3))
        # Columns are independently permuted per row: no positional rank leaks into the blind model.
        rng=np.random.default_rng(13312+yr);shuf=np.argsort(rng.random((m.n,20)),axis=1)
        col=shuf[m.a,m.pos];features=np.zeros((m.n,20,6),np.float32);present=np.zeros((m.n,20),bool)
        features[m.a,col]=feat;present[m.a,col]=True
        assert present.sum()==len(m.a)
        values={'features':features,'present':present}
        known=m.recorded>=0
        if train:
            y=np.full(m.n,20,np.uint8);y[known]=col[m.recorded[known]]
            values['target']=y
        np.savez_compressed(out/f'features{yr}.npz',**values)
        metadata[yr]=dict(**m.flow,degree_max=int(m.degree.max()))
        if not train:np.save(out/'test-col.npy',col.astype('uint8'))
        prev=m
    (out/'encoding.json').write_text(json.dumps(metadata,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('out',type=Path);a=p.parse_args();make(a.inputs,a.out)
