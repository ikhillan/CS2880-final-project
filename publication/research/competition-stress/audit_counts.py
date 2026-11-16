"""Build and run an independent serial C++ DA and reconcile every archived count."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--ranks',type=Path,required=True)
    cli.add_argument('--counts',type=Path,required=True)
    cli.add_argument('--crossed',action='store_true')
    args=cli.parse_args()
    meta=json.loads(args.ranks.with_suffix('.json').read_text())
    z=dict(np.load(args.ranks))
    n=len(z['lower'])
    methods=['UU','UT','TT']
    stats=['matched_lower','matched_higher','blocked_lower','blocked_higher']
    col=[f'{arm}_{stat}' for arm in methods for stat in stats]
    col+=['fully_visible_lower_lost','fully_visible_lower_gained']
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        with (root/'input.bin').open('wb') as f:
            f.write(np.array([n],'<i4').tobytes())
            for i in range(n):
                for key in ['pref','seller','uniform','transparent','lower']:
                    f.write(z[key][i].tobytes())
        subprocess.run(['c++','-std=c++17','-O3',str(Path(__file__).with_name('audit_da.cpp')),'-o',root/'audit'],check=True)
        subprocess.run([root/'audit',root/'input.bin',root/'output.bin','crossed' if args.crossed else 'primary'],check=True)
        result=np.fromfile(root/'output.bin','<u2').reshape(n,9,2,14)
    frame=pd.read_csv(args.counts)
    dims=['block','design','month','borough','w_p','policy_w_p' if args.crossed else 'sigma_r','budget']
    comp=meta['prices'] if args.crossed else meta['noises']
    ix=pd.MultiIndex.from_product([range(meta['blocks']),meta['designs'],meta['months'],meta['boroughs'],
                                   meta['prices'],comp,['k5','q40']],names=dims)
    frame=frame.set_index(dims).reindex(ix)
    assert np.array_equal(result.reshape(-1,14),frame[col].to_numpy())
    low=z['lower']; d=(z['seller']<64).sum(axis=2)
    sizes=np.broadcast_to(low.sum(1)[:,None,None],(n,9,2))
    assert np.array_equal(sizes.ravel(),frame['lower_n'].to_numpy())
    assert np.array_equal(64-sizes.ravel(),frame['higher_n'].to_numpy())
    quota=np.stack([np.minimum(5,d),(2*d+4)//5],axis=1)
    full=(low[:,None,:] & (d[:,None,:]>0) & (d[:,None,:]==quota)).sum(-1)
    assert np.array_equal(np.broadcast_to(full[:,None,:],(n,9,2)).ravel(),frame['fully_visible_lower_n'].to_numpy())
    print(f'{len(frame)} seed-cell-grid-budget rows and all 17 counts per row agree.')


if __name__=='__main__':
    main()
