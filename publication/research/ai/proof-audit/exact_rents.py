"""Rebuild simulator asking rents from first-mover catalogues and exact market keys."""
import argparse
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np


def extract(task):
    from hiddenoptions.markets import MarketSpec, build_market
    seed,design,month,borough=task
    spec=MarketSpec(borough,month,seed,donors='calibration',design=design)
    market=build_market(spec,catalog)
    return (market.listings[spec.rent_column].to_numpy(np.float64),
            market.renters.cap.to_numpy(np.float64),market.inst.E)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    root=a.root.resolve();repo=root/'work/todd-project';sys.path[:0]=[str(repo),str(repo/'scripts')]
    from hiddenoptions.markets import Catalog
    wpath=root/'outputs/ec-revision/winter/run.py';loader=importlib.util.spec_from_file_location('winter_clean',wpath)
    winter=importlib.util.module_from_spec(loader);loader.loader.exec_module(winter)
    global catalog
    catalog=Catalog()
    for split in ('summer','winter'):
        if split=='winter':
            winter.prepare(catalog,repo,root/'work/nyc-scarcity/data/firstmover',root/'work/nyc-scarcity/winter-inputs')
        keys=json.loads((a.base/f'{split}.json').read_text())['keys']
        with np.load(a.base/f'{split}.npz') as z:
            features=z['x'][:,:,:,0]
            feasible=z['e']
        rents=[]
        with mp.get_context('fork').Pool(a.workers) as pool:
            for t,(listed,cap,E) in enumerate(pool.imap(extract,keys,chunksize=4)):
                assert np.array_equal(E,feasible[t])
                exact=np.divide(listed[None,:],cap[:,None],out=np.zeros(E.shape),where=cap[:,None]>0)
                assert np.array_equal(exact[E].astype(np.float32),features[t][E])
                rents.append(listed)
        output=a.base/f'{split}-exact-rents.npz'
        np.savez_compressed(output,rent=np.stack(rents))
        print(split,len(keys),'verified eligible edges',int(feasible.sum()),flush=True)


if __name__=='__main__':main()
