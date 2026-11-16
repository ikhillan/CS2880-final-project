"""Maximum matching vs DA cardinality on the exact archived rental shortlist menus."""
import argparse
import csv
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def panel(split, data, learned, methods):
    source = data / f"{split}-scores.npz"
    if not source.exists():
        source = learned / "predictions" / source.name
    with np.load(source) as z:
        score = z["context"]
    with np.load(data / f"{split}.npz") as z:
        eligible, ranks, priority, uniform = (z[k] for k in ["e", "pref", "seller", "uniform"])
    meta = json.loads((data / f"{split}.json").read_text())
    keys = meta["keys"]
    assert len(keys) == len(score) == len(eligible) == 1152
    with gzip.open(learned / f"{split}-counts.csv.gz", "rt", newline="") as f:
        observed = {(int(r["seed"]), r["design"], r["month"], r["borough"], r["policy"]): int(r["matched_total"])
                    for r in csv.DictReader(f) if r["rollout"] == "all" and r["policy"] in ("uniform", "context")}
    assert len(observed) == 2*len(keys)
    rows = []
    masks = {
        "uniform": uniform < 5,
        "context": np.stack([methods.make_mask(x, e, key, False) for x, e, key in zip(score, eligible, keys)]),
    }
    for arm, displays in masks.items():
        assert not (displays & ~eligible).any()
        assert np.array_equal(displays.sum(2), np.minimum(5, eligible.sum(2)))
        placements = (methods.da(ranks, priority, displays) < eligible.shape[2]).sum(1)
        for ix, (key, graph, placed) in enumerate(zip(keys, displays, placements)):
            maximum = int((maximum_bipartite_matching(csr_matrix(graph), perm_type="column") >= 0).sum())
            # A maximum-weight complete assignment with real arcs of weight one
            # and all other arcs of weight zero gives an independent certificate.
            i,j = linear_sum_assignment(graph.astype(np.int8), maximize=True)
            assert maximum == int(graph[i,j].sum())
            assert placements[ix] == observed[(*key,arm)] <= maximum
            rows.append(dict(split=split, seed=key[0], design=key[1], month=key[2], borough=key[3], policy=arm,
                             renters=graph.shape[0], max_size=maximum, assigned=int(placed), gap=maximum-int(placed)))
    return pd.DataFrame(rows), meta


def summaries(records, meta):
    """Original estimator: pool counts inside cell, then average by known borough weight."""
    results=[]
    for design, df in records.groupby("design"):
        cells=sorted(set(zip(df.month,df.borough)))
        weights=np.array([meta["weights"][b] for _,b in cells])
        weights/=weights.sum()
        seeds=sorted(df.seed.unique())
        rng=np.random.default_rng(441903)
        freq=np.vstack([np.ones(len(seeds)),rng.multinomial(len(seeds), np.repeat(1/len(seeds),len(seeds)),size=5000)])
        vals={}
        for pol,g in df.groupby("policy"):
            index=pd.MultiIndex.from_tuples([(s,m,b) for s in seeds for m,b in cells],names=["seed","month","borough"])
            arr=g.set_index(["seed","month","borough"])[["max_size","assigned","gap","renters"]].reindex(index)
            assert not arr.isna().any().any()
            arr=arr.to_numpy().reshape(len(seeds),len(cells),4)
            totals=np.einsum("bs,scd->bcd",freq,arr,optimize=True)
            vals[pol]=100*np.einsum("bcd,c->bd",totals[:,:,:3]/totals[:,:,3:4],weights)
        vals["context_minus_uniform"]=vals["context"]-vals["uniform"]
        for lab,arr in vals.items():
            assert np.allclose(arr[:,0]-arr[:,2],arr[:,1])
            row={"split":df.split.iloc[0],"design":design,"contrast":lab}
            for k,met in enumerate(["visible_capacity","DA_placement","shortfall"]):
                row[met]=arr[0,k]
                row[met+"_lo"],row[met+"_hi"]=np.quantile(arr[1:,k],[.025,.975])
            results.append(row)
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True,help="Saved rental market NPZ and JSON panels")
    p.add_argument("--learned",type=Path,required=True,help="Archived learned directory with evaluate.py, predictions, and counts")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location("rental_methods",a.learned/"evaluate.py")
    methods=importlib.util.module_from_spec(spec);spec.loader.exec_module(methods)
    frames=[];stats=[]
    for split in ["summer","winter"]:
        df,meta=panel(split,a.data,a.learned,methods)
        frames.append(df);stats.extend(summaries(df,meta))
    a.out.mkdir(parents=True,exist_ok=True)
    pd.concat(frames).to_csv(a.out/"nyc-market-capacity.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    pd.DataFrame(stats).to_csv(a.out/"nyc-summary.csv",index=False)
