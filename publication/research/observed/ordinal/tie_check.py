"""Repeat the held-out comparison with menu-level ties randomized over replicates."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.stats import t

from benchmark import FEATURE_NAMES, QUOTAS, da_batch, features, mask_order, partner_frequency, read_profiles, refine, rng


def max_size(graph):
    rows=[np.flatnonzero(x).tolist() for x in graph]
    held=[-1]*graph.shape[1]
    def insert(i, seen):
        for j in rows[i]:
            if j in seen:
                continue
            seen.add(j)
            if held[j]<0 or insert(held[j],seen):
                held[j]=i
                return 1
        return 0
    return sum(insert(i,set()) for i in range(len(rows)))


def summary(values, denominators):
    avg=np.sum(values)/sum(denominators)
    variance=np.std((np.asarray(values)-avg*np.asarray(denominators))/np.mean(denominators),ddof=1)
    margin=t.ppf(.975,len(values)-1)*variance/np.sqrt(len(values))
    return [100*x for x in [avg,avg-margin,avg+margin]]


def main(source,out):
    s=2
    m=read_profiles(source)[s]
    e=m[2]
    fits=json.loads((out/"fits.json").read_text())
    x=features(m)
    scores={"own":-m[0],"conditional":partner_frequency(s,m,220)[:,:-1]}
    for fit in fits:
        if fit["test"]==3:
            terms=list(fit["coefficients"])
            scores[fit["model"]]=x[:,:,[FEATURE_NAMES.index(k) for k in terms]] @ np.array(list(fit["coefficients"].values()))
    cached={}
    diag=[]
    fixed_key=rng(s,240).random(e.shape)
    for b in range(32):
        n=256
        ranks,pri=refine(m,n,rng(s,721,b))
        full=da_batch(ranks,pri,e)
        lottery=rng(s,723,b).random(ranks.shape)
        uorder=np.argsort(np.argsort(np.where(e,lottery,2),axis=2),axis=2)
        menu_key=rng(s,724,b).random(ranks.shape)
        for k in QUOTAS:
            for mode in ("fixed","fresh"):
                if mode=="fixed" and k!=5:
                    continue
                key=np.broadcast_to(fixed_key,ranks.shape) if mode=="fixed" else menu_key
                arms={p:mask_order(np.broadcast_to(e,ranks.shape),np.broadcast_to(val,ranks.shape),key,k) for p,val in scores.items()}
                arms["uniform"]=e & (uorder<k)
                for p,mask in arms.items():
                    assert (mask.sum(-1)==np.minimum(e.sum(1),k)).all()
                    alloc=da_batch(ranks,pri,mask)
                    hit=(full>=0) & np.take_along_axis(mask,np.maximum(0,full)[:,:,None],axis=2)[:,:,0]
                    row=dict(block=b,mode=mode,k=k,arm=p,total=full.size,ref=int((full>=0).sum()),hit=int(hit.sum()),placed=int((alloc>=0).sum()))
                    if k==5:
                        max_counts=[]
                        for j,g in enumerate(mask):
                            token=np.packbits(g).tobytes()
                            if token not in cached:cached[token]=max_size(g)
                            size=cached[token]
                            if b==0 and j<24:
                                assert size == (maximum_bipartite_matching(csr_matrix(g),perm_type="column")>=0).sum()
                            assert size>=(alloc[j]>=0).sum()
                            max_counts.append(size)
                        row.update(visible=int(mask.any(1).sum()),max_place=sum(max_counts),shortfall=sum(max_counts)-row["placed"])
                    diag.append(row)
    comparisons=[]
    for mode in ["fixed","fresh"]:
        for k in QUOTAS:
            df={(p,b):r for r in diag if r["mode"]==mode and r["k"]==k for p,b in [(r["arm"],r["block"])]}
            if not df:continue
            for p in list(scores)+["uniform"]:
                for ref in ["level","uniform","graph"]:
                    if p==ref: continue
                    row={"mode":mode,"k":k,"arm":p,"reference":ref}
                    for key,denom in [("hit","ref"),("placed","total"),("max_place","total"),("shortfall","total"),("visible","total")]:
                        if key not in df[p,0]:continue
                        values=[df[p,b][key]-(0 if ref=="level" else df[ref,b][key]) for b in range(32)]
                        denoms=[df[p,b][denom] for b in range(32)]
                        row[key],row[key+"_lo"],row[key+"_hi"]=summary(values,denoms)
                    comparisons.append(row)
    for filename, data in [("sensitivity-blocks.csv",diag),("sensitivity.csv",comparisons)]:
        fields=list(dict.fromkeys(k for row in data for k in row))
        with (out/filename).open("w",newline="") as f:
            wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(data)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path(__file__).parent)
    a=parser.parse_args();main(a.source,a.out)
