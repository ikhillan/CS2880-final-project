"""Test shortlist prediction on the original mutually reported mentoring profiles."""
import argparse
import csv
import hashlib
import json
from collections import deque
from pathlib import Path
import re
from xml.etree import ElementTree as ET
import zipfile

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import t

SOURCE_HASH = "7905adc02befd67334143ee1298bacac231d538ced8c4a4de78a46cb7f06c6ad"
PROTOCOL_HASH = "886eff78a2b20faff154490af6cc57b27b09c39ffa250db4898066c9323794ed"
QUOTAS = [1, 2, 3, 5]
NONE = 1000
FEATURE_NAMES = ["intercept", "mentee_midrank", "mentor_midrank", "mentee_first", "mentor_first",
                 "inverse_mentee_degree", "inverse_mentor_degree", "other_first_demand",
                 "midrank_product", "mentee_rank_times_competitor_demand"]
GRAPH_FEATURES = [0, 5, 6]


def read_profiles(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_HASH
    with zipfile.ZipFile(path) as z:
        # The workbook's Scenario 1--3 tabs are sheets 2--4. Do not inspect results tabs.
        shared = ["".join(si.itertext()) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        book = ET.fromstring(z.read("xl/workbook.xml"))
        names = [v.attrib["name"] for v in book.findall(".//{*}sheet")]
        assert names[1:4] == ["Scenario 1", "Scenario 2", "Scenario 3"]
        profiles = []
        for scenario in range(1, 4):
            groups = [{}, {}]
            doc = ET.fromstring(z.read(f"xl/worksheets/sheet{scenario+1}.xml"))
            for row in doc.findall(".//{*}row")[1:]:
                values = {}
                for c in row.findall("{*}c"):
                    v = c.find("{*}v")
                    if v is None or not v.text:
                        continue
                    col = re.match(r"[A-Z]+", c.attrib["r"]).group()
                    values[col] = shared[int(v.text)] if c.get("t") == "s" else v.text
                side, ident = int(values["A"]), int(values["B"])
                assert ident not in groups[side]
                order = {}
                for rank, column in enumerate("CDEFG"):
                    cell = values.get(column, "").strip()
                    if not cell:
                        continue
                    for number in cell.split("-"):
                        item = int(number.strip())
                        assert item not in order, (ident, item)
                        order[item] = rank
                groups[side][ident] = order
            mentors, mentees = groups
            assert all(set(t).issubset(mentors) for t in mentees.values())
            assert all(set(t).issubset(mentees) for t in mentors.values())
            left = np.array([[mentees[i].get(j, NONE) for j in mentors] for i in mentees])
            right = np.array([[mentors[j].get(i, NONE) for j in mentors] for i in mentees])
            edge = (left < NONE) & (right < NONE)
            left[~edge] = NONE
            right[~edge] = NONE
            profiles.append((left, right, edge))
    return profiles


def rng(cohort, task, block=0):
    return np.random.default_rng(np.random.SeedSequence([28801113, cohort, task, block]))


def refine(market, count, random):
    l, r, e = market
    shape = (count,) + e.shape
    # Global integer ranks on both sides of each batch; lower is preferred.
    a = np.argsort(l + random.random(shape), axis=2)
    b = np.argsort(np.argsort(r + random.random(shape), axis=1), axis=1)
    ranks = np.argsort(a, axis=2)
    return ranks, b


def da_batch(ranks, priority, display):
    """Parallel proposal rounds; unmatched applicants are -1."""
    n_batch, n_left, n_right = ranks.shape
    shown = np.broadcast_to(display, ranks.shape)
    active_rank = np.where(shown, ranks, NONE)
    order = np.argsort(active_rank, axis=2)
    degree = shown.sum(2)
    partner = np.full((n_batch, n_left), -1)
    holder = np.full((n_batch, n_right), n_left)
    next_rank = np.zeros((n_batch, n_left), int)
    b_ix, j_ix = np.indices(holder.shape)
    missing_priority = np.full((n_batch, 1, n_right), n_left)
    ext_priority = np.concatenate((priority, missing_priority), axis=1)
    while True:
        b, i = np.nonzero((partner == -1) & (next_rank < degree))
        if len(i) == 0:
            break
        j = order[b, i, next_rank[b, i]]
        next_rank[b, i] += 1
        incumbent_prior = ext_priority[b_ix, holder, j_ix]
        best = incumbent_prior.copy()
        np.minimum.at(best, (b, j), priority[b, i, j])
        wins = priority[b, i, j] == best[b, j]
        wb, wi, wj = b[wins], i[wins], j[wins]
        old = holder[wb, wj]
        real = old < n_left
        partner[wb[real], old[real]] = -1
        holder[wb, wj] = wi
        partner[wb, wi] = wj
    old_pos = np.take_along_axis(active_rank, np.maximum(partner, 0)[:, :, None], axis=2)
    old_pos = np.where(partner[:, :, None] >= 0, old_pos, NONE)
    cutoff = ext_priority[b_ix, holder, j_ix]
    assert not (shown & (ranks < old_pos) & (priority < cutoff[:, None, :])).any()
    assert ((holder < n_left).sum(1) == (partner >= 0).sum(1)).all()
    return partner


def serial(ranks, priorities, mask):
    lists = [list(np.argsort(x)[mask[i, np.argsort(x)]]) for i, x in enumerate(ranks)]
    held = {}
    stack = deque(range(len(lists)))
    while stack:
        i = stack.popleft()
        if not lists[i]:
            continue
        j = lists[i].pop(0)
        incumbent = held.get(j)
        if incumbent is None or priorities[i, j] < priorities[incumbent, j]:
            held[j] = i
            if incumbent is not None:
                stack.append(incumbent)
        else:
            stack.append(i)
    z = np.full(len(lists), -1)
    for j, i in held.items():
        z[i] = j
    return z


def partner_frequency(cohort, market, stream, n=4096):
    edge = market[2]
    rows, cols = edge.shape
    counts = np.zeros((rows, cols + 1), int)
    for b in range(n // 256):
        ranks, priorities = refine(market, 256, rng(cohort, stream, b))
        ref = da_batch(ranks, priorities, edge)
        np.add.at(counts, (np.arange(rows)[None, :], np.where(ref >= 0, ref, cols)), 1)
    return counts / n


def midrank(rank, mask):
    """Express a position in [0,1], averaging positions when ranks tie."""
    result = np.zeros_like(rank, float)
    first = np.zeros_like(rank, bool)
    for i, row in enumerate(rank):
        inc = np.flatnonzero(mask[i])
        if inc.size:
            first[i, inc] = row[inc] == row[inc].min()
            for j in inc:
                result[i, j] = ((row[inc] < row[j]).sum() + ((row[inc] == row[j]).sum() - 1)/2) / max(1, inc.size - 1)
    return result, first


def features(market):
    left, right, e = market
    a, af = midrank(left,e)
    b_t, bf_t = midrank(right.T,e.T)
    b, bf = b_t.T, bf_t.T
    inv_l = 1 / np.maximum(1, e.sum(1)[:,None])
    inv_r = 1 / np.maximum(1, e.sum(0)[None,:])
    demand = ((af.sum(0)[None,:] - af) / np.maximum(1, e.sum(0)[None,:] - 1))
    return np.stack(np.broadcast_arrays(np.ones_like(a),a,b,af,bf,inv_l,inv_r,demand,a*b,a*demand),axis=2)


def fit(markets, xs, ys, train, indices):
    n = sum(len(ys[i]) for i in train)
    def loss(v):
        loss_sum, grad = 0., np.zeros_like(v)
        for s in train:
            X, E, Y = xs[s][:,:,indices], markets[s][2], ys[s]
            edge = np.where(E,X@v,-np.inf)
            logits = np.column_stack((edge, np.zeros(len(edge))))
            logp = logits-logsumexp(logits,axis=1)[:,None]
            lp = np.where(np.isfinite(logp),logp,0)
            loss_sum -= (Y*lp).sum()
            grad += np.einsum("nmd,nm->d",X,(np.exp(logp)-Y)[:,:-1])
        reg = np.where(np.arange(len(v)) == 0, 0.,v)
        return loss_sum/n + .0005*(reg@reg), grad/n + .001*reg
    out=minimize(loss,np.zeros(len(indices)),method="L-BFGS-B",jac=True,options={"ftol":1e-13,"gtol":1e-8})
    assert out.success
    return out.x


def mask_order(e, score, key, k):
    # Lexicographic: the common independent lottery breaks exact score ties.
    order = np.lexsort((key,-score,~e),axis=-1)
    return e & (np.argsort(order,axis=-1) < k)


def main(source,out):
    assert hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest() == PROTOCOL_HASH
    out.mkdir(exist_ok=True,parents=True)
    markets = read_profiles(source)
    features_all = list(map(features, markets))
    fit_label = [partner_frequency(s,m,130) for s,m in enumerate(markets)]
    cond = [partner_frequency(s,m,220) for s,m in enumerate(markets)]
    ds, records, fits, blocks = [],[],[],[]
    for s,market in enumerate(markets):
        left,right,e = market
        ds.append(dict(cohort=s+1,mentees=len(e),mentors=e.shape[1],active_mentees=int(e.any(1).sum()),
                       active_mentors=int(e.any(0).sum()),mutual_edges=int(e.sum()),
                       mentees_over_five=int((e.sum(1)>5).sum())))
        train = [i for i in range(3) if i!=s]
        scores = {"own":-left,"conditional":cond[s][:,:-1]}
        for name,ind in [("fitted",list(range(len(FEATURE_NAMES)))),("graph",GRAPH_FEATURES)]:
            coef=fit(markets,features_all,fit_label,train,ind)
            scores[name]=features_all[s][:,:,ind]@coef
            fits.append(dict(test=s+1,model=name,training=[i+1 for i in train],
                             coefficients=dict(zip([FEATURE_NAMES[i] for i in ind],coef.tolist()))))
        key = rng(s,240).random(e.shape)
        # Let the baseline see reported own ties only. The model sees the full weak profile.
        fixed = {k:{p:mask_order(e,sc,key,k) for p,sc in scores.items()} for k in QUOTAS}
        for b in range(32):
            n=256
            ranks, priority = refine(market,n,rng(s,330,b))
            full = da_batch(ranks,priority,e)
            lottery = rng(s,440,b).random(ranks.shape)
            lottery_order = np.argsort(np.argsort(np.where(e,lottery,2),axis=2),axis=2)
            ii=np.arange(e.shape[0])[None,:]; bb=np.arange(n)[:,None]
            for k in QUOTAS:
                arms=dict(fixed[k],uniform=e & (lottery_order<k))
                for arm,mask in arms.items():
                    mm=np.broadcast_to(mask,ranks.shape)
                    assert np.all(mm.sum(2) == np.minimum(e.sum(1),k))
                    alloc=da_batch(ranks,priority,mm)
                    if b == 0:
                        for q in range(40):
                            assert np.array_equal(serial(ranks[q],priority[q],mm[q]),alloc[q])
                    hit = (full>=0) & mm[bb,ii,np.maximum(full,0)]
                    assert ((full>=0)&(alloc<0)).sum(1).max() <= e.shape[0]
                    assert (((full>=0)&(alloc<0)).sum(1) <= ((full>=0)&~hit).sum(1)).all()
                    blocks.append(dict(cohort=s+1,block=b,k=k,arm=arm,total=full.size,ref=int((full>=0).sum()),
                                       hits=int(hit.sum()),placed=int((alloc>=0).sum()),lost=int(((full>=0)&(alloc<0)).sum()),
                                       gained=int(((full<0)&(alloc>=0)).sum()),any_loss=int(((full>=0)&(alloc<0)).any(1).sum()),
                                       any_gain=int(((full<0)&(alloc>=0)).any(1).sum())))
        for k in QUOTAS:
            def entries(name):
                return [r for r in blocks if r["cohort"]==s+1 and r["k"]==k and r["arm"]==name]
            for name in list(scores)+["uniform"]:
                arm,base=entries(name),entries("uniform")
                record=dict(cohort=s+1,k=k,arm=name)
                for numerator,denominator,keyname in (("hits","ref","recovery"),("placed","total","placement")):
                    refs=np.array([r[denominator] for r in arm],float)
                    x=np.array([r[numerator] for r in arm],float)
                    u=np.array([r[numerator] for r in base],float)
                    record[keyname]=x.sum()/refs.sum()
                    change=(x-u).sum()/refs.sum()
                    # Ratio-estimator MC uncertainty via block influence functions.
                    influence=(x-u-change*refs)/refs.mean()
                    half=t.ppf(.975,len(refs)-1)*np.std(influence,ddof=1)/np.sqrt(len(refs))
                    record[keyname+"_delta_pp"]=100*change
                    record[keyname+"_lo_pp"]=100*(change-half)
                    record[keyname+"_hi_pp"]=100*(change+half)
                records.append(record)
    for file,rows in [("results.csv",records),("blocks.csv",blocks),("support.csv",ds)]:
        with (out/file).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    (out/"fits.json").write_text(json.dumps(fits,indent=2)+"\n")
    print(ds)
    print([r for r in records if r["cohort"]==3 and r["k"]==5])

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path(__file__).parent)
    args=parser.parse_args()
    main(args.source,args.out)
