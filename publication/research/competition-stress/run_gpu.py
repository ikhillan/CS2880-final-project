"""Run independent parallel-round renter-proposing DA on an integer-rank archive."""
import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

N = 64
MISSING = 64
MASKED_RANK = 127


def solve(rank, priority, visibility):
    batch, n, m = rank.shape
    assert n == m == N
    pref = torch.where(visibility, rank, MASKED_RANK).argsort(dim=2,stable=True)
    degree = visibility.sum(2)
    held = torch.full((batch,m), MISSING,device=rank.device,dtype=torch.long)
    next_choice = torch.zeros_like(held)
    county = torch.arange(n, device=rank.device).view(1,n)
    by_listing = priority.transpose(1,2)
    empty = torch.full_like(held,MASKED_RANK)
    candidate_best = empty.clone()
    # Strict receiver rankings have unique holders. Their rank uniquely
    # identifies the proposer, even when the choice ties with a retained holder.
    rank_to_buyer = by_listing.argsort(dim=2,stable=True)
    for _ in range(n*m+1):
        held_by_buyer = torch.zeros((batch,n+1),device=rank.device,dtype=torch.bool)
        held_by_buyer.scatter_(1,held,True)
        active = (~held_by_buyer[:,:n]) & (next_choice < degree)
        if not active.any().item():
            break
        proposals = pref.gather(2,next_choice.clamp(max=m-1).unsqueeze(2)).squeeze(2)
        next_choice += active.long()
        pr = priority.gather(2,proposals.unsqueeze(2)).squeeze(2).long()
        prev_rank = by_listing.gather(2,held.clamp(max=n-1).unsqueeze(2)).squeeze(2).long()
        candidate_best.copy_(torch.where(held==MISSING,empty,prev_rank))
        candidate_best.scatter_reduce_(1,proposals,torch.where(active,pr,empty),reduce='amin',include_self=True)
        wins = rank_to_buyer.gather(2,candidate_best.clamp(max=n-1).unsqueeze(2)).squeeze(2)
        held = torch.where(candidate_best==MASKED_RANK,torch.full_like(wins,MISSING),wins)
    else:
        raise RuntimeError('DA exceeded number of eligible pairs')
    buyer = torch.full((batch,n+1),MISSING,device=rank.device,dtype=torch.long)
    buyer.scatter_(1,held,county.expand(batch,n))
    matched = buyer[:,:n]
    held_rank = by_listing.gather(2,held.clamp(max=n-1).unsqueeze(2)).squeeze(2)
    held_rank = torch.where(held==MISSING, 126, held_rank)
    buyer_rank = rank.gather(2,matched.clamp(max=m-1).unsqueeze(2)).squeeze(2)
    buyer_rank = torch.where(matched==MISSING, 126,buyer_rank)
    blockers = (rank<buyer_rank[:,:,None]) & (priority<held_rank[:,None,:])
    assert not (blockers & visibility).any().item(), 'blocking visible pair'
    in_graph = visibility.gather(2,matched.clamp(max=m-1).unsqueeze(2)).squeeze(2)
    assert (in_graph | (matched==MISSING)).all().item(), 'invisible match'
    return matched, blockers.any(2)


def sha(path):
    with path.open('rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ranks',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--chunk',type=int,default=18)
    p.add_argument('--device',default='cuda')
    p.add_argument('--cross-prices',action='store_true',help='Cross latent (outer axis) and observable policy price weights, fixing sigma at 0.5.')
    a=p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    start=time.perf_counter()
    config=json.loads(a.ranks.with_suffix('.json').read_text())
    assert sha(a.ranks)==config['sha256']
    z=dict(np.load(a.ranks)); device=torch.device(a.device)
    outcomes=[]; comparisons=0
    total=len(z['lower'])
    for offset in range(0,total,a.chunk):
        end=min(total,offset+a.chunk); chunk=end-offset
        def ten(k):
            return torch.from_numpy(z[k][offset:end]).to(device)
        rank=ten('pref'); priority=ten('seller'); lower=ten('lower')
        if a.cross_prices:
            rank=rank[:,::3].repeat_interleave(3,dim=1)
        d=(priority<127).sum(2)
        quota=torch.stack([d.clamp(max=5),(d*2+4)//5],dim=1)
        u = ten('uniform')[:,None,:,:]<quota[:,:,:,None]
        t=ten('transparent')[:,:,None,:,:]<quota[:,None,:,:,None]
        t=t.repeat(1,3,1,1,1) if a.cross_prices else t.repeat_interleave(3,dim=1)
        uu=u[:,None,:,:,:].expand(-1,9,-1,-1,-1)
        ut=torch.where(lower[:,None,None,:,None],uu,t)
        masks=torch.stack([uu,ut,t],dim=3)
        assert (masks.sum(-1)==quota[:,None,:,None,:]).all().item()
        shape=(chunk,9,2,3,N,N)
        r=rank[:,:,None,None,:,:].expand(shape).reshape(-1,N,N)
        s=priority[:,None,None,None,:,:].expand(shape).reshape(-1,N,N)
        matched,blocked=solve(r,s,masks.reshape(-1,N,N))
        matched=matched.reshape(shape[:-1]); blocked=blocked.reshape(shape[:-1])
        if offset<len(z['reference']):
            ncheck=min(len(z['reference'])-offset,chunk)
            if a.cross_prices:
                assert np.array_equal(matched[:ncheck,::4].cpu().numpy(),z['reference'][offset:offset+ncheck,::3])
                comparisons+=ncheck*3*2*3
            else:
                assert np.array_equal(matched[:ncheck].cpu().numpy(),z['reference'][offset:offset+ncheck])
                comparisons+=ncheck*9*2*3
        matched = matched!=MISSING
        lo=lower[:,None,None,None,:]
        exposed=(lower[:,None,:] & (d[:,None,:]>0) & (d[:,None,:]==quota))
        same=exposed[:,None,:,:].expand(-1,9,-1,-1)
        both=lo.expand_as(matched)
        assert not (blocked & same[:,:,:,None,:]).any().item()
        low_counts=torch.stack([torch.sum(both & matched, dim=-1),torch.sum(~both & matched,dim=-1),
                                torch.sum(both & blocked,dim=-1),torch.sum(~both & blocked,dim=-1)],-1).cpu().numpy()
        losses=(matched[:,:,:,0,:] & ~matched[:,:,:,1,:])
        gains=(~matched[:,:,:,0,:] & matched[:,:,:,1,:])
        trans=torch.stack([(losses & same).sum(-1),(gains & same).sum(-1)],-1).cpu().numpy()
        exposed_n=exposed.sum(-1).cpu().numpy(); low=lower.sum(-1).cpu().numpy()
        for i in range(chunk):
            code=offset+i
            cell=code%12
            blk=code//24
            design=(code//12)%2
            for g in range(9):
                for q in range(2):
                    row=[blk,config['designs'][design],config['months'][cell//4],config['boroughs'][cell%4],
                         config['prices'][g//3],(config['prices'] if a.cross_prices else config['noises'])[g%3],('k5','q40')[q],
                         int(low[i]),int(N-low[i]),int(exposed_n[i,q])]
                    outcomes.append(row+low_counts[i,g,q].flatten().tolist()+trans[i,g,q].tolist())
    head=['block','design','month','borough','w_p','policy_w_p' if a.cross_prices else 'sigma_r','budget','lower_n','higher_n','fully_visible_lower_n']
    head += [arm+'_'+out for arm in ('UU','UT','TT') for out in ('matched_lower','matched_higher','blocked_lower','blocked_higher')]
    head += ['fully_visible_lower_lost','fully_visible_lower_gained']
    with gzip.open(a.out/'counts.csv.gz','wt',newline='') as f:
        w=csv.writer(f);w.writerow(head);w.writerows(outcomes)
    meta=dict(host=platform.node(),job=os.getenv('SLURM_JOB_ID'),partition=os.getenv('SLURM_JOB_PARTITION'),
              crossed_prices=a.cross_prices,
              device=torch.cuda.get_device_name() if a.device=='cuda' else a.device,torch=str(torch.__version__),
              cuda=str(torch.version.cuda),seconds=time.perf_counter()-start,
              matched_vectors_checked_against_reference=comparisons,
              allocations_checked_full_visible_stability=total*9*2*3,markets=total,sha256_ranks=sha(a.ranks))
    (a.out/'compute.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)


if __name__=='__main__':
    main()
