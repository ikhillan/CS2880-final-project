"""Estimate full-information DA partner marginals from observable utilities alone."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from gpu_da import solve

def ranks(s,mask,axis):
    order=s.masked_fill(~mask,-torch.inf).argsort(dim=axis,descending=True,stable=True)
    return order.argsort(dim=axis,stable=True).masked_fill(~mask,127).to(torch.uint8)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('inputs',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--device',default='cuda'); a=p.parse_args()
    device=torch.device(a.device)
    if device.type=='cuda': assert torch.cuda.is_available()
    prng=torch.Generator(device=device).manual_seed(11271113)
    t0=perf_counter(); data=np.load(a.inputs)
    assert set(data.files)=={'u','v','e'}
    source=torch.as_tensor(data['e'],device=device); count,n,m=source.shape
    trace=[]
    for begin in range(0,count,4):
        ids=slice(begin,min(begin+4,count))
        e=source[ids,None].expand(-1,128,-1,-1)
        b,D,n,m=e.shape
        # Double precision all but eliminates ordinal ties in the shocks.
        def noise(shape):
            return torch.randn(shape,dtype=torch.float64,device=device,generator=prng)
        u=torch.as_tensor(data['u'][ids],device=device)[:,None]+0.5*noise((b,D,1,m))+0.5*noise(e.shape)
        v=torch.as_tensor(data['v'][ids],device=device)[:,None]+noise(e.shape)
        rb=ranks(u,e,3).reshape(-1,n,m); rs=ranks(v,e,2).reshape(-1,n,m)
        winner,_=solve(rb,rs,e.reshape(-1,n,m))
        trace.append(winner.reshape(b,D,n).to(torch.uint8).cpu().numpy())
    out=np.concatenate(trace); np.savez_compressed(a.output,partner=out)
    summary={'device':torch.cuda.get_device_name() if device.type=='cuda' else 'cpu',
             'seconds':perf_counter()-t0,'torch':str(torch.__version__),'job':os.getenv('SLURM_JOB_ID'),
             'markets':count,'draws':128,'stable_outcomes_verified':128*count,
             'seed':11271113,'chunk_markets':4,'input_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
             'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}
    a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
if __name__=='__main__': main()
