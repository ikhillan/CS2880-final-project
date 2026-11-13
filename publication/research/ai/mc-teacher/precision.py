"""Estimate the same winner probabilities with more draws; retain sufficient counts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from gpu_da import solve
from run import ranks

def main():
    p=argparse.ArgumentParser()
    p.add_argument('inputs',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    t=perf_counter();out={512:[],2048:[]}
    with np.load(args.inputs) as data:
        assert set(data.files)=={'u','v','e'}
        U,V,E=[torch.as_tensor(data[k],device='cuda') for k in ('u','v','e')]
    rng=torch.Generator(device='cuda').manual_seed(11271271)
    def normal(shape):
        return torch.randn(shape,dtype=torch.float64,device='cuda',generator=rng)
    for start in range(0,len(E),4):
        ix=slice(start,min(start+4,len(E)))
        e=E[ix,None].expand(-1,128,-1,-1);b,d,n,m=e.shape
        counts=torch.zeros((b,n,m+1),dtype=torch.int32,device='cuda')
        for rep in range(1,17):
            u=U[ix,None]+.5*normal((b,d,1,m))+.5*normal(e.shape)
            v=V[ix,None]+normal(e.shape)
            rb=ranks(u,e,3).reshape(-1,n,m);rs=ranks(v,e,2).reshape(-1,n,m)
            target,_=solve(rb,rs,e.reshape(-1,n,m))
            target=target.reshape(b,d,n).transpose(1,2)
            counts.scatter_add_(2,target,torch.ones_like(target,dtype=torch.int32))
            if rep*d in out:
                assert (counts.sum(2)==rep*d).all()
                out[rep*d].append(counts.cpu().numpy().astype(np.uint16))
    np.savez_compressed(args.output,**{'mc'+str(k):np.concatenate(v) for k,v in out.items()})
    receipt=dict(outcomes=len(E)*2048,markets=len(E),device=torch.cuda.get_device_name(),job=os.getenv('SLURM_JOB_ID'),
                 torch=str(torch.__version__),seconds=perf_counter()-t,seed=11271271,
                 input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                 output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
