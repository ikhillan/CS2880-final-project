"""Fixed-epoch cohort transfer with or without the applicant's submitted ranks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(6,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.opt=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        self.out=nn.Sequential(nn.Linear(33,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,x,m):
        h=self.enc(x);d=m.sum(1,keepdim=True)
        row=(h*m[:,:,None]).sum(1)/d.clamp_min(1)
        logits=self.opt(torch.cat((h,row[:,None].expand_as(h)),-1)).squeeze(-1).masked_fill(~m,-10000.)
        return torch.cat((logits,self.out(torch.cat((row,d.log1p()/3),-1))),-1)

def run(task,data,out,device):
    name=('blind','informed')[task//3];seed=8431+task%3
    tic=time.perf_counter();torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    def load(yr):
        with np.load(data/f'features{yr}.npz') as f:
            values=[torch.as_tensor(f[k],device=device) for k in ('features','present')]
            if name=='blind':values[0][:,:,5]=0
            if yr==2025:values.append(torch.as_tensor(f['target'],device=device,dtype=torch.long))
            else: assert 'target' not in f
            return values
    X,M,Y=load(2025); model=Net().to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    losslog=[]
    for epoch in range(12):
        losses=torch.zeros((),device=device)
        for idx in torch.randperm(len(X),device=device).split(512):
            optim.zero_grad(set_to_none=True); loss=F.cross_entropy(model(X[idx],M[idx]),Y[idx])
            loss.backward();optim.step();losses+=loss.detach()*len(idx)
        losslog.append(float(losses.item()/len(X)))
    out.mkdir(exist_ok=True,parents=True)
    weights=out/f'{name}-{seed}.pt';torch.save(model.to('cpu').state_dict(),weights)
    model.to(device).eval();xt,mt=load(2026)
    with torch.no_grad():pred=torch.cat([model(x,m).softmax(-1).cpu() for x,m in zip(xt.split(512),mt.split(512))]).numpy()
    # Choose score for each reported alternative later locally using the frozen mapping.
    np.save(out/f'{name}-{seed}.npy',pred)
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=dict(model=name,seed=seed,train_loss=losslog,train_n=len(X),test_n=len(xt),
                 train_features_sha=h(data/'features2025.npz'),test_features_sha=h(data/'features2026.npz'),
                 weights=h(weights),seconds=time.perf_counter()-tic,parameters=sum(v.numel() for v in model.parameters()),
                 device=torch.cuda.get_device_name() if device=='cuda' else 'cpu',job=os.getenv('SLURM_JOB_ID'),torch=str(torch.__version__))
    (out/f'{name}-{seed}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',type=int,choices=range(6));p.add_argument('--device',default='cuda');p.add_argument('--data',type=Path,default=Path('.'));p.add_argument('--out',type=Path,default=Path('fit'));a=p.parse_args();run(a.task,a.data,a.out,a.device)
