"""Train on simulated full-exposure DA targets and freeze each validation-selected model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WIDTH=32
SEEDS=(8101,8102,8103)


class PartnerNet(nn.Module):
    def __init__(self,context):
        super().__init__()
        self.context=context
        self.encoder=nn.Sequential(nn.Linear(5,WIDTH),nn.SiLU(),nn.Linear(WIDTH,WIDTH),nn.SiLU())
        if context:
            self.messages=nn.ModuleList(nn.Sequential(nn.Linear(3*WIDTH+2,WIDTH),nn.SiLU(),nn.Linear(WIDTH,WIDTH)) for _ in range(2))
        self.edge_head=nn.Sequential(nn.Linear(WIDTH,WIDTH),nn.SiLU(),nn.Linear(WIDTH,1))
        # Outside odds can depend on own options in either model; only context pools competitors.
        self.outside_head=nn.Sequential(nn.Linear(WIDTH+1,WIDTH),nn.SiLU(),nn.Linear(WIDTH,1))
        self.register_buffer('mu',torch.zeros(5));self.register_buffer('sd',torch.ones(5))

    def forward(self,x,e):
        x=torch.clamp((x-self.mu)/self.sd,-12,12)
        h=self.encoder(x)
        m=e.unsqueeze(-1); dr=m.sum(2).clamp_min(1); dc=m.sum(1).clamp_min(1)
        if self.context:
            for layer in self.messages:
                masked=h*m
                renters=(masked.sum(2)/dr).unsqueeze(2).expand_as(h)
                listings=(masked.sum(1)/dc).unsqueeze(1).expand_as(h)
                log_dr=dr.log1p().unsqueeze(2).expand_as(m)/4
                log_dc=dc.log1p().unsqueeze(1).expand_as(m)/4
                h=F.silu(h+layer(torch.cat((h,renters,listings,log_dr,log_dc),-1)))
        logits=self.edge_head(h).squeeze(-1).masked_fill(~e,-1e4)
        own=(h*m).sum(2)/dr
        outside=self.outside_head(torch.cat((own,dr.log1p()/4),-1))
        return torch.cat((logits,outside),dim=-1)


def load(path,device,labels=True):
    # Explicitly whitelist input tensors. Realized ranks/priority/group never enter the forward pass.
    with np.load(path) as a:
        x=torch.as_tensor(a['x'],device=device);e=torch.as_tensor(a['e'],device=device)
        y=torch.as_tensor(a['y'],dtype=torch.long,device=device) if labels else None
    return x,e,y


def sha(path):
    with Path(path).open('rb') as src: return hashlib.file_digest(src,'sha256').hexdigest()


def train(name,seed,base,out,device):
    tic=time.monotonic();torch.manual_seed(seed);np.random.seed(seed)
    xt,et,yt=load(base/'train.npz',device); xv,ev,yv=load(base/'val.npz',device)
    model=PartnerNet(name=='context').to(device)
    xe=xt[et];model.mu.copy_(xe.mean(0));model.sd.copy_(xe.std(0).clamp_min(.05));del xe
    optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    best=float('inf');path=out/f'{name}-{seed}.pt';trace=[]
    for epoch in range(28):
        model.train(); train_sum=torch.zeros((),device=device)
        for idx in torch.randperm(len(xt),device=device).split(32):
            optim.zero_grad(set_to_none=True);pred=model(xt[idx],et[idx]);loss=F.cross_entropy(pred.flatten(0,1),yt[idx].flatten())
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3.0);optim.step();train_sum+=loss.detach()*len(idx)
        model.eval();vsum=torch.zeros((),device=device)
        with torch.no_grad():
            for x,e,y in zip(xv.split(32),ev.split(32),yv.split(32)):
                vsum+=F.cross_entropy(model(x,e).flatten(0,1),y.flatten(),reduction='sum')
        val=(vsum/yv.numel()).item(); trace.append({'epoch':epoch+1,'train':train_sum.item()/len(xt),'val':val})
        if val<best:
            best=val;best_epoch=epoch+1
            torch.save({'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'name':name,'seed':seed,'epoch':best_epoch},path)
    rec=dict(name=name,seed=seed,best_epoch=best_epoch,best_val=best,epochs=trace,sha256=sha(path),
             train_sha=sha(base/'train.npz'),val_sha=sha(base/'val.npz'),code_sha=sha(__file__),wall_seconds=time.monotonic()-tic,
             device=torch.cuda.get_device_name() if device=='cuda' else 'cpu',torch=torch.__version__,cuda=torch.version.cuda,
             python=platform.python_version(),host=socket.gethostname(),job=os.getenv('SLURM_JOB_ID'))
    path.with_suffix('.json').write_text(json.dumps(rec,indent=2)+'\n');print(name,seed,best_epoch,best,flush=True)


def infer(base,out,device,splits):
    paths=[out/f'{name}-{s}.pt' for name in ('pointwise','context') for s in SEEDS]; hashes={p.name:sha(p) for p in paths}
    for split in splits:
        x,e,_=load(base/f'{split}.npz',device,labels=False);result={}
        for name in ('context','pointwise'):
            members=[]
            for seed in SEEDS:
                model=PartnerNet(name=='context').to(device)
                model.load_state_dict(torch.load(out/f'{name}-{seed}.pt',map_location=device,weights_only=True)['state']);model.eval()
                with torch.no_grad():
                    members.append(torch.cat([model(xx,ee).log_softmax(-1).cpu() for xx,ee in zip(x.split(24),e.split(24))]))
            logp=(torch.stack(members).logsumexp(0)-np.log(len(members))).numpy();result[name]=logp[...,:64];result[name+'_outside']=logp[...,64]
        path=base/f'{split}-scores.npz';np.savez_compressed(path,**result)
        path.with_suffix('.json').write_text(json.dumps(dict(weights=hashes,data_sha=sha(base/f'{split}.npz'),prediction_sha=sha(path),code_sha=sha(__file__)),indent=2)+'\n')
        print('predicted',split,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','infer']);p.add_argument('--model',choices=['pointwise','context']);p.add_argument('--seed',type=int)
    p.add_argument('--data',type=Path,default=Path('.'));p.add_argument('--out',type=Path,default=Path('models'));p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--splits',nargs='+',default=['cal','summer','winter']);a=p.parse_args()
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');a.out.mkdir(parents=True,exist_ok=True)
    if a.mode=='train': train(a.model,a.seed,a.data,a.out,a.device)
    else: infer(a.data,a.out,a.device,a.splits)


if __name__=='__main__': main()
