"""Same-parameter edge/own-row ablation; training and checkpoint code are shared."""
import argparse
from pathlib import Path
import json

import numpy as np
import torch
from torch.nn import functional as F
import train as training


class OwnRowNet(training.PartnerNet):
    def __init__(self,context):
        super().__init__(True)

    def forward(self,x,e):
        x=torch.clamp((x-self.mu)/self.sd,-12,12);h=self.encoder(x);m=e.unsqueeze(-1);dr=m.sum(2).clamp_min(1)
        for layer in self.messages:
            renter=((h*m).sum(2)/dr).unsqueeze(2).expand_as(h)
            degree=dr.log1p().unsqueeze(2).expand_as(m)/4
            h=F.silu(h+layer(torch.cat((h,renter,h,degree,degree),-1)))
        logits=self.edge_head(h).squeeze(-1).masked_fill(~e,-1e4)
        outside=self.outside_head(torch.cat(((h*m).sum(2)/dr,dr.log1p()/4),-1))
        return torch.cat((logits,outside),dim=-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','infer']);p.add_argument('--seed',type=int);p.add_argument('--data',type=Path,default=Path('.'));p.add_argument('--models',type=Path,default=Path('models'));a=p.parse_args()
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');device='cuda';a.models.mkdir(exist_ok=True)
    reference=training.PartnerNet(True);model=OwnRowNet(True)
    assert {k:p.shape for k,p in reference.named_parameters()}=={k:p.shape for k,p in model.named_parameters()}
    npar=sum(p.numel() for p in model.parameters());training.PartnerNet=OwnRowNet
    if a.mode=='train':
        training.train('matched',a.seed,a.data,a.models,device)
        path=a.models/f'matched-{a.seed}.json';rec=json.loads(path.read_text());rec.update(ablation_sha=training.sha(__file__),parameters=npar);path.write_text(json.dumps(rec,indent=2)+'\n')
    else:
        for split in ('summer','winter'):
            x,e,_=training.load(a.data/f'{split}.npz',device,labels=False);members=[];weights={}
            for seed in training.SEEDS:
                path=a.models/f'matched-{seed}.pt';weights[path.name]=training.sha(path);model=OwnRowNet(True).to(device);model.load_state_dict(torch.load(path,map_location=device,weights_only=True)['state']);model.eval()
                with torch.no_grad():members.append(torch.cat([model(xx,ee).log_softmax(-1).cpu() for xx,ee in zip(x.split(24),e.split(24))]))
            arr=(torch.stack(members).logsumexp(0)-np.log(len(members))).numpy();path=a.data/f'{split}-matched.npz';np.savez_compressed(path,matched=arr[...,:64],matched_outside=arr[...,64]);rec=dict(weights=weights,ablation_sha=training.sha(__file__),common_train_sha=training.sha(training.__file__),data_sha=training.sha(a.data/f'{split}.npz'),prediction_sha=training.sha(path),parameters=npar);path.with_suffix('.json').write_text(json.dumps(rec,indent=2)+'\n')


if __name__=='__main__': main()
