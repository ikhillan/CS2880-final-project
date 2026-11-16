"""Regenerate anonymous 2026 prediction files using archived offer-trained weights."""
import argparse
from pathlib import Path
import numpy as np
import torch
from train_offer import Net
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('features','models','out'):p.add_argument(n,type=Path)
    p.add_argument('--device',default='cpu');a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    with np.load(a.features/'features2026.npz') as src:
        arr=src['features'];mask=src['present'];assert 'target' not in src
    for name in ('blind','informed'):
        x=torch.as_tensor(arr.copy(),device=a.device);m=torch.as_tensor(mask,device=a.device)
        if name=='blind':x[:,:,5]=0
        for seed in [8431,8432,8433]:
            model=Net().to(a.device);model.load_state_dict(torch.load(a.models/f'{name}-{seed}.pt',map_location=a.device,weights_only=True));model.eval()
            with torch.no_grad():prob=torch.cat([model(row,vis).softmax(-1).cpu() for row,vis in zip(x.split(512),m.split(512))])
            np.save(a.out/f'{name}-{seed}.npy',prob.numpy())
