"""Average the three frozen probability vectors for each architecture."""
from pathlib import Path
import argparse
import numpy as np
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fit',type=Path);a=p.parse_args()
    for name in ('blind','informed'):
        pbar=sum(np.load(a.fit/f'{name}-{s}.npy').astype(float) for s in (8431,8432,8433))/3
        np.savez_compressed(a.fit/f'{name}-ensemble.npz',p=pbar)
