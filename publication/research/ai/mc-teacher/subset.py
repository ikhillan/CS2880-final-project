"""Export only inputs available at inference, in the fixed block order."""
import argparse
import json
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path); p.add_argument('output',type=Path)
    a=p.parse_args()
    keys=[]; arrays={k:[] for k in ('e','u','v')}
    for name,start in [('summer',2026120500),('winter',2026120700)]:
        m=json.loads((a.source/f'{name}.json').read_text())
        index=[i for i,key in enumerate(m['keys']) if start <= key[0] < start+8]
        assert len(index)==192
        keys.extend([[name]+m['keys'][i] for i in index])
        with np.load(a.source/f'{name}.npz') as src:
            for k in arrays: arrays[k].append(src[k][index])
    np.savez_compressed(a.output,**{k:np.concatenate(v) for k,v in arrays.items()})
    a.output.with_suffix('.json').write_text(json.dumps(keys)+'\n')
