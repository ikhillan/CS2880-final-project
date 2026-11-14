"""Build a private returner mask from the aligned B1/B2 reapplication join."""
import argparse
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--counts",required=True,type=Path)
p.add_argument("--out",required=True,type=Path)
a = p.parse_args()
with np.load(a.counts,allow_pickle=False) as data:
    mask = data["returned_same_grade"]
    assert mask.ndim == 1 and mask.dtype == np.bool_
    mask.astype("uint8").tofile(a.out)
