"""Download only the pinned public inputs for this development diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.request import urlopen

parser=argparse.ArgumentParser()
parser.add_argument("--data",type=Path,default=Path(__file__).resolve().parent/"data")
root=parser.parse_args().data
listing=json.loads((root/'firstmover/PROVENANCE.json').read_text())
census=json.loads((root/'acs/summary.json').read_text())
def fetch(path, url, expected):
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected:
        return
    temp=path.with_suffix(path.suffix+'.part')
    with urlopen(url, timeout=120) as response,temp.open('wb') as out:
        shutil.copyfileobj(response,out)
    if hashlib.sha256(temp.read_bytes()).hexdigest()!=expected:
        raise ValueError('Published snapshot differs from pinned bytes: '+str(path))
    os.replace(temp,path)
for month in ('2026-06.csv','2026-07.csv','2026-08.csv'):
    info=listing['files'][month]
    fetch(root/'firstmover'/month,info['source_url'],info['sha256'])
for filename,source,hashkey in [('csv_hny.zip','housing','housing_zip'),('tract_to_puma_2020.txt','geography','crosswalk')]:
    fetch(root/'acs'/filename,census['sources'][source],census['sha256'][hashkey])
