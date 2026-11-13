"""Fetch only the six official public archive files and verify their release checksums."""
from pathlib import Path
import argparse, hashlib, json, subprocess
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args()
    sources=json.loads((HERE/'sources.json').read_text()); hashes=json.loads((HERE/'hashes.json').read_text())
    for year,paths in sources['archives'].items():
        yr=a.destination/year;raw=a.destination/'raw';yr.mkdir(parents=True,exist_ok=True);raw.mkdir(parents=True,exist_ok=True)
        for suffix in paths:
            archive=raw/(Path(suffix).name+'.rar')
            if not archive.is_file():subprocess.run(['curl','--fail','--location','--retry','2','--output',str(archive),'https://datosabiertos.mineduc.cl/wp-content/uploads/'+suffix+'.rar'],check=True)
            assert hashlib.sha256(archive.read_bytes()).hexdigest()==hashes[archive.name]
            subprocess.run(['bsdtar','-xf',str(archive),'-C',str(yr)],check=True)
