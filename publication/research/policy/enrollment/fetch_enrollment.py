"""Fetch the two hashed ministry enrollment archives into private year directories."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args()
    sources=json.loads(Path(__file__).with_name('sources.json').read_text())
    for year,rec in sources.items():
        dst=a.destination/year;dst.mkdir(parents=True,exist_ok=True)
        archive=a.destination/f'matricula-{year}.rar'
        if not archive.exists():subprocess.run(['curl','--fail','--location','--retry','2','--output',str(archive),rec['url']],check=True)
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==rec['archive_sha256']
        subprocess.run(['bsdtar','-xf',str(archive),'-C',str(dst)],check=True)
