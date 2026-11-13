"""Download versioned public source files into a user-selected directory."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

REPO="https://raw.githubusercontent.com/cc6580/Analysis-of-Contributing-Factors-for-the-Success-of-Speed-Dating/9096ea5b375011680cf23e5d2ab69dcd2b79467f/"
SOURCES={
    "data.tab":("https://dataverse.harvard.edu/api/access/datafile/2509001","8616bd64ade62c07d8668229c926761ed7a2f59a3f97b8064bd69efc3e07f81c"),
    "profiles.csv":(REPO+"Speed%20Dating%20Data.csv","2e518c3315c4ae360d4c4b143b64b49dc3efc21b93d51007703ec59f97524309"),
    "key.doc":(REPO+"Speed%20Dating%20Data%20Key.doc","1cdd10222becc9685e91c58a806bc1077174f9aaa333a4039a0170fe69cd96cf"),
}
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("directory",type=Path)
    dest=parser.parse_args().directory;dest.mkdir(exist_ok=True,parents=True)
    for name,(url,checksum) in SOURCES.items():
        with urlopen(url,timeout=90) as r:data=r.read()
        assert hashlib.sha256(data).hexdigest()==checksum,name
        (dest/name).write_bytes(data)
