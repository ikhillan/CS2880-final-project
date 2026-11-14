"""Perturb archived outcome and ordinary-priority fields on private copies only."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_private_copy(original, modified_path, changes):
    # changes are (byte offset, number of 32-bit elements, replacement function).
    modified = bytearray(original.read_bytes())
    initial = bytes(modified)
    counts = []
    regions = []
    for offset, length, replacement in changes:
        if offset < 0 or offset + 4*length > len(modified):
            raise ValueError("Invalid slice in serialized input")
        if any(offset < right and offset+4*length > left for left,right in regions):
            raise ValueError("Overlapping field locations")
        view = np.frombuffer(modified, dtype="<i4", count=length, offset=offset)
        old = view.copy()
        view[:] = replacement(old)
        counts.append(int(np.count_nonzero(old!=view)))
        if counts[-1] < max(1,length//2):
            raise ValueError("Perturbation did not change enough field values")
        regions.append((offset,offset+4*length))
    # Check every byte surrounding each perturbed field, including all attributes,
    # model scores, special-category orders, quotas, and the family graph.
    previous = 0
    for left,right in sorted(regions):
        if initial[previous:left]!=modified[previous:left]:
            raise ValueError("Unexpected change outside perturbed fields")
        previous=right
    if initial[previous:]!=modified[previous:]:
        raise ValueError("Unexpected change after perturbed fields")
    modified_path.write_bytes(modified)
    return counts


def cyclic_partner(ids, courses):
    # -1 is outside and 0..courses-1 are program indices; this cycles all labels.
    return ((ids.astype(np.int64)+2)%(courses+1)-1).astype("<i4")


def reverse_ties(order):
    # Keep dtype and range while reversing numerical preference in the packed vector.
    return (int(order.max())+int(order.min())-order.astype(np.int64)).astype("<i4")


def run(*cmd):
    subprocess.run([str(arg) for arg in cmd],check=True)


def main(args):
    spec=json.loads(args.config.read_text())
    original_hashes=json.loads((args.learned/"input-sha256.json").read_text())
    learned_hashes=json.loads((args.learned/"policy-freeze.json").read_text())
    with args.structural_receipt.open(newline="") as f:
        structural_hashes={int(row["year"]):row["sha256"] for row in csv.DictReader(f) if row["mode"]=="prospective"}
    if len(structural_hashes)!=3:raise ValueError("expected three structural receipts")
    args.private.mkdir(parents=True,exist_ok=True)
    fixed_calibration=args.learned/"calibration-2021.csv"
    if sha(fixed_calibration)!=learned_hashes["calibration-2021.csv"]:raise ValueError("calibration receipt")
    def verify_one(year):
        c={k:Path(v).resolve() for k,v in spec[str(year)].items()}
        original=c["prefix"];market=Path(str(original)+".bin");side=c["side"]
        if sha(market)!=original_hashes[str(year)]["market"] or sha(side)!=original_hashes[str(year)]["side"]:
            raise ValueError(f"unexpected original market or sidecar in {year}")
        with tempfile.TemporaryDirectory(prefix=f"isolation-{year}-",dir=args.private) as tmp:
            root=Path(tmp)
            prefix=root/str(year);new_side=root/f"{year}.side"
            mheader=np.fromfile(market,dtype="<i4",count=3)
            n,m,ne=map(int,mheader)
            ns,ms,es,_=map(int,np.fromfile(side,dtype="<i4",count=4))
            if (n,m,ne)!=(ns,ms,es):raise ValueError("market/side dimensions")
            # h[3], offsets[n+1], program[ne], priority[ne], capacity[m], official[n].
            priority_offset=4*(3+n+1+ne)
            label_offset=priority_offset+4*(ne+m)
            priority_changed,label_changed=write_private_copy(market,Path(str(prefix)+".bin"),[
                (priority_offset,ne,reverse_ties),(label_offset,n,lambda v:cyclic_partner(v,m))])
            # h[4], school[m], grade[m], raw_capacity[5*m], honors[n] and flags[ne] bytes.
            ordinary_offset=4*(4+7*m)+n+ne
            (lottery_changed,)=write_private_copy(side,new_side,[(ordinary_offset,ne,reverse_ties)])
            for arm in ("blind","rank"):
                original_mask=Path(str(original)+"-"+arm+".mask")
                if sha(original_mask)!=original_hashes[str(year)][arm+"_mask"]:raise ValueError("original mask")
                Path(str(prefix)+"-"+arm+".mask").symlink_to(original_mask)
            structural_mask=root/"structural.mask"
            run(args.structural_binary,prefix,new_side,structural_mask,root/"structural.csv",year,"prospective")
            if sha(structural_mask)!=structural_hashes[year]:raise ValueError("structural design changed")
            # Mask selection is checked for the full previously frozen 1,024-draw
            # guard vector. The runtime's dependence on priority/outcome fields
            # during guard construction is also tested on three separated ranges.
            learned_mask=root/"learned.mask"
            learned_args=[args.learned_binary,"choose",prefix,new_side,args.flags/f"{year}.flags",fixed_calibration,learned_mask]
            run(*learned_args)
            if sha(learned_mask)!=learned_hashes[str(year)]:raise ValueError("learned design changed")
            for first,last in ((1,4),(511,514),(1021,1024)):
                before=root/f"{first}.before";after=root/f"{first}.after"
                for pref,sid,out in ((original,side,before),(prefix,new_side,after)):
                    run(args.learned_binary,"screen",pref,sid,c["tiers"],year,first,last,out)
                if sha(before)!=sha(after):raise ValueError("design constraints changed")
            return dict(year=year,pupils=n,edges=ne,official_outcomes_changed=label_changed,
                        market_priority_values_changed=priority_changed,ordinary_lottery_values_changed=lottery_changed,
                        subset_draws_checked=12,screen_shards_identical=3,structural_mask_sha256=sha(structural_mask),
                        learned_mask_sha256=sha(learned_mask))
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(verify_one,(2022,2023,2024)))
    args.out.parent.mkdir(exist_ok=True,parents=True)
    with args.out.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","private","learned","flags","structural_binary","learned_binary","structural_receipt","out"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    main(parser.parse_args())
