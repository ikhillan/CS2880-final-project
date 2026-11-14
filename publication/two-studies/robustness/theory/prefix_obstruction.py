"""Verify that all covered-but-unplaced paths under a prefix end at protected exceptions."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np


def assess(full, new, shown, old_rank, new_rank, continuity):
    placed = new >= 0
    covered = (full >= 0) & shown
    obstructs = placed & (full >= 0) & ~shown & (new_rank > old_rank)
    assert np.all(new[obstructs] == continuity[obstructs])
    starters = covered & ~placed
    assert starters.sum() <= obstructs.sum()
    return Counter(markets=1, applicants=len(full), reference_placed=int((full >= 0).sum()),
                   full_partner_shown=int(covered.sum()), placed=int(placed.sum()),
                   covered_but_unplaced=int(starters.sum()), hidden_worse_continuity=int(obstructs.sum()))


def school(numeric, repo, years):
    sys.path.insert(0, str(repo/'publication/research/policy/equity'))
    sys.path.insert(0, str(repo/'publication/two-studies/analysis'))
    from transitions import Market
    from displacement_paths import displacements
    for year in years:
        m = Market(numeric/f'{year}.bin')
        control = m.solve(np.ones(m.ne,bool))
        retained = m.ret >= 0
        assert np.all(m.j[m.at[1:][retained]-1] == m.ret[retained])
        first_five = (m.cols < 5) | (m.j == m.ret[m.i])
        pred = m.solve(first_five)
        old = np.where(control >= 0,m.j[control],-1)
        new = np.where(pred >= 0,m.j[pred],-1)
        shown = (control>=0) & first_five[control]
        record = assess(old,new,shown,control,pred,m.ret)
        lens,_ = displacements(old,new,shown,m.pr[control],m.pr[pred],control,pred,
                                np.ones(m.n,bool))
        assert sum(lens.values()) == record['covered_but_unplaced']
        record['traced_paths'] = sum(lens.values())
        yield 'Chile', str(year),record


def rental(repo):
    sys.path.insert(0,str(repo))
    from experiments.config import expand_labeled, load_config
    from experiments.pilot_outcomes import CONFIG
    from hiddenoptions.markets import Catalog, InsufficientSupply, build_market
    from hiddenoptions.matching import deferred_acceptance
    catalog = Catalog()
    for spec, panel in expand_labeled(load_config(CONFIG)):
        try:
            m = build_market(spec,catalog)
        except InsufficientSupply:
            continue
        i = np.arange(m.inst.n_b)
        first = m.inst.E & (m.inst.rank_b < 5)
        full = deferred_acceptance(m.inst,m.inst.E)[0]
        new = deferred_acceptance(m.inst,first)[0]
        shown = (full >= 0) & first[i,full]
        rec = assess(full,new,shown,m.inst.rank_b[i,full],m.inst.rank_b[i,new],np.full(len(i),-1,int))
        assert rec['covered_but_unplaced'] == rec['hidden_worse_continuity'] == 0
        yield 'NYC',panel,rec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--numeric',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(); total = defaultdict(Counter)
    repo = args.repo.resolve()
    for stream in (school(args.numeric,repo,[2022,2023]),rental(repo)):
        for site, panel, row in stream:
            total[site,panel].update(row)
            if site=='Chile':
                print(panel,row,flush=True)
    with args.out.open('w',newline='') as f:
        w=csv.DictWriter(f,['setting','panel','markets','applicants','reference_placed','full_partner_shown','placed','covered_but_unplaced','hidden_worse_continuity','traced_paths'],lineterminator='\n')
        w.writeheader()
        for (site,panel),row in sorted(total.items()):w.writerow(dict(setting=site,panel=panel,**row))
