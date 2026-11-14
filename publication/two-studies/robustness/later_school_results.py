"""Confirm frozen predictions on later official releases with two DA implementations."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def cohort(year, numeric, load):
    m = load(numeric/f"{year}.bin")
    answers = {}
    for policy in ("full","blind","rank"):
        visible=np.fromfile(numeric/f"{year}-{policy}.mask", np.uint8).astype(bool)
        reference=np.fromfile(numeric/f"{year}-{policy}.partners", "<i4")
        solution=m.solve(visible)
        partners=np.full(m.n, -1, int)
        place=solution>=0
        partners[place]=m.j[solution[place]]
        assert np.array_equal(partners,reference)
        offers = visible & (m.j == m.off[m.i])
        assert len(np.unique(m.i[offers]))==int(offers.sum())
        answers[policy]=(place, visible, solution, partners,int(offers.sum()))
    groups = [("all",np.ones(m.n,bool)),("priority",m.ses==1),("other",m.ses==0)]
    for group,keep in groups:
        for policy in answers:
            placed, visible, solution, partners, _=answers[policy]
            full=answers["full"][2]
            has_full = (full>=0)&keep
            lost = keep & answers["blind"][0] & ~placed
            gained = keep & ~answers["blind"][0] & placed
            edited = np.zeros(m.n,bool)
            edited[m.i[visible!=answers["blind"][1]]] = True
            yield dict(year=year,group=group,policy=policy,n=int(keep.sum()),
                historical_offered=int((keep & (m.off>=0)).sum()),official_shown=int((visible & (m.j==m.off[m.i]) & keep[m.i]).sum()),
                full_partner_shown=int(visible[full[has_full]].sum()),placed=int((keep & placed).sum()),
                agreement=int(((partners==m.off)&keep).sum()),edited=int((keep&edited).sum()),
                gained_vs_blind=int(gained.sum()),lost_vs_blind=int(lost.sum()),
                lost_same_menu=int((lost & ~edited).sum()),gained_same_menu=int((gained & ~edited).sum()))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("publication","numeric","out"):p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str((args.publication/"research/policy/equity").resolve()))
    from transitions import Market
    rows=[row for year in (2024,2025) for row in cohort(year,args.numeric,Market)]
    with args.out.open("w",newline="") as f:
        header=csv.DictWriter(f,fieldnames=rows[0],lineterminator="\n")
        header.writeheader();header.writerows(rows)
    for r in rows:
        if r["group"]=="all": print(r)
