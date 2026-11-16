"""Check saved primal matchings and dual vertex covers without an optimization solver."""
import argparse
import json
from pathlib import Path

import numpy as np

from nested_support import extract


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--evidence',type=Path,required=True)
    args=ap.parse_args()
    calibration=json.loads(args.calibration.read_text())
    results=json.loads((args.evidence/'results.json').read_text())
    witnesses=json.loads((args.evidence/'witnesses.json').read_text())
    data=extract(args.repo,calibration['selected'])
    cases={(x['size'],x['price_floor']):x for x in results['sizes']}
    n=results['lower_income_placement_volume']
    required=results['threshold_required_count']
    tau=results['target_burden']
    assert len(data)==240 and n==217 and required==55
    certs=0
    for witness in witnesses:
        size,floor,scope=(witness[k] for k in ('size','price_floor','scope'))
        report=cases[size,floor]
        if scope!='pooled':
            report=report[scope]
        q25=report['exact_minimum_q25']
        groups=[m.select(size,floor) for m in data]
        # The reported feasible assignment matches each side at most once per market.
        rows=witness['attaining_edges']
        assert len(rows)==n and len({(k,b) for k,b,s in rows})==n and len({(k,s) for k,b,s in rows})==n
        qual=[]
        for k,g in enumerate(groups):
            matched=[(b,s) for a,b,s in rows if a==k]
            if scope!='pooled':
                assert len(matched)==int(g.initial_matched.sum())
            for b,s in matched:
                if scope=='original_recipients':
                    assert g.initial_matched[b]
                which=np.flatnonzero((g.renters==b)&(g.slots==s))
                assert len(which)==1
                qual.append(float(g.burden[which[0]]))
        assert np.sort(qual)[54]==q25 and np.quantile(qual,.25)==q25
        # A covering set bounds every matching since each chosen edge needs a different vertex.
        for name,strict,expected in [('cover_strictly_below_optimum',True,report['capacity_strictly_below']),
                                    ('cover_at_acs_target',False,report['threshold_capacity_or_upper_bound'])]:
            upper=0
            for k,(idx,left,right) in enumerate(witness[name]):
                assert idx==k
                g=groups[idx]
                mask=g.burden<q25 if strict else g.burden<=tau
                if scope=='original_recipients':
                    mask &= g.initial_matched[g.renters]
                assert np.all(np.isin(g.renters[mask],left)|np.isin(g.slots[mask],right))
                bound=len(set(left))+len(set(right))
                upper+=min(bound,n if scope=='pooled' else int(g.initial_matched.sum()))
            assert len(witness[name])==len(groups)
            assert upper==expected
            if strict:
                assert upper<required
            certs+=len(groups)
    print(f'Verified {len(witnesses)} fixed-volume assignments and {certs} market-level vertex cover bounds.')


if __name__=='__main__':
    main()
