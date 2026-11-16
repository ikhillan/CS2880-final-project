"""Evaluate previously frozen menus on observed enrolment destination, using local-only joins."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

def main(args):
    sys.path.insert(0,str(args.previous.resolve()))
    sys.path.insert(0,str(args.frontier.resolve()))
    from run import Market,history,features,ordering
    from solve import model_menu
    m=Market(args.inputs,2026,positive_scores=True)
    appl=pd.read_csv(next((args.inputs/'2026').glob('C_POST*.csv')),sep=';',usecols=['MRUN','TIPO_PREF'])
    people=pd.Index(np.sort(appl.loc[appl.TIPO_PREF.eq('REGULAR'),'MRUN'].unique()))
    assert len(people)==m.n
    raw=pd.read_csv(next(args.enrollment.glob('D_MATRICULA*.csv')),sep=';',usecols=['MRUN','CODIGO_CARRERA','VIA_INGRESO'])
    assert raw.MRUN.notna().all() and raw.CODIGO_CARRERA.notna().all()
    # Release has exactly one destination per person; fail explicitly if a future release differs.
    assert raw.MRUN.is_unique
    ix=people.get_indexer(raw.MRUN)
    use=ix>=0; registered=np.zeros(m.n,bool);student=ix[use];registered[student]=True
    programs=pd.Index(m.program).get_indexer(raw.CODIGO_CARRERA[use])
    regular=np.zeros(m.n,bool);regular[student]=raw.VIA_INGRESO[use].to_numpy()==1
    edgecodes=m.a.astype(np.int64)*m.m+m.school
    e=pd.Index(edgecodes).get_indexer(student.astype(np.int64)*m.m+programs)
    e[programs<0]=-1
    destination=np.full(m.n,-1,int);destination[student]=e
    has=destination>=0;selected=m.recorded>=0
    same=registered&selected&(destination==m.recorded)
    groups={'all_registrants':registered, 'on_scored_list':has,'off_scored_list':registered&~has,
            'initial_offer_same_program':same,'initial_offer_other_enrollment':registered&selected&~same,
            'no_initial_regular_offer':registered&~selected,'regular_route':registered&regular,
            'other_route':registered&~regular}
    X,x,_=features(m,history(Market(args.inputs,2025,positive_scores=True)))
    meta=json.loads((args.previous/'metrics.json').read_text())['fitted'];w=np.array(meta['weights_intercept_last'])
    score=(X-np.array(meta['feature_mean']))@(w[:-1]/np.array(meta['feature_sd']))+w[-1]
    masks={'own_rank':m.pos<5,'historical_cutoff':ordering(m,(x>=0)*40.-m.r)<5,'edge_classifier':ordering(m,score)<5}
    for name in ('blind','informed'):
        _,masks[name],_=model_menu(m,args.features,args.fit,name)
    rows=[];coverage={}
    for name in ['uniform']+list(masks):
        h=np.zeros(m.n,float)
        if name=='uniform':h[has]=np.minimum(1.,5/m.degree[has])
        else:h[has]=masks[name][destination[has]]
        coverage[name]=h
        for group,subset in groups.items():
            rows.append(dict(policy=name,group=group,n=int(subset.sum()),covered=float(h[subset].sum())))
    comparison=(coverage['informed']-coverage['blind'])
    movement={group:{'gained':int((subset&(comparison>0)).sum()),'lost':int((subset&(comparison<0)).sum())} for group,subset in groups.items()}
    out=args.out;out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(rows).to_csv(out/'enrollment.csv',index=False)
    flow={'registered_rows':len(raw),'registered_distinct':raw.MRUN.nunique(),'outside_regular_cohort':int((~use).sum()),
          'regular_applicants':m.n,'matched_join':int(use.sum()),'no_registration_in_release':int((~registered).sum()),
          'eligible_destinations':int(has.sum()),'unknown_program_among_joined':int((programs<0).sum()),
          'cohort_correctness_check':int(same.sum()),'informed_vs_blind':movement}
    assert sum(groups[g].sum() for g in ('initial_offer_same_program','initial_offer_other_enrollment','no_initial_regular_offer'))==registered.sum()
    (out/'join-check.json').write_text(json.dumps(flow,indent=2)+'\n')
    print(json.dumps(flow),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('previous','frontier','inputs','enrollment','features','fit','out'):p.add_argument(name,type=Path)
    main(p.parse_args())
