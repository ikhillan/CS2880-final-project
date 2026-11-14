"""Aggregate national-cohort comparisons and numerical integration errors. No private inputs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


DIFFS = ['history', 'full', 'placed', 'first', 'named', 'edges']
COUNTS = ['drawn_for_recovery','rows_extended','identical_effective_menus','gains',
          'losses','identical_gains','identical_losses']


def summarize(path):
    data = pd.concat([pd.read_csv(path/f'school-recovery-{year}.csv') for year in range(2022, 2026)])
    keys = ['year','response','recovery','pair','half']
    assert not data.duplicated(keys).any()
    for field in DIFFS:
        data['delta_'+field] = data['rank_'+field] - data['blind_'+field]
    measures = ['n'] + ['delta_'+name for name in DIFFS] + COUNTS + [p+'_'+name for p in ('blind','rank') for name in DIFFS]
    assert (data.delta_placed == data.gains - data.losses).all()
    assert (data.identical_losses <= data.losses).all() and (data.identical_gains <= data.gains).all()
    assert (data.delta_edges == 0).all()
    endings = data[data.pair==-1].set_index(['year','response','recovery'])
    random = data[data.pair>=0]
    assert (random.groupby(['year','response','recovery']).size() == 64).all()
    assert (random.groupby(['year','response','recovery','pair']).size() == 2).all()
    for (y,rule),part in random[random.recovery==.5].groupby(['year','response']):
        a,b=endings.loc[(y,rule,0)], endings.loc[(y,rule,1)]
        mids=part.groupby('pair')[['delta_history','delta_full']].mean()
        assert (mids.delta_history == (a.delta_history+b.delta_history)/2).all()
        assert (mids.delta_full == (a.delta_full+b.delta_full)/2).all()
    pooled=data.groupby(keys[1:],as_index=False)[measures].sum().assign(year=0)
    rows=[]
    for (y,rule,p), sample in pd.concat([data,pooled]).groupby(keys[:3]):
        vals=sample.groupby('pair')[measures].mean()
        means=vals.mean(); se=vals.sem() if len(vals)>1 else vals.iloc[0]*0
        rows.append(dict(year=y,response=rule,recovery=p,independent_pairs=len(vals) if 0<p<1 else 0,
                         **means.to_dict(),**{k+'_mcse':se[k] for k in measures if k!='n'}))
    summary=pd.DataFrame(rows)
    for (y,rule),part in summary.groupby(['year','response']):
        base=part[part.recovery==0].iloc[0]
        restored=part[part.recovery==1].iloc[0]
        for measure in ('delta_history','delta_full'):
            exact=base[measure]*(1-part.recovery) + restored[measure]*part.recovery
            summary.loc[part.index,measure+'_expectation']=exact
            assert np.allclose(part.loc[part.recovery==.5,measure],exact[part.recovery==.5],atol=0,rtol=0)
    summary.to_csv(path/'school-recovery-summary.csv',index=False)
    arms=pd.concat(pd.read_csv(path/f'school-tail-{year}-arms.csv',dtype={'assignment_hash':str}) for year in range(2022,2026))
    tail=pd.concat(pd.read_csv(path/f'school-tail-{year}-comparison.csv') for year in range(2022,2026))
    for y,block in tail.groupby('year'):
        arms_year=arms[arms.year==y].set_index(['response','policy'])
        t=block.set_index('response'); data_year=endings.loc[y]
        assert (t.loc['binding',DIFFS[:4]].to_numpy() == data_year.loc[('full',0),['delta_'+v for v in DIFFS[:4]]].to_numpy()).all()
        assert (t.loc['in_order',DIFFS[:4]].to_numpy() == data_year.loc[('one',1),['delta_'+v for v in DIFFS[:4]]].to_numpy()).all()
        for rule in ['all_original','all_last']:
            assert t.loc[rule,'history']==t.loc[rule,'full']==0 and t.loc[rule,'same_set']==t.loc[rule,'n']
        assert t.loc['all_original','same_order']==t.loc['all_original','n']
        for rule in ['binding','in_order','before_right','last','all_last','all_original']:
            ra,rb=arms_year.loc[(rule,'blind')],arms_year.loc[(rule,'rank')]
            assert (rb[DIFFS[:4]]-ra[DIFFS[:4]]==t.loc[rule,DIFFS[:4]]).all()
            for pol,v in [('blind',ra),('rank',rb)]:
                assert v['placed_gain']-v['placed_loss']==v.placed-arms_year.loc[('binding',pol),'placed']
                if rule in ['last','all_last']:
                    assert v['less_occupied_programs']==v['entitled_at_recovered']==0
                    assert v['placed_gain']==v['newly_placed_at_recovered']
                if rule in ['in_order','before_right','last']:
                    assert (v[['history','full','edges']]==arms_year.loc[('in_order',pol),['history','full','edges']]).all()
    total=tail.groupby('response',as_index=False).sum().assign(year=0)
    pd.concat([tail,total]).to_csv(path/'school-tail-summary.csv',index=False)
    print(pd.DataFrame(rows).query('year==0 and recovery==0.5')[['response','delta_history','delta_placed','delta_placed_mcse','delta_first','delta_first_mcse','identical_losses']].to_string(index=False))
    print(total[['response','placed','history','full','first','gains','losses','same_order_losses']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    summarize(parser.parse_args().directory)
