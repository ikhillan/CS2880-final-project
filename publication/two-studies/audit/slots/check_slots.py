"""Validate the published tables and compare the independent national implementations."""
from pathlib import Path
import csv

HERE = Path(__file__).parent


def records(name):
    with (HERE/name).open() as f:
        return list(csv.DictReader(f))


def main():
    school = records('school-a.csv')+records('school-b.csv')
    rental = records('nyc.csv')
    assert len(school)==8 and len(rental)==4
    for r in school+rental:
        for w in (['full','history'] if r['site']=='Chile' else ['full']):
            missing = sum(int(r[w+'_'+k]) for k in ['outside_pool','blocks','no_slot','slot_oracle_gain'])
            assert missing == int(r[w+'_missed']) == int(r[w+'_reference_edges']) - int(r[w+'_baseline'])
            assert int(r[w+'_whole_oracle_gain']) <= int(r[w+'_slot_oracle_gain'])
        assert r['full_control_placed'] == r['full_full_control_coverage'] == r['full_reference_edges']
    independently = 0
    for year in ['2022','2023','2024','2025']:
        controls = records(f'check-{year}.csv')
        assert {r['pool'] for r in controls} == {'union','all'}
        for control in controls:
            row = next(r for r in school if r['pool']==control['pool'] and r['panel']==control['year'])
            for col,val in control.items():
                if col in ('year','pool'): continue
                assert val == row[col], (year,row['pool'],col,row[col],val)
                independently += 1
    print(f'{len(school)+len(rental)} panel/pool rows pass accounting and {independently} national counts reproduce independently.')


if __name__=='__main__':
    main()
