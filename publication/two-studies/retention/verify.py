"""Reconcile the paper's national seat-retention results from public aggregates."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YEARS = (2022, 2023, 2024)


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def total(rows, field):
    return sum(int(row[field]) for row in rows)


def bounds():
    values = []
    scores = read(ROOT/'bounds/aggregates/mask-scores.csv')
    assert len(scores)==6
    by_mask = {(int(r['year']),r['method']):r for r in scores}
    assert len(by_mask)==6
    for year in YEARS:
        folder = ROOT / 'bounds/aggregates'
        first, = read(folder / f'{year}-oracle.csv')
        second, = read(folder / f'{year}-addback.csv')
        upper, = read(ROOT / 'upper/results' / f'joint-{year}.csv')
        attainable = []
        for mode, result in (('oracle', first), ('addback', second)):
            draws = read(folder / f'{year}-{mode}-draws.csv')
            assert len(draws) == 1024
            assert {int(row['draw']) for row in draws} == set(range(1, 1025))
            assert all((int(row['year']),int(row['batch']),int(row['exits'])) == (year,513,0) for row in draws)
            checked = by_mask[year,mode]
            gain = int(checked['gains'])
            assert int(checked['final_hits'])-int(checked['baseline_hits'])==gain
            assert int(checked['losses'])==0 and len(checked['final_mask_sha256'])==64
            assert gain==int(result['retention_lower'])==int(checked['changed_rows'])
            attainable.append(gain)
        exact = int(first['full_contract_lower'])
        lower = max(attainable)
        outer = int(upper['joint_upper'])
        potential = int(upper['possible_gains'])
        for tag,field in [('cutoff','relaxation_1_upper'),('capacity','relaxation_2_upper')]:
            check, = read(ROOT / 'upper/results' / tag / f'prefix-upper-{year}.csv')
            assert (int(check['year']),int(check['batch']),int(check['draws'])) == (year,513,1024)
            assert int(check['retention_upper']) == int(upper[field])
        assert int(upper['relaxation_2_upper']) == outer
        assert int(first['rowwise_upper']) == potential
        assert exact <= lower <= outer <= potential
        values.append((exact, lower, outer, potential))
    answer = tuple(map(sum, zip(*values)))
    assert answer[0] == 314 and answer[-1] == 4823
    return answer


def policies():
    groups = read(ROOT / 'coverage-groups.csv')
    counts = {(r['rule'],int(r['year']),r['group']):r for r in groups}
    for rule in ('structural','learned'):
        for year in YEARS:
            cohort = counts[rule,year,'all']
            for a,b in [('disadvantaged','other_priority_status'),('initially_placed','initially_unplaced')]:
                for field in ('applicants','linked','gains','losses','net'):
                    assert int(cohort[field]) == int(counts[rule,year,a][field]) + int(counts[rule,year,b][field])
            assert int(cohort['gains']) - int(cohort['losses']) == int(cohort['net'])
            missed = counts[rule,year,'initially_unplaced']
            assert int(missed['gains']) == int(missed['losses']) == 0
        assert total([counts[rule,y,'initially_unplaced'] for y in YEARS],'linked') == 115626
    assert [int(counts['structural',y,'all']['net']) for y in YEARS] == [-2,5,25]
    assert [int(counts['learned',y,'all']['net']) for y in YEARS] == [9,22,19]

    learned_hashes = json.loads((ROOT/'policy/policy-freeze.json').read_text())
    structural_hashes = {int(r['year']):r['sha256'] for r in read(ROOT/'structural/frozen-masks.csv') if r['mode']=='prospective'}
    isolation = read(ROOT/'isolation/isolation.csv')
    assert len(isolation)==3 and total(isolation,'pupils')==1580726
    for row in isolation:
        assert row['learned_mask_sha256'] == learned_hashes[row['year']]
        assert row['structural_mask_sha256'] == structural_hashes[int(row['year'])]
        assert row['pupils']==row['official_outcomes_changed'] and int(row['subset_draws_checked'])==12
    for split, draw_count in [('holdout_course',128),('holdout_family',32)]:
        learned = [r for r in read(ROOT/'policy'/f'{split}.csv') if r['policy']=='learned_retention']
        structural = read(ROOT/'comparison'/f'{split}.csv')
        def paired_counts(rows):
            assert len(rows) == 3 * draw_count
            return {(int(r['year']),int(r['draw'])):(r['batch'],r['architecture'],r['baseline_placed']) for r in rows}
        refs = paired_counts(learned)
        assert set(refs) == {(y,d) for y in YEARS for d in range(1,draw_count+1)}
        assert refs == paired_counts(structural)
        assert total(learned,'exits') == 1 and total(structural,'exits') == 0
        assert all(r['policy']=='structural_all_orders' for r in structural)
        print(split, 'school switches (learned / all-order)', total(learned,'school_switch'),total(structural,'school_switch'))


if __name__ == '__main__':
    policies()
    print('Exact contracts, feasible retention, retention upper, unrestricted optimum:', *bounds())
