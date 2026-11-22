"""Verify pinned public aggregates and rebuild the Chile presentation tables.

Uses only the Python standard library. These checks reconcile archived aggregates.
Regenerating allocations from ministry records requires the upstream replication code.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YEARS = (2022, 2023, 2024, 2025)
DESIGN_YEARS = YEARS[:3]
GROUPS = ('enrolled_at_initial_offer', 'enrolled_elsewhere', 'no_initial_offer')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(name):
    with (ROOT / 'sources' / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    return int(row[key])


def add(rows, field):
    return sum(number(row, field) for row in rows)


def write(name, rows):
    with (ROOT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def authenticate():
    manifest = json.loads((ROOT / 'provenance.json').read_text())
    for relative, entry in manifest['files'].items():
        require(entry['upstream_commit'] == manifest['commit'], 'Mixed upstream commits')
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        require(actual == entry['sha256'], f'Source hash differs: {relative}')
    return manifest['commit']


def comparison_row(year, comparison, group, rows, baseline):
    result = dict(year=year, comparison=comparison, group=group,
                  linked_applicant_years=add(rows, 'linked'),
                  baseline_hits=add(rows, baseline + '_hits'), rank_hits=add(rows, 'rank_hits'),
                  baseline_placed=add(rows, baseline + '_placed'), rank_placed=add(rows, 'rank_placed'),
                  gross_exits=add(rows, 'exits'), gross_entries=add(rows, 'entrants'))
    result['hit_change'] = result['rank_hits'] - result['baseline_hits']
    result['placement_change'] = result['rank_placed'] - result['baseline_placed']
    require(result['placement_change'] == result['gross_entries'] - result['gross_exits'],
            f'Placement accounting fails: {year} {comparison} {group}')
    count = result['linked_applicant_years']
    result['hit_change_pp'] = f"{100 * result['hit_change'] / count:.9f}"
    result['placement_change_pp'] = f"{100 * result['placement_change'] / count:.9f}"
    return result


def comparisons():
    cohorts, groups, fidelity, spillovers = [], [], [], []
    annual = {}
    for year in YEARS:
        official = {r['group']: r for r in read(f'all-{year}.csv')}
        full = official['all']
        for row in official.values():
            require(number(row, 'applicants') == number(row, 'same_course_or_outside'),
                    f'Historical offer mismatch: {year}')
            require(number(row, 'official_only') + number(row, 'replay_only') +
                    number(row, 'both_placed_different_course') == 0, 'Historical error count')
        fidelity.append(dict(year=year, applicant_years=number(full, 'applicants'),
                             exact_course_or_no_offer=number(full, 'same_course_or_outside'),
                             historical_no_offer=number(official['no_official_offer'], 'applicants')))
        transitions = {r['stratum']:r for r in read(f'transitions-{year}.csv')}
        all_people, same = transitions['all'], transitions['identical_menu']
        require(number(all_people, 'n') == number(full, 'applicants'), 'Different cohort totals')
        spillovers.append(dict(year=year, applicant_years=number(all_people, 'n'),
                               gross_exits=number(all_people, 'gross_exits'),
                               gross_entries=number(all_people, 'gross_entries'),
                               net_placement_change=number(all_people, 'rank_placed')-number(all_people, 'blind_placed'),
                               unchanged_menu_applicant_years=number(same, 'n'),
                               unchanged_menu_exits=number(same, 'gross_exits'),
                               unchanged_menu_entries=number(same, 'gross_entries')))
        for prefix, baseline, comparison in [('linked', 'blind', 'rank_vs_blind'),
                                              ('first', 'first', 'rank_vs_first_five')]:
            table = {r['group']:r for r in read(f'{prefix}-{year}.csv')}
            for field in ('linked', baseline+'_hits', 'rank_hits', baseline+'_placed',
                          'rank_placed', 'exits', 'entrants'):
                require(number(table['linked'], field) == add([table[g] for g in GROUPS], field),
                        f'Partition does not sum: {year} {comparison} {field}')
            annual[year,comparison] = table
            row = comparison_row(year, comparison, 'all_linked', [table['linked']], baseline)
            cohorts.append({k:v for k,v in row.items() if k != 'group'})
            for group in (*GROUPS, 'initial_offer', 'all_linked'):
                parts = [table[group]] if group in GROUPS else [table[g] for g in GROUPS[:2]]
                if group == 'all_linked':
                    parts = [table['linked']]
                groups.append(comparison_row(year, comparison, group, parts, baseline))
            if baseline == 'blind':
                mapping = [('linked','all_linked'),('no_initial_offer','linked_no_offer'),
                           ('enrolled_at_initial_offer','enrolled_at_initial_offer'),
                           ('enrolled_elsewhere','enrolled_elsewhere')]
                for group, stratum in mapping:
                    for left,right in [('linked','n'),('exits','gross_exits'),('entrants','gross_entries'),
                                       ('blind_hits','blind_april_shown'),('rank_hits','rank_april_shown'),
                                       ('blind_placed','blind_placed'),('rank_placed','rank_placed')]:
                        require(table[group][left] == transitions[stratum][right],
                                f'Enrollment and transition tables disagree: {year} {group} {left}')
    for comparison,baseline in [('rank_vs_blind','blind'),('rank_vs_first_five','first')]:
        row = comparison_row('total', comparison, 'all_linked',
                             [annual[y,comparison]['linked'] for y in YEARS], baseline)
        cohorts.append({k:v for k,v in row.items() if k != 'group'})
        for group in (*GROUPS, 'initial_offer', 'all_linked'):
            names = [group] if group in GROUPS else list(GROUPS[:2])
            if group == 'all_linked':
                names = ['linked']
            groups.append(comparison_row('total', comparison, group,
                                         [annual[y,comparison][g] for y in YEARS for g in names], baseline))
    for output in (fidelity, spillovers):
        output.append({'year':'total', **{k:add(output,k) for k in output[0] if k!='year'}})
    write('cohort_comparison.csv', cohorts)
    write('group_comparison.csv', groups)
    write('historical_fidelity.csv', fidelity)
    write('all_population_spillovers.csv', spillovers)
    return cohorts, groups


def retention_bounds():
    ceilings = {(r['model'],r['criterion'],r['support'],r['target']):r for r in read('ceiling-summary.csv')}
    exact = ceilings['paired','design','1024','april_school']
    unrestricted = ceilings['paired','design','0','april_school']
    selection = {int(r['year']):r for r in read('retention-selection.csv')}
    scores = {(int(r['year']),r['method']):r for r in read('retention-mask-scores.csv')}
    all_draws = read('exact-design-checks.csv')
    result = []
    for year in DESIGN_YEARS:
        selected = selection[year]
        require(selected['selected'] == 'addback', 'Draw file must match selected method')
        score = scores[year,selected['selected']]
        require(number(score,'gains') == number(selected,'retention_lower'), 'Coverage audit differs')
        require(number(score,'losses') == 0, 'Constructed menu lost an old coverage hit')
        require(number(score,'final_hits')-number(score,'baseline_hits') == number(score,'gains'),
                'Coverage arithmetic differs')
        draw_rows = read(f'retention-draws-{year}.csv')
        require(len(draw_rows)==1024 and {number(r,'draw') for r in draw_rows}==set(range(1,1025)),
                'Missing or duplicate hindsight design draw')
        require(all(number(r,'year')==year and number(r,'batch')==513 for r in draw_rows),
                'Wrong hindsight lottery panel')
        require(add(draw_rows,'exits')==0, 'Retained-seat certificate has an exit')
        require(add(draw_rows,'entrants')==number(selected,'entrants'), 'Entrants differ')
        require(add(draw_rows,'switches')==number(selected,'course_switches'), 'Course switches differ')
        contract_checks = [r for r in all_draws if r['year']==str(year) and r['specification']=='exact'
                           and r['split']=='design_check' and r['certificate']=='stable']
        require(add(contract_checks,'count')==1024 and add(contract_checks,'changed_detailed')==0,
                'Full-contract certificate differs')
        upper, = read(f'retention-upper-{year}.csv')
        values = [number(exact,str(year)),number(selected,'retention_lower'),
                  number(upper,'joint_upper'),number(unrestricted,str(year))]
        require(values==sorted(values), 'Inconsistent retention interval')
        result.append(dict(year=year,design_national_draws=1024,exact_contract_gain=values[0],
                           attained_retention_gain=values[1],retention_upper=values[2],
                           unconstrained_optimum_gain=values[3],incumbent_exit_events=0,
                           entrant_events=add(draw_rows,'entrants'),
                           retained_course_switch_events=add(draw_rows,'switches')))
    result.append({'year':'total', **{k:add(result,k) for k in result[0] if k!='year'}})
    write('retention_bounds.csv',result)
    return result


def prospective():
    historical = {(r['rule'],int(r['year']),r['group']):r for r in read('coverage-groups.csv')}
    learned = {(int(r['year']),r['group']):r for r in read('learned-coverage.csv')}
    design = read('learned-design-audit.csv')
    require(len(design)==3072, 'Learned design draw count')
    require({(number(r,'year'),number(r,'draw')) for r in design} ==
            {(y,d) for y in DESIGN_YEARS for d in range(1,1025)}, 'Learned design panel')
    require(all(r['batch']=='513' and r['exits']=='0' for r in design), 'Learned design failures')
    for year in DESIGN_YEARS:
        a = historical['learned',year,'all']
        b = learned[year,'all']
        require(a['linked']==b['linked'] and a['gains']==b['new_hits'] and a['losses']==b['lost_hits'],
                'Learned historical coverage tables differ')
    output = []
    draw_output = []
    for split, draws, batch in [('course',128,8615729),('family',32,8615731)]:
        panel = {}
        for rule,arm in [('learned','learned_retention'),('structural','structural_all_orders')]:
            rows = [r for r in read(f'{rule}-holdout-{split}.csv') if r['policy']==arm]
            panel[rule] = {(number(r,'year'),number(r,'draw')):r for r in rows}
            expected = {(y,d) for y in DESIGN_YEARS for d in range(1,draws+1)}
            require(set(panel[rule])==expected and len(rows)==len(expected), 'Held-out panel differs')
            require(all(number(r,'batch')==batch for r in rows), 'Held-out batch differs')
            for r in rows:
                draw_output.append(dict(year=number(r,'year'),rule=rule,lottery=split,
                                        batch=number(r,'batch'),draw=number(r,'draw'),
                                        baseline_placed=number(r,'baseline_placed'),
                                        policy_placed=number(r,'placed'),
                                        incumbent_exits=number(r,'exits'),entrants=number(r,'entries'),
                                        retained_school_switches=number(r,'school_switch')))
                require(number(r,'placed') == number(r,'baseline_placed') - number(r,'exits') +
                        number(r,'entries'), 'Held-out placement accounting')
            summaries=[]
            for year in DESIGN_YEARS:
                year_rows=[r for r in rows if number(r,'year')==year]
                coverage=historical[rule,year,'all']
                no_offer=historical[rule,year,'initially_unplaced']
                require(number(coverage,'gains')-number(coverage,'losses')==number(coverage,'net'),
                        'Coverage accounting')
                summaries.append(dict(year=year,rule=rule,lottery=split,
                                      linked_applicant_years=number(coverage,'linked'),
                                      new_historical_hits=number(coverage,'gains'),
                                      lost_historical_hits=number(coverage,'losses'),
                                      net_historical_hit_change=number(coverage,'net'),
                                      no_initial_offer_hit_change=number(no_offer,'net'),
                                      simulated_national_draws=draws,
                                      draws_with_exit=sum(number(r,'exits')>0 for r in year_rows),
                                      incumbent_exit_events=add(year_rows,'exits'),
                                      entrant_events=add(year_rows,'entries'),
                                      retained_school_switch_events=add(year_rows,'school_switch')))
            total={'year':'total','rule':rule,'lottery':split}
            total.update({k:add(summaries,k) for k in summaries[0] if k not in total})
            output.extend([*summaries,total])
        for key,row in panel['learned'].items():
            other=panel['structural'][key]
            require(all(row[k]==other[k] for k in ['year','draw','batch','architecture','baseline_placed']),
                    'The rules used different comparison lotteries')
    write('prospective.csv',output)
    write('prospective_draws.csv',draw_output)
    return output


def field_cohort():
    original={r['group']:r for r in read('field-entry-2021.csv')}
    mixed={r['group']:r for r in read('field-mixed-endpoints.csv')}
    a,b=original['entry'],mixed['entry_linked']
    require(a['linked']==b['n'] and a['blind_linked_placed']==b['blind_placed'] and
            a['rank_linked_placed']==b['rank_placed'], '2021 entry cohort denominators differ')
    require(number(a,'linked_gains')-number(a,'linked_losses') ==
            number(b,'rank_placed')-number(b,'blind_placed'), '2021 gain/loss accounting')
    result=dict(year=2021,entry_grade_applicant_years=number(a,'applicants'),
                linked_applicant_years=number(a,'linked'),
                changed_menu_applicant_years=number(b,'changed'),
                historical_hit_change=number(a,'rank_april_hits')-number(a,'blind_april_hits'),
                simulated_placement_change=number(b,'rank_placed')-number(b,'blind_placed'),
                simulated_unchanged_menu_placement_change=number(b,'rank_unchanged_placed')-
                number(b,'blind_unchanged_placed'))
    write('field_cohort.csv',[result])


def main():
    commit=authenticate()
    cohorts,groups=comparisons()
    bounds=retention_bounds()
    policies=prospective()
    field_cohort()
    mainrow=next(r for r in cohorts if r['year']=='total' and r['comparison']=='rank_vs_blind')
    print(f"Authenticated {len(json.loads((ROOT/'provenance.json').read_text())['files'])} source files at {commit}")
    print('Linked applicant-years:',mainrow['linked_applicant_years'],
          'additional historical hits:',mainrow['hit_change'],
          'simulated placement change:',mainrow['placement_change'])
    print('Retention interval:',bounds[-1]['attained_retention_gain'],bounds[-1]['retention_upper'])
    for r in policies:
        if r['year']=='total':
            print(r['rule'],r['lottery'],'historical gain',r['net_historical_hit_change'],
                  'exit events',r['incumbent_exit_events'],'national draws',r['simulated_national_draws'])


if __name__=='__main__':
    main()
