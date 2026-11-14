"""Reconcile the statements added in the EC review using only distributed aggregates."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def ranks():
    c = pd.read_csv(ROOT / 'chile/school-rank-changes.csv')
    a = c[c.group == 'all'].set_index('year')
    t = pd.read_csv(ROOT / 'chile/school-rank-thresholds.csv').query("group == 'all'")
    first = t[t.k == 1].set_index('year').sort_index()
    assert (first.delta == first.submitted_delta).all() and first.delta.sum() == 2218
    assert a.applicants.sum() == 2044649 and a.placed_delta.sum() == -1787
    assert a.loss.sum() == 2527 and a.same_menu_losses.sum() == 1752
    assert a.historical_coverage_delta.sum() == 5552
    for g, limits in [('same_menu',(2254,7215)), ('changed_menu',(8076,2200))]:
        s = c[c.group == g]
        assert (s.better_including_gains.sum(), s.worse_including_losses.sum()) == limits
        assert ((s.better_including_gains > s.worse_including_losses) == (g=='changed_menu')).all()
    rs = pd.read_csv(ROOT / 'ordinal/rental-rank-summary.csv')
    rc = pd.read_csv(ROOT / 'ordinal/rental-rank-counts.csv.gz')
    published = pd.read_csv(ROOT.parent / 'evidence/policy-rates.csv').query(
        "comparison == 'haiku_vs_uniform' and group == 'all' and metric == 'placed'").set_index('design')
    for (label, cutoff), g in rc.groupby(['design', 'cutoff']):
        bycell = g.groupby(['month','borough']).agg({'uniform':'sum','haiku':'sum','applicants':'sum','cell_weight':'first'})
        est = 100*np.average((bycell.haiku-bycell.uniform)/bycell.applicants,weights=bycell.cell_weight)
        row = rs[(rs.design==label)&(rs.cutoff==cutoff)].iloc[0]
        np.testing.assert_allclose(row.delta_pp,est)
        if cutoff==64:
            np.testing.assert_allclose([est,row.lo_pp,row.hi_pp],published.loc[label,['delta_pp','lo_pp','hi_pp']].to_numpy(float))
        if cutoff==1: assert row.lo_pp>0


def protection():
    s = pd.concat([pd.read_csv(ROOT/'slots'/x) for x in ('school-a.csv','school-b.csv')])
    a = s[s.pool=='all']
    totals = a.select_dtypes('number').sum()
    for outcome in ('full','history'):
        np.testing.assert_array_equal(a[outcome+'_missed'],a[outcome+'_blocks']+a[outcome+'_slot_oracle_gain'])
    assert (totals.full_reference_edges,totals.full_missed,totals.full_blocks,totals.full_slot_oracle_gain)==(1885296,9407,9333,74)
    assert (totals.full_candidate_gain,totals.history_candidate_gain)==(5975,5552)
    assert (totals.full_slot_score_gain,totals.history_slot_score_gain)==(10,-30)
    assert (totals.full_control_placement_gain_vs_b,totals.history_full_control_gain_vs_b)==(179,2819)
    assert (a.full_control_placement_gain_vs_b>0).all() and (a.history_full_control_gain_vs_b>0).all()
    n = pd.read_csv(ROOT/'slots/nyc.csv').query("pool=='all'").set_index('panel')
    assert (n.loc['primary','full_slot_oracle_gain'], n.loc['enriched','full_slot_oracle_gain'])==(2014,3018)
    assert (n.full_missed==n.full_blocks+n.full_slot_oracle_gain).all()


def sensitivity():
    c=pd.read_csv(ROOT/'chile/school-capacity-bracket.csv')
    assert len(c)==12 and (c.placed_delta==c.placed_rank-c.placed_blind).all()
    assert (c.full_shown_delta>0).all() and (c.placed_delta<0).all() and (c.hist_delta>0).all()
    b=c[c.scenario=='baseline'].set_index('year'); s=c[c.scenario=='sum_rescored_menus'].set_index('year')
    assert (s.full_exact<b.full_exact).all() and (s.added_seats>0).all()
    n=ROOT/'nyc-temporal'; meta=json.loads((n/'temporal-verification.json').read_text())
    for name, digest in meta['output_sha256'].items(): assert hashlib.sha256((n/name).read_bytes()).hexdigest()==digest
    t=pd.read_csv(n/'temporal-summary.csv')
    assert t.used_markets.sum()==8400 and t.applicants.sum()==537600 and t.supported_cells.sum()==39
    assert (t.delta_coverage_low_pp>0).all() and (t.delta_placement_high_pp<0).all()
    assert (t.target_geographic_month_weight.min()*100).round(2)==79.97


def simulated_orders():
    x=pd.read_csv(ROOT/'chile/school-simulated-menu-batches.csv')
    m=['historical_shown','placed','full_shown','submitted_first']
    base=x[x.policy=='rank'].set_index(['year','batch'])[m]
    blind=x[x.policy=='blind'].set_index(['year','batch'])[m]
    for draws,passes in ((16,5),(32,16),(64,16)):
        sim=x[x.lottery_draws==draws].set_index(['year','batch'])[m]
        d=sim-base
        assert len(sim)==16 and not d.isna().any().any()
        assert ((d.historical_shown>0)&(d.placed>0)).sum()==passes
        if draws==64:
            assert (d[m].min().tolist(),d[m].max().tolist())==([494,25,605,-163],[541,56,680,-106])
            assert (sim.placed<blind.placed).all() and (sim.historical_shown>blind.historical_shown).all()

    late=pd.read_csv(ROOT/'chile/school-simulated-menu-2024-batches.csv')
    base=late[late.policy=='rank'].set_index('batch')[m]
    for draws,passes in ((16,0),(32,3),(64,8)):
        d=late[late.lottery_draws==draws].set_index('batch')[m]-base
        assert len(d)==8 and not d.isna().any().any()
        assert ((d.historical_shown>0)&(d.placed>0)).sum()==passes
        if draws==64:
            assert (d[m].min().tolist(),d[m].max().tolist())==([447,2,540,-123],[483,24,557,-105])


if __name__ == '__main__':
    ranks(); protection(); sensitivity(); simulated_orders()
    print('Assignment rank profiles, stable-witness limits, positive controls, capacity and catalog checks reconcile.')
