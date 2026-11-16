"""Data, market, exposure, metric and recommender checks. Real-data tests skip when the data files are absent."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments.config import expand_labeled
from hiddenoptions import audit, llm
from hiddenoptions.data import (CPI_CSV, FIRSTMOVER_CLEAN, MONTHS, PUMS_HOUSEHOLDS, bedroom_need, cpi_factor, cpi_value, donor_split,
                                load_renters, lower_income_cutoff, weighted_quantile)
from hiddenoptions.exposure import (LlmRecommendations, expensive_first_exposure, show_count, transparent_exposure, uniform_exposure)
from hiddenoptions.markets import Catalog, InsufficientSupply, MarketSpec, build_market, listing_quality
from hiddenoptions.metrics import group_switch, outcomes, price_percentile, rank_score

needs_data = pytest.mark.skipif(not (PUMS_HOUSEHOLDS.exists() and (FIRSTMOVER_CLEAN / "firstmover_listings_2026-08.csv.gz").exists()),
                                reason="data not present")


@pytest.fixture(scope="module")
def catalog():
    return Catalog()


def test_bedroom_rules():
    assert bedroom_need([1, 2, 3, 4, 5, 6, 40]).tolist() == [0, 0, 1, 1, 2, 2, 8]
    assert bedroom_need([1, 2, 3, 4, 5, 6], "strict").tolist() == [0, 0, 2, 2, 3, 3]


def test_show_count():
    assert show_count(np.array([0, 1, 3, 10]), 5, None).tolist() == [0, 1, 3, 5]
    assert show_count(np.array([0, 1, 3, 10]), None, 0.4).tolist() == [0, 1, 2, 4]
    with pytest.raises(ValueError):
        show_count(np.array([1]), 5, 0.4)


def test_cpi_rules():
    t = pd.read_csv(CPI_CSV, dtype={"period": str})
    sep, nov = (float(t[(t.year == 2025) & (t.period == p)].value.iloc[0]) for p in ("M09", "M11"))
    assert cpi_value("2025-10") == ((sep + nov) / 2, "interpolated")  # lost to the 2025 funding lapse
    assert cpi_value("2026-07")[1] == "observed"
    last = t[t.value.notna() & t.period.str.match(r"M(0[1-9]|1[0-2])$")].iloc[-1]
    later = f"{int(last.year) + 1}-01"
    assert cpi_value(later) == (float(last.value), "carried_forward")


def test_donor_split_is_fixed_and_70_30():
    ids = pd.Series([f"2022HU{i:07d}" for i in range(20000)])
    a, b = donor_split(ids), donor_split(ids)
    assert a.equals(b)
    assert abs((a == "calibration").mean() - 0.70) < 0.02


def test_months_cover_the_published_range():
    assert MONTHS[0] == "2025-02" and MONTHS[-1] == "2026-10" and len(MONTHS) == 21


@needs_data
def test_renter_frame_matches_the_reports():
    r = load_renters()
    assert len(r) == 71_908 and int(r.recent_mover.sum()) == 10_302
    assert r.weight.sum() == 2_184_936
    assert round(weighted_quantile(r.income_2024.to_numpy(), r.weight.to_numpy(), 0.5) * cpi_factor("2026-08")) == 68_808
    assert 68_000 < lower_income_cutoff(r) < 69_500  # July 2026 dollars


@needs_data
@pytest.mark.parametrize("design,borough", [("uniform", "Brooklyn"), ("enriched", "Brooklyn"), ("enriched", "Bronx")])
def test_market_invariants(catalog, design, borough):
    spec = MarketSpec(borough, "2026-08", 3, 24, 24, design=design)
    m = build_market(spec, catalog)
    E, r, l = m.inst.E, m.renters, m.listings
    for b in range(24):
        for s in range(24):
            ok = l.rent[s] <= r.cap[b] and l.bedrooms[s] >= r.need_bedrooms[b] and r.income[b] > 0
            assert E[b, s] == ok
    assert (l.borough == borough).all() and (r.borough == borough).all() and r.recent_mover.all()
    assert (r.donors == "evaluation").all()
    for b in range(24):  # strict rankings over exactly E
        assert sorted(m.inst.rank_b[b, E[b]].tolist()) == list(range(E[b].sum()))
    for s in range(24):
        assert sorted(m.inst.rank_s[E[:, s], s].tolist()) == list(range(E[:, s].sum()))
    # One private firm per renter; listing firms are the brokerages; the two sides never share a firm.
    assert m.n_firms_renter == 24 and len(set(m.inst.firm_b)) == 24
    assert set(m.inst.firm_b) & set(m.inst.firm_s) == set()
    if design == "enriched":
        low = r.lower_income.to_numpy()
        assert low.sum() == 12 and (E[low].sum(1) > 0).all()


@needs_data
@pytest.mark.parametrize("borough,n", [("Brooklyn", 24), ("Brooklyn", 64), ("Bronx", 24), ("Bronx", 64)])
def test_enriched_affordability_band(catalog, borough, n):
    spec = MarketSpec(borough, "2026-08", 7, n, n, design="enriched", donors="calibration")
    m = build_market(spec, catalog)
    log, r, l = m.sampling_log, m.renters, m.listings
    frame, pool = catalog.frame(spec)
    g_plus = frame[frame.group == "G+"]
    cap = weighted_quantile(g_plus.cap.to_numpy(), g_plus.weight.to_numpy(), 0.75)
    assert log["band_cap"] == pytest.approx(cap) and log["band_quantile"] == 0.75
    assert log["band_listings"] == int((pool.rent <= cap).sum()) >= n // 2
    assert log["band_listings"] + log["above_band_listings"] == len(pool)
    assert 0 < log["lower_with_edge_weight_share"] <= 1 and log["lower_with_edge_donors"] > 0
    in_band = (l.rent <= cap).to_numpy()
    assert in_band[: n // 2].all() and not in_band[n // 2:].any()  # half from the band, half from above it
    assert l.listing_uid.is_unique
    low = r.lower_income.to_numpy()
    assert low.sum() == n // 2 and (m.inst.E[low].sum(1) >= 1).all()
    assert m.renters.n_feasible_market[low].min() >= 1


@needs_data
def test_enriched_quartile_rule_still_works(catalog):
    spec = MarketSpec("Bronx", "2026-08", 3, 24, 24, design="enriched_quartile", donors="calibration")
    m = build_market(spec, catalog)
    _, pool = catalog.frame(spec)
    q25 = float(pool.rent.quantile(0.25))
    assert m.sampling_log["band_cap"] == pytest.approx(q25) and "band_quantile" not in m.sampling_log
    assert (m.listings.rent[:12] <= q25).all() and (m.listings.rent[12:] > q25).all()
    low = m.renters.lower_income.to_numpy()
    assert low.sum() == 12 and (m.inst.E[low].sum(1) > 0).all()
    new = build_market(MarketSpec("Bronx", "2026-08", 3, 24, 24, design="enriched", donors="calibration"), catalog)
    assert new.spec.market_id != m.spec.market_id


@needs_data
def test_enriched_raises_when_band_too_small(catalog):
    with pytest.raises(InsufficientSupply, match="affordability band"):
        build_market(MarketSpec("Staten Island", "2026-08", 0, 24, 24, design="enriched", donors="calibration"), catalog)


def test_band_quantile_leaves_other_market_ids_unchanged():
    a = MarketSpec("Queens", "2026-08", 1)
    assert a.market_id == MarketSpec("Queens", "2026-08", 1, band_quantile=0.5).market_id
    e = MarketSpec("Queens", "2026-08", 1, design="enriched")
    assert e.market_id != MarketSpec("Queens", "2026-08", 1, design="enriched", band_quantile=0.5).market_id


@needs_data
def test_staten_island_is_unestimable_at_64(catalog):
    with pytest.raises(InsufficientSupply):
        build_market(MarketSpec("Staten Island", "2026-08", 0, 64, 64), catalog)


@needs_data
def test_seed_reproducible_and_paired_across_preferences(catalog):
    a = build_market(MarketSpec("Queens", "2026-07", 5, 24, 24), catalog)
    b = build_market(MarketSpec("Queens", "2026-07", 5, 24, 24), catalog)
    assert np.array_equal(a.inst.rank_b, b.inst.rank_b) and a.spec.market_id == b.spec.market_id
    c = build_market(MarketSpec("Queens", "2026-07", 5, 24, 24, w_p=4.0, sigma_r=1.25), catalog)
    assert c.renters.serialno.tolist() == a.renters.serialno.tolist() and np.array_equal(c.inst.E, a.inst.E)  # same draws
    assert c.spec.market_id != a.spec.market_id


@needs_data
def test_quality_reference_is_centered_on_development_listings(catalog):
    dev = pd.concat([catalog.listings(m) for m in ("2026-06", "2026-07", "2026-08")])
    q = listing_quality(dev.sample(3000, random_state=0), catalog, "features")
    assert abs(q.mean()) < 0.1 and np.abs(q).max() <= 3.0


@needs_data
def test_exposures_are_subsets_with_equal_counts(catalog):
    m = build_market(MarketSpec("Brooklyn", "2026-08", 1, 64, 64), catalog)
    E = m.inst.E
    for k, q in ((5, None), (None, 0.4)):
        shown = [f(m, k, q) for f in (uniform_exposure, transparent_exposure, expensive_first_exposure)]
        for e0 in shown:
            assert not (e0 & ~E).any() and np.array_equal(e0.sum(1), show_count(E.sum(1), k, q))
    t = transparent_exposure(m, 5)
    for b in np.nonzero(E.sum(1) > 5)[0]:  # transparent shows the top observable scores
        assert m.u_observable[b, t[b]].min() >= np.sort(m.u_observable[b, E[b]])[-5]


@needs_data
def test_exact_matches_enumeration_on_real_markets(catalog):
    checked = 0
    for seed in range(40):
        m = build_market(MarketSpec("Queens", "2026-08", seed, 8, 8), catalog)
        if m.inst.n_firms > 15 or not m.inst.E.any():
            continue
        o = audit.Oracle(m.inst, uniform_exposure(m, 2))
        assert audit.exact(m.inst, o, time_cap=30).cost == audit.exhaustive(m.inst, o).cost
        checked += 1
    assert checked >= 5


@needs_data
def test_brokerage_only_audits_never_use_renter_firms(catalog):
    m = build_market(MarketSpec("Bronx", "2026-08", 2, 24, 24, audit_costs="brokerage_only"), catalog)
    r = audit.exact(m.inst, audit.Oracle(m.inst, uniform_exposure(m, 5)), time_cap=60)
    assert r.certified and not r.x[: m.n_firms_renter].any()


@needs_data
def test_metrics_are_consistent(catalog):
    m = build_market(MarketSpec("Bronx", "2026-08", 2, 24, 24), catalog)
    o = audit.Oracle(m.inst, uniform_exposure(m, 5))
    r = audit.exact(m.inst, o, time_cap=30)
    out = outcomes(m, o, r.x)
    lo, hi = out["lower_income"], out["higher_income"]
    assert lo["n"] + hi["n"] == 24
    assert out["all"]["match_after"] <= 1 - out["all"]["no_feasible_listing"] + 1e-9
    assert out["all"]["match_after"] == pytest.approx(out["all"]["match_full_disclosure"])  # matched-set invariance
    assert out["all"]["severity_mean"] >= 0  # a successful plan never beats M_E
    assert sum(out["all"][k] for k in ("scarcity_no_option_in_cohort", "scarcity_none_in_draw", "scarcity_unmatched_full",
                                       "scarcity_matched_full")) == pytest.approx(1.0)
    mb = np.full(24, -1)
    assert rank_score(m.inst, mb).sum() == 0


@needs_data
def test_price_percentile_reference_values(catalog):
    m = build_market(MarketSpec("Queens", "2026-08", 3, 64, 64), catalog)
    E = m.inst.E
    full = price_percentile(m, E)
    assert np.allclose(full[E.sum(1) > 0], 0.5)  # showing everything averages exactly 0.5
    exp = price_percentile(m, expensive_first_exposure(m, 5))
    multi = E.sum(1) > 5
    assert (exp[multi] > 0.5).all()


@needs_data
def test_group_switch_reproduces_pure_policies(catalog):
    m = build_market(MarketSpec("Queens", "2026-08", 4, 64, 64), catalog)
    a, b = uniform_exposure(m, 5), transparent_exposure(m, 5)
    sw = group_switch(m, a, b)
    for e0, combo in ((a, "AA"), (b, "BB")):
        out = outcomes(m, audit.Oracle(m.inst, e0), np.zeros(m.inst.n_firms, bool))
        assert sw[combo]["lower_count"] == out["lower_income"]["blocker_count"]
        assert sw[combo]["higher_count"] == out["higher_income"]["blocker_count"]


def test_switch_decomposition_identity():
    z = {"AA": 0.1, "BA": 0.25, "AB": 0.05, "BB": 0.3}
    s_l = ((z["BA"] - z["AA"]) + (z["BB"] - z["AB"])) / 2
    s_h = ((z["AB"] - z["AA"]) + (z["BB"] - z["BA"])) / 2
    assert s_l + s_h == pytest.approx(z["BB"] - z["AA"])


@needs_data
def test_llm_round_trip_with_fallback(catalog, tmp_path):
    """No API call: prompts, parsing and the protocol's fill rule, with a refusal, garbage, a bad ID and a duplicate."""
    m = build_market(MarketSpec("Queens", "2026-08", 4, 64, 64), catalog)
    exposure = {"name": "claude_k5", "policy": "llm", "k": 5}
    reqs = [llm.build_request(m, b, exposure, catalog) for b in range(64)]
    E = m.inst.E
    for b, req in enumerate(reqs):  # no call when there is nothing to choose
        assert (req is None) == (E[b].sum() <= 5)
    asked = [r for r in reqs if r]
    assert asked
    for req in asked:
        p = req["prompt"].lower()
        assert "income" not in p and "household" not in p and "race" not in p and "budget" in p
    # The presentation order is shared across model arms (paired comparisons).
    other = llm.build_request(m, asked[0]["renter"], {"name": "qwen_k5", "policy": "llm", "k": 5}, catalog)
    assert other["id_map"] == asked[0]["id_map"]
    rows = []
    for i, req in enumerate(asked):
        ids = list(req["id_map"])
        text = json.dumps({"recommended_ids": ids[:2] + ["L999", ids[0]]})  # 2 valid, 1 unknown, 1 duplicate
        rows.append(llm.parse_response(req, "not json" if i == 1 else text, "refused" if i == 0 else "ok"))
    assert rows[1]["status"] == "unparseable" and rows[2]["invalid_ids"] == 1 and rows[2]["duplicate_ids"] == 1
    path = tmp_path / "recs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    e0, stats = LlmRecommendations(path).exposure(m, "claude_k5", k=5)
    assert not (e0 & ~E).any() and np.array_equal(e0.sum(1), show_count(E.sum(1), 5, None))
    assert stats["full_fallback"] == 2 and stats["partial_fill"] == len(asked) - 2
    assert all(e0[r["renter"], r["listings"]].all() for r in rows[2:])  # the model's valid picks are kept
    params = llm.request_params(asked[0], llm.DEFAULT_MODEL)
    assert "temperature" not in params and params["output_config"]["format"]["type"] == "json_schema"


def test_config_expansion_labels_variants_and_seed_offsets():
    cfg = {"grid": {"boroughs": ["Queens"], "months": ["2026-08"], "sizes": [24], "seeds": 2, "seed_start": 100},
           "variants": [{"label": "primary"}, {"label": "s30", "share": 0.3}, {"label": "tight", "tightness": 0.5},
                        {"label": "city", "boroughs": [None], "seeds": 1}]}
    got = expand_labeled(cfg)
    assert [lab for _, lab in got] == ["primary"] * 2 + ["s30"] * 2 + ["tight"] * 2 + ["city"]
    assert [s.seed for s, _ in got[:2]] == [100, 101]
    assert got[4][0].n_listings == 12 and got[6][0].borough is None


def test_jobs_mirror_still_imports():
    """Elaine's jobs code reuses the housing audit, metrics and exposure helpers."""
    for module in ("hiddenoptions.jobs.markets", "hiddenoptions.jobs.exposure", "experiments_jobs.run"):
        importlib.import_module(module)
