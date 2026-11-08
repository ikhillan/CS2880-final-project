"""Data, market, exposure and recommender checks. Real-data tests skip when the data files are absent."""
import json

import numpy as np
import pytest

from experiments.config import expand_labeled
from hiddenoptions import audit, llm
from hiddenoptions.data import FIRSTMOVER_CLEAN, PUMS_HOUSEHOLDS, bedroom_need, cpi_factor, load_renters, weighted_quantile
from hiddenoptions.exposure import LlmRecommendations, random_exposure, rent_max_exposure, show_count
from hiddenoptions.markets import Catalog, MarketSpec, build_market
from hiddenoptions.metrics import outcomes, rank_score

needs_data = pytest.mark.skipif(not (PUMS_HOUSEHOLDS.exists() and FIRSTMOVER_CLEAN.exists()), reason="data not present")


@pytest.fixture(scope="module")
def catalog():
    return Catalog()


def test_bedroom_rule():
    assert bedroom_need([1, 2, 3, 4, 5, 6, 40]).tolist() == [0, 0, 1, 1, 2, 2, 8]


def test_show_count():
    assert show_count(np.array([0, 1, 3, 10]), 5, None).tolist() == [0, 1, 3, 5]
    assert show_count(np.array([0, 1, 3, 10]), None, 0.4).tolist() == [0, 1, 2, 4]
    with pytest.raises(ValueError):
        show_count(np.array([1]), 5, 0.4)


@needs_data
def test_renter_frame_matches_the_reports():
    r = load_renters()
    assert len(r) == 71_908 and int(r.recent_mover.sum()) == 10_302
    assert r.weight.sum() == 2_184_936
    assert round(weighted_quantile(r.income_2024.to_numpy(), r.weight.to_numpy(), 0.5) * cpi_factor("2026-08")) == 68_808


@needs_data
def test_slide_example_pair_is_feasible():
    assert 0.4 * 82_800 / 12 == pytest.approx(2760) and 2550 <= 0.4 * 82_800 / 12 and bedroom_need([2])[0] <= 1


@needs_data
@pytest.mark.parametrize("design", ["stratified", "natural"])
def test_market_invariants(catalog, design):
    spec = MarketSpec("Brooklyn", "2026-08", 3, 24, 24, design=design)
    m = build_market(spec, catalog)
    E, r, l = m.inst.E, m.renters, m.listings
    for b in range(24):
        for s in range(24):
            ok = l.rent[s] <= r.cap_month[b] and l.bedrooms[s] >= r.need_bedrooms[b] and r.income_month[b] > 0
            assert E[b, s] == ok
    assert (l.borough == "Brooklyn").all() and (r.borough == "Brooklyn").all() and r.recent_mover.all()
    for b in range(24):  # strict rankings over exactly E
        assert sorted(m.inst.rank_b[b, E[b]].tolist()) == list(range(E[b].sum()))
    for s in range(24):
        assert sorted(m.inst.rank_s[E[:, s], s].tolist()) == list(range(E[:, s].sum()))
    if design == "stratified":
        assert r.lower_income.sum() == 12 and (r.group[r.lower_income] == "G+").all()
    # Firms are side-exclusive, with renter-side firms first.
    assert set(m.inst.firm_b) & set(m.inst.firm_s) == set()
    assert m.inst.firm_b.max() < m.n_firms_renter <= m.inst.firm_s.min()


@needs_data
def test_seed_reproducible_and_paired_across_shares(catalog):
    a, b = build_market(MarketSpec("Queens", "2026-07", 5, 24, 24), catalog), build_market(MarketSpec("Queens", "2026-07", 5, 24, 24), catalog)
    assert np.array_equal(a.inst.rank_b, b.inst.rank_b) and np.array_equal(a.inst.rank_s, b.inst.rank_s) and a.spec.market_id == b.spec.market_id
    assert build_market(MarketSpec("Queens", "2026-07", 6, 24, 24), catalog).spec.market_id != a.spec.market_id


@needs_data
def test_exposures_are_subsets_with_equal_counts(catalog):
    m = build_market(MarketSpec("Brooklyn", "2026-08", 1, 64, 64), catalog)
    E = m.inst.E
    for k, q in ((5, None), (None, 0.4)):
        r, h = random_exposure(m, k, q), rent_max_exposure(m, k, q)
        assert not (r & ~E).any() and not (h & ~E).any()
        assert np.array_equal(r.sum(1), h.sum(1)) and np.array_equal(r.sum(1), show_count(E.sum(1), k, q))
    assert np.array_equal(random_exposure(m, 5), random_exposure(m, 5))


@needs_data
def test_exact_matches_enumeration_on_real_markets(catalog):
    checked = 0
    for seed in range(30):
        m = build_market(MarketSpec("Queens", "2026-08", seed, 24, 24), catalog)
        if m.inst.n_firms > 15:
            continue
        o = audit.Oracle(m.inst, random_exposure(m, 5))
        r, truth = audit.exact(m.inst, o, time_cap=30), audit.exhaustive(m.inst, o)
        assert r.certified and r.cost == truth.cost
        checked += 1
    assert checked >= 5


@needs_data
def test_metrics_are_consistent(catalog):
    m = build_market(MarketSpec("Bronx", "2026-08", 2, 24, 24), catalog)
    o = audit.Oracle(m.inst, random_exposure(m, 5))
    r = audit.exact(m.inst, o, time_cap=30)
    out = outcomes(m, o, r.x)
    assert out["lower_income"]["n"] + out["higher_income"]["n"] == 24
    assert out["all"]["match_after"] <= 1 - out["all"]["no_feasible_listing"] + 1e-9
    # Top choice scores 1 and unmatched scores 0.
    mb = np.full(24, -1)
    assert rank_score(m.inst, mb).sum() == 0
    b = int(np.argmax(m.inst.list_len_b))
    mb[b] = m.inst.prefs_b[b][0]
    assert rank_score(m.inst, mb)[b] == 1


@needs_data
def test_llm_round_trip_with_fake_answers(catalog, tmp_path):
    """No API call: the prompt, parsing and E0 steps, with a bad ID, a repeat, a refusal and garbage text."""
    m = build_market(MarketSpec("Queens", "2026-08", 4, 24, 24), catalog)
    exposure = {"name": "llm_k5", "policy": "llm", "k": 5}
    reqs = [llm.build_request(m, b, exposure, catalog) for b in range(24)]
    rows = []
    for i, req in enumerate([r for r in reqs if r]):
        ids = list(req["id_map"])[: req["count"]]
        assert len(req["custom_id"]) <= 64 and str(req["market_id"]) == m.spec.market_id
        assert "race" not in req["prompt"].lower()
        text = json.dumps({"recommended_ids": ids + ["L999"] + ids[:1]})
        row = llm.parse_response(req, text if i != 1 else "not json", "ok" if i != 0 else "refused")
        if i >= 2:
            assert row["invalid_ids"] == 1 and len(row["listings"]) == req["count"]
        rows.append(row)
    path = tmp_path / "recs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    e0, stats = LlmRecommendations(path).exposure(m, "llm_k5")
    assert not (e0 & ~m.inst.E).any() and stats["refused"] == 1 and stats["missing"] == 0
    answered = [r for r in rows if r["status"] == "ok"]
    assert all(e0[r["renter"]].sum() == r["count"] for r in answered)
    assert audit.Oracle(m.inst, e0).stable(np.ones(m.inst.n_firms, bool))
    params = llm.request_params(reqs[0] or [r for r in reqs if r][0], llm.DEFAULT_MODEL)
    assert "temperature" not in params and params["output_config"]["format"]["type"] == "json_schema"


def test_config_expansion_labels_variants():
    cfg = {"grid": {"boroughs": ["Queens"], "months": ["2026-08"], "sizes": [24], "seeds": 2},
           "variants": [{"label": "primary"}, {"label": "s30", "share": 0.3}, {"label": "tight", "tightness": 0.5}, {"label": "city", "boroughs": [None], "seeds": 1}]}
    got = expand_labeled(cfg)
    assert [lab for _, lab in got] == ["primary"] * 2 + ["s30"] * 2 + ["tight"] * 2 + ["city"]
    assert got[4][0].n_listings == 12 and got[6][0].borough is None
