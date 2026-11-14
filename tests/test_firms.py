"""Brokerage alias map: normalization, firm labels, and the opt-in MarketSpec.firm_aliases (default off changes nothing)."""
import numpy as np
import pandas as pd
import pytest

from hiddenoptions.data import FIRSTMOVER_CLEAN, PUMS_HOUSEHOLDS
from hiddenoptions.firms import ALIAS_MAP, AliasMap, loose_key, normalize_name
from hiddenoptions.markets import Catalog, MarketSpec, build_market

needs_data = pytest.mark.skipif(not (PUMS_HOUSEHOLDS.exists() and (FIRSTMOVER_CLEAN / "firstmover_listings_2026-08.csv.gz").exists()
                                     and ALIAS_MAP.exists()), reason="data or alias map not present")


@pytest.mark.parametrize("raw,key", [
    ("Skyhigh Realty NYC LLC", "skyhigh"),
    ("FIND Real Estate", "find"), ("The Find NY", "find"),
    ("Voro, LLC", "voro"), ("Voro NYC", "voro"), ("Voro New York", "voro"),
    ("C & C Apartment Management, LLC", "c and c apartment management"), ("C&C Apartment Management LLC", "c and c apartment management"),
    ("Rose Associates, Inc.", "rose associates"), ("SERHANT.", "serhant"), ("Mega Homes Realty INC", "mega homes"),
    ("Guidance Realty Corp.", "guidance"), ("Guidance Realty Group, LLC", "guidance"),
    ("R New York", "r new york"), ("346 Realty Group LLC", "346 realty group"),  # never stripped down to a stub
    ("Brooklyn Group", "brooklyn group"), ("Brooklyn Properties", "brooklyn properties"),  # loose words are not merged
    ("Café Realty L.L.C.", "cafe"), (None, ""), (float("nan"), ""), ("", ""),
])
def test_normalize_name(raw, key):
    assert normalize_name(raw) == key


def test_loose_key_only_suggests():
    assert loose_key("Brooklyn Group") == loose_key("Brooklyn Properties") == "brooklyn"
    assert normalize_name("Brooklyn Group") != normalize_name("Brooklyn Properties")


def small_map(reviewed=False):
    return AliasMap({"reviewed": reviewed, "singleton_kinds": ["unresolved"],
                     "firms": {"voro": {"name": "Voro NYC", "kind": "brokerage"}, "blueground": {"name": "The Blueground", "kind": "platform"},
                               "mystery": {"name": "Mystery", "kind": "unresolved"}},
                     "aliases": {"Voro NYC": "voro", "Voro, LLC": "voro", "The Blueground": "blueground", "Mystery": "mystery"},
                     "keys": {"voro": "voro", "blueground": "blueground", "mystery": "mystery"}})


def test_firm_ids_follow_the_map():
    l = pd.DataFrame({"source_type": ["PARTNER", "PARTNER", "PARTNER", "FEED", "OWNER", "PARTNER", "PARTNER", "PARTNER"],
                      "source_group": ["Voro NYC", "Voro, LLC", "VORO Realty Inc.", "The Blueground", "Owner", "Mystery", None, "New Shop LLC"],
                      "unit_id": [f"fmu:{i}" for i in range(8)],
                      "firm_id": ["firm:voro-nyc", "firm:voro-llc", "firm:voro-realty-inc", "firm:the-blueground", "owner:fmu:4",
                                  "firm:mystery", "firm:unknown", "firm:new-shop-llc"]})
    got = small_map().firm_ids(l).tolist()
    assert got[:3] == ["firm:voro"] * 3  # aliases and an unseen spelling variant (normalized-key lookup) share one firm
    assert got[3] == "firm:blueground"  # platforms stay firms unless listed in singleton_kinds
    assert got[4] == "owner:fmu:4"  # owner listings keep their private firm
    assert got[5:] == ["single:fmu:5", "single:fmu:6", "single:fmu:7"]  # unresolved, missing and unknown names: singletons


def test_firm_aliases_off_keeps_market_ids():
    spec = MarketSpec("Brooklyn", "2026-08", 3, 24, 24)
    assert spec.market_id == "Brooklyn-2026-08-n24x24-s3-17f2bf4d"  # value from before the field existed
    assert MarketSpec("Brooklyn", "2026-08", 3, 24, 24, design="enriched").market_id == "Brooklyn-2026-08-n24x24-s3-68a5e279"
    assert MarketSpec("Brooklyn", "2026-08", 3, 24, 24, firm_aliases=True).market_id != spec.market_id


@needs_data
@pytest.mark.parametrize("design", ["uniform", "enriched"])
def test_firm_aliases_only_relabel_firms(design):
    catalog = Catalog()
    for seed in range(3):
        base = build_market(MarketSpec("Manhattan", "2026-07", seed, 64, 64, design=design), catalog)
        alias = build_market(MarketSpec("Manhattan", "2026-07", seed, 64, 64, design=design, firm_aliases=True), catalog)
        # Same listings, renters, edges and preferences; only listing-side firm labels can change.
        assert base.listings.listing_uid.tolist() == alias.listings.listing_uid.tolist()
        assert base.renters.serialno.tolist() == alias.renters.serialno.tolist()
        assert np.array_equal(base.inst.E, alias.inst.E)
        assert np.array_equal(base.inst.rank_b, alias.inst.rank_b) and np.array_equal(base.inst.rank_s, alias.inst.rank_s)
        assert (alias.listings.firm_id_raw == base.listings.firm_id).all()
        owner = alias.listings.source_type == "OWNER"
        assert (alias.listings.firm_id[owner] == base.listings.firm_id[owner]).all()
        assert alias.listings.firm_id.str.match(r"^(firm|single|owner):").all()
        assert alias.sampling_log["firm_aliases"] is True and "firm_aliases" not in base.sampling_log


def test_alias_map_fingerprint_changes_market_identity(monkeypatch):
    from hiddenoptions import markets

    first = small_map()
    spec = MarketSpec("Brooklyn", "2026-08", 1, firm_aliases=True)
    monkeypatch.setattr(markets, "load_alias_map", lambda: first)
    first_id = spec.market_id
    reordered = AliasMap(dict(reversed(list(first.data.items()))))
    assert first.sha256 == reordered.sha256
    revised = AliasMap({**first.data, "aliases": {**first.aliases, "New Shop LLC": "voro"}})
    monkeypatch.setattr(markets, "load_alias_map", lambda: revised)
    assert spec.market_id != first_id


def test_aliases_preserve_clustered_listings(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from hiddenoptions import markets

    ids = [f"l_{j}" for j in range(24)]
    pool = pd.DataFrame({"listing_uid": ids, "firm_id": [f"raw_{j // 3}" for j in range(24)]})
    frame = pd.DataFrame({"weight": [1, 2]})
    new_ids = pd.Series(["reviewed" for _ in ids], index=ids)
    cat = SimpleNamespace(heldout_dedup=True, dedup_counts={}, frame=lambda spec: (frame, pool),
                          alias_firms=lambda *a, **kw: new_ids)
    monkeypatch.setattr(markets, "load_alias_map", small_map)
    spec = MarketSpec("Brooklyn", "2026-08", 7, n_renters=4, n_listings=12, firm_sampling="clustered")
    base_r, base_l, _ = markets.sample_people(spec, cat)
    alias_r, alias_l, log = markets.sample_people(replace(spec, firm_aliases=True), cat)
    pd.testing.assert_frame_equal(base_r, alias_r)
    assert base_l.listing_uid.tolist() == alias_l.listing_uid.tolist()
    assert base_l.firm_id.tolist() == alias_l.firm_id_raw.tolist()
    assert set(alias_l.firm_id) == {"reviewed"}
    assert log["alias_map_sha256"] == small_map().sha256


def test_review_updates_normalized_fallback_and_validates_review_set():
    from scripts.brokerage_aliases import build_map, propose

    listings = pd.DataFrame({"source_group": ["Voro LLC", "Voro NYC", "Other Realty"],
                             "source_type": ["PARTNER"] * 3, "listing_uid": ["a", "b", "c"]})
    state, sheet = propose(listings)
    sheet["confirmed_kind"] = "brokerage"
    incomplete = sheet.iloc[:1]
    with pytest.raises(ValueError, match="review sheet"):
        build_map(state, incomplete)
    sheet.loc[sheet.source_group.str.startswith("Voro"), "confirmed_canonical"] = "other"
    merged = AliasMap(build_map(state, sheet))
    assert merged.reviewed and merged.canonical("Voro, Inc.") == "other"

    state, sheet = propose(listings)
    sheet["confirmed_kind"] = "brokerage"
    sheet.loc[sheet.source_group == "Voro LLC", "confirmed_canonical"] = "other"
    split = AliasMap(build_map(state, sheet))
    assert split.canonical("Voro LLC") == "other"
    assert split.canonical("Voro NYC") == "voro"
    assert split.canonical("Voro, Inc.") is None


def test_relabeling_pairs_exposure_and_request_randomness_but_keeps_routes_distinct():
    from dataclasses import replace
    from types import SimpleNamespace
    from experiments.open_model_recommend import request_seed
    from hiddenoptions import exposure, llm

    spec = MarketSpec("Brooklyn", "2026-08", 12)
    listings = pd.DataFrame({"listing_uid": [f"L-{j}" for j in range(10)],
                             "rent_asking": range(1000, 1010), "bedrooms": 1, "bathrooms_total": 1.0, "sqft": 450})
    renters = pd.DataFrame([{"puma": "00123", "borough": "Brooklyn", "cap": 1600, "need_bedrooms": 1}])
    data = dict(inst=SimpleNamespace(E=np.ones((1, len(listings)), bool)), listings=listings,
                renters=renters, distance_km=np.zeros((1, len(listings))))
    raw = SimpleNamespace(spec=spec, **data)
    alias = SimpleNamespace(spec=replace(spec, firm_aliases=True), **data)
    catalog = SimpleNamespace(puma_label={"00123": "Central"})
    arm = {"name": "qwen_k5", "k": 5}
    assert raw.spec.market_id != alias.spec.market_id
    assert exposure.fallback_order(raw, 0).tolist() == exposure.fallback_order(alias, 0).tolist()
    assert np.array_equal(exposure.uniform_exposure(raw, k=5), exposure.uniform_exposure(alias, k=5))
    for rep in (0, 1):
        a, b = (llm.build_request(m, 0, arm, catalog, rep) for m in (raw, alias))
        assert a["custom_id"] != b["custom_id"]
        assert a["prompt"] == b["prompt"] and a["id_map"] == b["id_map"]
        assert a["candidates_sha256"] == b["candidates_sha256"]
        assert "sampling_seed" not in a
        assert request_seed(a) == request_seed(b)
