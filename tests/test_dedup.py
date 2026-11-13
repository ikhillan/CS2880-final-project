"""Development/held-out physical-unit dedup (protocol section 3): held-out months only, identifiable units only."""
import numpy as np
import pandas as pd
import pytest

from hiddenoptions import data
from hiddenoptions.data import (DEV_MONTHS, FIRSTMOVER_CLEAN, HELDOUT_PRIMARY, HELDOUT_REPLICATION, PUMS_HOUSEHOLDS, dedup_heldout,
                                heldout_period, load_heldout_listings, load_listings)
from hiddenoptions.markets import Catalog, MarketSpec, build_market

needs_data = pytest.mark.skipif(not (PUMS_HOUSEHOLDS.exists() and all((FIRSTMOVER_CLEAN / f"firstmover_listings_{m}.csv.gz").exists()
                                                                      for m in DEV_MONTHS + HELDOUT_PRIMARY)),
                                reason="data not present")


def test_periods():
    assert all(heldout_period(m) == "primary" for m in HELDOUT_PRIMARY)
    assert all(heldout_period(m) == "replication" for m in HELDOUT_REPLICATION)
    assert all(heldout_period(m) is None for m in DEV_MONTHS)


def test_dedup_rules(monkeypatch):
    l = pd.DataFrame({"listing_uid": [f"fm:{i}" for i in range(6)],
                      "unit_id": ["u_dev", "u_new", "u_sep", "u_dev", "u_sep", "u_other"]})
    sources = pd.Series(["url_unit", "url_unit", "url_unit", "url_building+unit_field", "address_only", "url_building_only"],
                        index=l.listing_uid)
    monkeypatch.setattr(data, "unit_key_sources", lambda month: sources)
    out, c = dedup_heldout(l, "2026-10", dev_units={"u_dev"}, earlier_units={"u_sep", "u_dev"})
    # identifiable dev unit and identifiable repeat from September are removed; ambiguous units are kept and flagged
    assert out.listing_uid.tolist() == ["fm:1", "fm:3", "fm:4", "fm:5"]
    assert out.unit_ambiguous.tolist() == [False, True, True, True]
    assert out.ambiguous_dev_match.tolist() == [False, True, False, False]
    assert out.ambiguous_repeat.tolist() == [False, True, True, False]
    assert (c["removed_dev_overlap"], c["removed_earlier_in_period"], c["listings_after"], c["ambiguous_kept"]) == (1, 1, 4, 3)
    assert c["listings_before"] == 6 and c["period"] == "primary"
    with pytest.raises(ValueError):
        load_heldout_listings(DEV_MONTHS[0])


@needs_data
def test_heldout_catalog_is_deduplicated():
    dev_units = set(pd.concat([load_listings(m).unit_id for m in DEV_MONTHS]))
    sep, c_sep = load_heldout_listings("2026-09")
    oct_, c_oct = load_heldout_listings("2026-10")
    for d in (sep, oct_):
        ident = ~d.unit_ambiguous
        assert not d.unit_id[ident].isin(dev_units).any()  # no identifiable development unit is left
        assert not d.unit_id[ident].duplicated().any()
    assert not oct_.unit_id[~oct_.unit_ambiguous].isin(set(load_listings("2026-09").unit_id)).any()  # earliest arrival kept
    assert c_oct["removed_dev_overlap"] > 0 and c_oct["listings_after"] == len(oct_)
    raw = load_listings("2026-09")
    assert c_sep["listings_before"] == len(raw) and len(sep) == len(raw) - c_sep["removed_dev_overlap"] - c_sep["removed_earlier_in_period"]


@needs_data
def test_catalog_dedups_heldout_only_and_leaves_development_unchanged():
    on, off = Catalog(), Catalog(heldout_dedup=False)
    for m in DEV_MONTHS:
        pd.testing.assert_frame_equal(on.listings(m), off.listings(m))
        pd.testing.assert_frame_equal(on.listings(m), load_listings(m))
    for design in ("uniform", "enriched"):
        a = build_market(MarketSpec("Brooklyn", "2026-08", 7, 64, 64, design=design), on)
        b = build_market(MarketSpec("Brooklyn", "2026-08", 7, 64, 64, design=design), off)
        assert a.listings.listing_uid.tolist() == b.listings.listing_uid.tolist()
        assert np.array_equal(a.inst.rank_b, b.inst.rank_b) and a.sampling_log == b.sampling_log
    assert len(on.listings("2026-09")) < len(off.listings("2026-09"))
    assert set(on.dedup_counts) == {"2026-09"} and not off.dedup_counts
    m = build_market(MarketSpec("Brooklyn", "2026-09", 99_999, 24, 24), on)  # a seed outside every study stage
    assert "heldout_dedup" in m.sampling_log and m.listings.unit_ambiguous.dtype == bool
