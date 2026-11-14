import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.pilot_outcomes import load_pilot_cache, pilot_displays, policy_events
from hiddenoptions.matching import Instance


def cache_row():
    return {
        "market_id": "m", "renter": 0, "rep": 0, "exposure": "claude_k5",
        "count": 5, "status": "ok", "model": "claude-haiku-5-5", "listings": list(range(5)),
        "raw_valid_ids": 5, "invalid_ids": 0, "duplicate_ids": 0, "extra_ids": 0,
    }


def test_duplicate_or_missing_reply_fails_instead_of_running_the_fallback(tmp_path):
    row = cache_row()
    path = tmp_path / "recs.jsonl"
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_pilot_cache(path)

    path.write_text(json.dumps(row))
    cache = load_pilot_cache(path)
    market = SimpleNamespace(spec=SimpleNamespace(market_id="m"), inst=SimpleNamespace(E=np.ones((2, 6), bool)))
    with pytest.raises(ValueError, match="required renter rows"):
        pilot_displays(market, cache)


def test_hidden_blocker_can_be_placed_and_an_unmatched_full_recipient_has_no_partner_hit():
    # Full DA: applicants 0,1 obtain programs 0,1; applicant 2 is unplaced.
    # On display: 0 only sees 1 and gets it. 0 prefers hidden vacant 0.
    inst = Instance(np.array([[0, 1], [1, 0], [0, 1]]),
                    np.array([[0, 0], [1, 1], [2, 2]]),
                    np.arange(3), np.arange(3, 5), np.ones(5))
    display = np.array([[False, True], [False, True], [False, True]])
    events = policy_events(inst, display, np.array([0, 1, -1]))
    assert events["placed"].tolist() == [True, False, False]
    assert events["partner_shown"].tolist() == [False, True, False]
    assert events["hidden_blocker"][0]


def test_planned_empty_cell_overrides_supported_target():
    import pandas as pd
    from experiments.pilot_size import planning_record

    months = ["2026-06", "2026-07", "2026-08"]
    weights = {(b, m): 0.5 for b in ("observed", "missing") for m in months}
    data = pd.DataFrame([{"month": month, "borough": "observed", "group": group,
                         "seed": seed, "uniform": 0, "haiku": int(seed == 0), "n": 5}
                        for seed in range(20) for group in ("lower_income", "higher_income")
                        for month in months])
    record = planning_record(data, weights)
    assert record["bounded_B_supported"] == 100
    assert record["protocol_B"] == 400
    assert all(r["unavailable_boroughs"] == ["missing"] for r in record["monthly"])


def test_stage2_primary_uses_documented_rule_without_expanding_replication():
    from collections import Counter
    from experiments.config import expand_labeled, load_config
    from experiments.pilot_outcomes import BASE, CONFIG

    decision = json.loads((BASE / "paired-outcomes-planning.json").read_text())
    plans = expand_labeled(load_config(CONFIG.with_name("stage2_main.json")))
    sizes = Counter(label for _, label in plans)
    assert sizes["primary"] == decision["protocol_B"] * 2 * 5
    assert sizes["replication_2025"] == 100 * 3 * 5
