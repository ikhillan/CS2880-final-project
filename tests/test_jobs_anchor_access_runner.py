"""Data-free integration tests; these are not empirical jobs results."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments_jobs.anchor_access import (
    build_menus, drop_factory, evaluate_packet, preflight, replay_market, synthetic_packet)
from experiments_jobs.capacity_anchor import make_drop_order, posting_graph


def load_fixture(tmp_path):
    path = tmp_path / "panel.npz"
    synthetic_packet(path, count=3)
    with np.load(path, allow_pickle=False) as packet:
        metadata = json.loads(str(packet["metadata"]))
        block = {key[7:]: packet[key] for key in packet.files if key.startswith("m00000_")}
    return path, metadata["markets"][0], block


def test_missing_input_is_reported_without_silent_fallback(tmp_path):
    result = preflight(tmp_path)
    assert result["status"] == "missing_inputs"
    assert result["api_calls_required"] == 0
    assert len([x for x in result["inputs"] if not x["exists"]]) == 5


def test_original_random_drop_rng_is_reproduced(tmp_path):
    _, record, block = load_fixture(tmp_path)
    market = SimpleNamespace(spec=SimpleNamespace(market_id=record["market"], seed=record["seed"]))
    old = make_drop_order(market, {"policy": "random", "name": "uniform_k5"}, None)
    new = drop_factory("random", record, block)
    for b, current in [(2, np.array([0, 2, 4])), (4, np.array([1, 2, 3])), (5, np.array([0, 1]))]:
        assert np.array_equal(old(b, current), new(b, current))


def test_tie_drop_randomness_is_per_person_not_call_order(tmp_path):
    _, record, block = load_fixture(tmp_path)
    first = drop_factory("random", record, block, per_person_random=True)
    second = drop_factory("random", record, block, per_person_random=True)
    options = np.array([0, 1, 2, 3])
    first(1, options)
    assert np.array_equal(first(4, options), second(4, options))


def test_constructor_never_needs_realized_preferences(tmp_path):
    _, record, block = load_fixture(tmp_path)
    first = build_menus(record, block)
    restricted = {k: v for k, v in block.items() if k not in ("rank_b", "rank_s", "lower_wage")}
    second = build_menus(record, restricted)
    assert len(first) == 9
    for name in first:
        assert np.array_equal(first[name], second[name])
        assert np.array_equal(posting_graph(first[name], block["s2p"]).sum(1),
                              posting_graph(block["shown_gpt"], block["s2p"]).sum(1))


def test_synthetic_packet_requires_explicit_validation_flag(tmp_path):
    path, _, _ = load_fixture(tmp_path)
    with pytest.raises(ValueError, match="not empirical"):
        evaluate_packet(path, tmp_path / "out", 0, 1, 1)
    assert not (tmp_path / "out").exists()


def test_replay_roundtrip_and_manifest(tmp_path):
    path, record, block = load_fixture(tmp_path)
    rows, checks = replay_market(record, block, ties=2)
    assert len(checks) == 13  # 9 fixed arms and 2 arms per tie draw
    assert all(row["menu_budget_kept"] for row in checks)
    assert any(row["metric"] == "coverage_conditional" for row in rows)
    evaluate_packet(path, tmp_path / "out", 0, 1, 2, allow_synthetic=True)
    manifest = json.loads((tmp_path / "out/run-00.json").read_text())
    assert manifest["data_kind"] == "synthetic_validation"
    assert manifest["markets"] == 3
    table = pd.read_csv(tmp_path / "out/metrics-00.csv.gz", keep_default_na=False)
    assert set(table.market) == {"invented-0", "invented-1", "invented-2"}
    assert (table.denominator >= 0).all()


def test_complete_analysis_retains_eight_primary_endpoints(tmp_path):
    from experiments_jobs.anchor_access_analysis import analyze
    path, _, _ = load_fixture(tmp_path)
    evaluate_packet(path, tmp_path / "runs", 0, 1, 2, allow_synthetic=True)
    analyze(tmp_path / "runs", tmp_path / "analysis", allow_synthetic=True, reps=100)
    primary = pd.read_csv(tmp_path / "analysis/primary.csv")
    assert len(primary) == 8
    assert set(primary.family_size) == {8}
    assert primary.simultaneous_family_complete.all()
    manifest = json.loads((tmp_path / "analysis/manifest.json").read_text())
    assert manifest["original_placement_reproduced"] is None
    assert manifest["data_kind"] == "synthetic_validation"


def test_missing_tie_cannot_be_silently_averaged(tmp_path):
    from experiments_jobs.anchor_access_analysis import load_runs
    path, _, _ = load_fixture(tmp_path)
    evaluate_packet(path, tmp_path / "runs", 0, 1, 2, allow_synthetic=True)
    target = tmp_path / "runs/metrics-00.csv.gz"
    table = pd.read_csv(target, keep_default_na=False)
    missing = table.index[(table.arm == "anchor_gpt_ties") & (table.tie == 1)][0]
    table.drop(index=missing).to_csv(target, index=False)
    with pytest.raises(ValueError, match="Incomplete or duplicated policy randomization"):
        load_runs(tmp_path / "runs", allow_synthetic=True)
