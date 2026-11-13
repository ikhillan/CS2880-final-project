"""Offline coverage and identity tests for Claude batch collection and scoring."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import llm_recommend as batch
from experiments.config import load_config
from hiddenoptions.exposure import LlmRecommendations, build_exposure, uniform_exposure


ROOT = Path(__file__).resolve().parents[1]
ARM = {"name": "claude_k1", "policy": "llm", "k": 1}


def market(edges):
    return SimpleNamespace(spec=SimpleNamespace(market_id="toy", seed=103),
                           inst=SimpleNamespace(E=np.array(edges, dtype=bool)))


def request(renter):
    return dict(custom_id=f"id{renter}", market_id="toy", exposure="claude_k1", renter=renter, rep=0,
                count=1, prompt="select one", prompt_hash="template", id_map={"L001": 0, "L002": 1})


def test_default_anthropic_route_and_block_sampling():
    cfg = load_config(ROOT / "configs/stage1_pilot.json")
    assert [e["name"] for e in batch.claude_arms(cfg)] == ["claude_k5", "claude_q40"]
    with pytest.raises(ValueError, match="No Claude exposure"):
        batch.claude_arms(cfg, "mistral_k5")
    specs = batch.selected_markets(cfg, seeds=1)
    assert {(s.design, s.seed) for s in specs} == {("uniform", 100), ("enriched", 150)}
    assert len(specs) == 30


def test_model_free_markets_retain_all_renters_and_full_quota():
    m = market([[1, 0], [0, 0], [0, 1]])
    visible, stats = build_exposure(m, ARM, None)
    assert np.array_equal(visible, m.inst.E)
    assert stats["whole_set"] == 2
    assert stats["status_by_renter"] == ["whole_set", "no_options", "whole_set"]
    assert build_exposure(market([[1, 1]]), ARM, None) is None


def test_partial_submission_is_not_a_model_failure(tmp_path):
    m = market([[1, 1], [1, 1]])
    file = tmp_path / "recs.jsonl"
    row = dict(market_id="toy", exposure="claude_k1", renter=0, rep=0, listings=[0],
               batch_request_id="id0", status="ok")
    file.write_text(json.dumps(row) + "\n")
    assert build_exposure(m, ARM, LlmRecommendations(file)) is None
    refused = dict(row, renter=1, batch_request_id="id1", status="refused", listings=[])
    batch.save_recommendations(file, [refused])
    visible, stats = build_exposure(m, ARM, LlmRecommendations(file))
    assert visible[0].tolist() == [True, False]
    assert np.array_equal(visible[1], uniform_exposure(m, k=1)[1])
    assert stats["refused"] == stats["full_fallback"] == 1


def fake_results(state_dir, inputs, result_ids, monkeypatch):
    state = state_dir / "batches.json"
    state.write_text(json.dumps({"batch_ids": ["batch1"], "model": "test-model", "effort": "high"}))
    (state_dir / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inputs))
    results = [SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="errored")) for cid in result_ids]
    batches = SimpleNamespace(retrieve=lambda _: SimpleNamespace(processing_status="ended"), results=lambda _: results)
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda: SimpleNamespace(
        messages=SimpleNamespace(batches=batches))))
    return state


@pytest.mark.parametrize("returned", [["id0"], ["id0", "id0"], ["unrelated"]])
def test_incomplete_or_unidentified_collection_preserves_cache(tmp_path, monkeypatch, returned):
    requests = [request(0), request(1)]
    state = fake_results(tmp_path, requests, returned, monkeypatch)
    output = tmp_path / "saved.jsonl"
    output.write_text("earlier run\n")
    with pytest.raises(ValueError):
        batch.collect(state, output, {"claude_k1"})
    assert output.read_text() == "earlier run\n"


def test_complete_collection_has_identity_and_preserves_other_arms(tmp_path, monkeypatch):
    state = fake_results(tmp_path, [request(0), request(1)], ["id1", "id0"], monkeypatch)
    output = tmp_path / "recs.jsonl"
    other = dict(exposure="qwen_k1", market_id="toy", renter=0)
    output.write_text(json.dumps(other) + "\n")
    batch.collect(state, output, {"claude_k1"})
    before = output.read_text()
    batch.collect(state, output, {"claude_k1"})
    assert before == output.read_text()
    kept, *records = [json.loads(line) for line in before.splitlines()]
    assert kept == other and len(records) == 2
    assert all(r["model"] == "test-model" and r["effort"] == "high" and len(r["prompt_sha256"]) == 64 for r in records)
    assert all(r["batch_id"] == "batch1" and r["status"] == "errored" for r in records)
    assert {r["batch_request_id"] for r in records} == {"id0", "id1"}


def test_cached_local_index_cannot_silently_point_to_a_different_unit(tmp_path):
    import pandas as pd
    from hiddenoptions.exposure import candidate_digest
    from hiddenoptions.llm import parse_response

    m = market([[1, 1]])
    m.listings = pd.DataFrame({"listing_uid": ["unit-a", "unit-b"]})
    req = request(0)
    req["candidates_sha256"] = candidate_digest(m, 0)
    row = parse_response(req, '{"recommended_ids":["L001"]}', "ok")
    store = tmp_path / "recs.jsonl"
    batch.save_recommendations(store, [row])
    table = LlmRecommendations(store)
    assert build_exposure(m, ARM, table)[0].tolist() == [[True, False]]
    m.listings = m.listings.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="candidate identities differ"):
        build_exposure(m, ARM, table)
