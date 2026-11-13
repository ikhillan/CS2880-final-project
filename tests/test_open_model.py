"""The open-model recommender without vLLM: request building, generation bookkeeping, parsing and the merge into recs files."""
import json
import sys

import numpy as np
import pytest

from experiments import open_model_recommend as om
from hiddenoptions import llm
from hiddenoptions.data import FIRSTMOVER_CLEAN, PUMS_HOUSEHOLDS
from hiddenoptions.exposure import LlmRecommendations, show_count
from hiddenoptions.markets import Catalog, MarketSpec, build_market

needs_data = pytest.mark.skipif(not (PUMS_HOUSEHOLDS.exists() and (FIRSTMOVER_CLEAN / "firstmover_listings_2026-08.csv.gz").exists()),
                                reason="data not present")
ROW_KEYS = {"market_id", "exposure", "renter", "rep", "count", "status", "listings", "raw_valid_ids", "invalid_ids",
            "duplicate_ids", "extra_ids", "model"}


def fake_request(i: int, exposure: str = "qwen_k5", n: int = 8) -> dict:
    key = {"market_id": f"M{i // 3}", "exposure": exposure, "renter": i % 3, "rep": 0}
    return {"custom_id": f"{i:08x}" + "0" * 32, **key, "count": 3, "prompt": "x" * (100 + i),
            "id_map": {f"L{j + 1:03d}": j for j in range(n)}, "prompt_hash": llm.prompt_hash()}


class FakeGenerator:
    """Valid picks for most requests; one garbage reply, one truncation and one invalid ID, like a real run."""

    def __init__(self):
        self.calls, self.seen = 0, []

    def info(self):
        return {"engine": "fake", "engine_version": "0", "json_constraint": "fake", "chat_template_sha256": "abc"}

    def __call__(self, batch):
        self.calls += 1
        out = []
        for req in batch:
            self.seen.append(req["custom_id"])
            ids = list(req["id_map"])
            if req["renter"] == 1 and req["market_id"] == "M0":
                out.append({"text": "Sure! L001, L002", "status": "ok", "input_tokens": 10, "output_tokens": 5})
            elif req["renter"] == 2 and req["market_id"] == "M0":
                out.append({"text": '{"recommended_ids": ["L0', "status": "truncated", "input_tokens": 10, "output_tokens": 512})
            else:
                text = json.dumps({"recommended_ids": ids[:2] + ["L999"] + ids[2:4]})
                out.append({"text": text, "status": "ok", "input_tokens": 10, "output_tokens": 20})
        return out


def test_generate_parse_and_resume(tmp_path):
    reqs = [fake_request(i) for i in range(7)]
    gen, path = FakeGenerator(), tmp_path / "outputs.jsonl"
    stats = om.generate(reqs[:4], gen, path, "Qwen/Qwen2.5-32B-Instruct", batch_size=2, log=lambda *_: None)
    assert stats["generated"] == 4 and gen.calls == 2
    # Resumes: only the new requests are generated on a rerun.
    stats = om.generate(reqs, gen, path, "Qwen/Qwen2.5-32B-Instruct", batch_size=2, log=lambda *_: None)
    assert stats["already_done"] == 4 and stats["generated"] == 3 and len(gen.seen) == 7 == len(set(gen.seen))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    rows = {(r["row"]["market_id"], r["row"]["renter"]): r["row"] for r in lines}
    assert all(set(r) == ROW_KEYS for r in rows.values())  # the Claude arm's row format plus "model"
    assert rows[("M0", 1)]["status"] == "unparseable" and rows[("M0", 2)]["status"] == "truncated"
    ok = rows[("M0", 0)]
    assert ok["status"] == "ok" and ok["listings"] == [0, 1, 2] and ok["invalid_ids"] == 1 and ok["extra_ids"] == 1
    assert all("raw" in r and "text" in r["raw"] for r in lines)  # raw replies are preserved


def test_merge_keeps_other_arms_and_replaces_this_arm(tmp_path):
    recs = tmp_path / "recs.jsonl"
    claude = llm.parse_response(fake_request(0, "claude_k5"), json.dumps({"recommended_ids": ["L001", "L002", "L003"]}), "ok")
    stale = llm.parse_response(fake_request(0), None, "errored")
    recs.write_text(json.dumps({**claude, "model": "claude-opus-5-5"}) + "\n" + json.dumps({**stale, "model": "old"}) + "\n")
    workdir = tmp_path / "open_qwen"
    om.generate([fake_request(i) for i in range(3)], FakeGenerator(), workdir / "outputs-0000.jsonl", "Q", log=lambda *_: None)
    om.generate([fake_request(i) for i in range(3, 6)], FakeGenerator(), workdir / "outputs-0001.jsonl", "Q", log=lambda *_: None)
    assert om.merge(workdir, recs, "Q") == 6
    rows = [json.loads(line) for line in recs.read_text().splitlines()]
    assert len(rows) == 7
    assert [r["model"] for r in rows if r["exposure"] == "claude_k5"] == ["claude-opus-5-5"]
    qwen = [r for r in rows if r["exposure"] == "qwen_k5"]
    assert len(qwen) == 6 and {r["model"] for r in qwen} == {"Q"}  # the stale copy was replaced
    with pytest.raises(SystemExit):
        om.merge(workdir, recs, "another-model")


def test_state_refuses_changed_settings(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"pinned": {"model": "A", "revision": "r1"}}))
    om.check_state(state, {"model": "A", "revision": "r1"})
    with pytest.raises(SystemExit):
        om.check_state(state, {"model": "A", "revision": "r2"})


def test_arm_resolution_and_selection():
    cfg = {"exposures": [{"name": "claude_k5", "policy": "llm"}, {"name": "qwen_k5", "policy": "llm"},
                         {"name": "qwen_q40", "policy": "llm"}, {"name": "uniform_k5", "policy": "uniform"}]}
    assert om.resolve_arms(cfg, None, "qwen") == ["qwen_k5", "qwen_q40"]
    with pytest.raises(SystemExit):
        om.resolve_arms(cfg, ["qwen_k5", "claude_k5"], None)
    with pytest.raises(SystemExit):
        om.resolve_arms(cfg, None, "mistral")
    reqs = [fake_request(i) for i in range(10)]
    shards = [om.select(reqs, 8, s, 3) for s in range(3)]
    assert sorted(r["custom_id"] for s in shards for r in s) == sorted(r["custom_id"] for r in om.select(reqs, 8, None, None))
    assert om.messages(reqs[0])[0] == {"role": "system", "content": llm.SYSTEM_PROMPT}
    for family, (model, revision, settings) in om.ARM_MODELS.items():
        assert len(revision) == 40 and set(settings) == {"engine", "sampling"}


@needs_data
def test_end_to_end_with_fake_generator(tmp_path, monkeypatch):
    """A real market: open-model prompts equal the Claude arm's, rows merge into the recs file and drive the llm exposure."""
    cfg = {"name": "open_model_test", "grid": {"boroughs": ["Queens"], "months": ["2026-08"], "sizes": [64], "seeds": 1, "seed_start": 4},
           "exposures": [{"name": "claude_k5", "policy": "llm", "k": 5}, {"name": "qwen_k5", "policy": "llm", "k": 5},
                         {"name": "qwen_q40", "policy": "llm", "q": 0.4}],
           "methods": ["no_audit"], "llm_results": str(tmp_path / "recs.jsonl")}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(om, "ROOT", tmp_path)
    reqs = om.main([str(cfg_path), "--family", "qwen", "--dry-run"])
    assert "vllm" not in sys.modules
    catalog = Catalog()
    m = build_market(MarketSpec("Queens", "2026-08", 4, 64, 64), catalog)
    claude = {r["renter"]: r for r in (llm.build_request(m, b, cfg["exposures"][0], catalog) for b in range(64)) if r}
    k5 = [r for r in reqs if r["exposure"] == "qwen_k5"]
    assert {r["renter"] for r in k5} == set(claude)
    for r in k5:  # identical prompt text and presentation order; only the key differs
        assert r["prompt"] == claude[r["renter"]]["prompt"] and r["id_map"] == claude[r["renter"]]["id_map"]
        assert r["custom_id"] != claude[r["renter"]]["custom_id"]

    def factory(**_):
        return FakeGenerator()

    stats = om.main([str(cfg_path), "--family", "qwen", "--batch-size", "50"], generator_factory=factory)
    assert stats["generated"] == len(reqs)
    state = json.loads((tmp_path / "results/llm/open_model_test/open_qwen/state.json").read_text())
    assert state["pinned"]["model"] == "Qwen/Qwen2.5-32B-Instruct" and state["pinned"]["prompt_hash"] == llm.prompt_hash()
    recs = LlmRecommendations(tmp_path / "recs.jsonl")
    E = m.inst.E
    for name, k, q in (("qwen_k5", 5, None), ("qwen_q40", None, 0.4)):
        e0, st = recs.exposure(m, name, k=k, q=q)
        assert not (e0 & ~E).any() and np.array_equal(e0.sum(1), show_count(E.sum(1), k, q))
        assert st["missing"] == 0
    # A rerun generates nothing new and leaves the merged file the same.
    before = (tmp_path / "recs.jsonl").read_text()
    stats = om.main([str(cfg_path), "--family", "qwen"], generator_factory=factory)
    assert stats["generated"] == 0 and (tmp_path / "recs.jsonl").read_text() == before
