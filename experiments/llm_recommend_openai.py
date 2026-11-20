"""Get the rental recommender's picks from an OpenAI-format model (gpt-6-luna) through Harvard's HUIT API Platform.

The prompts are exactly the Claude ones: hiddenoptions.llm.build_request builds them (same system prompt, renter fields,
anonymous listing IDs, seeded listing order, prompt_hash and candidates_sha256). Only the endpoint and model change.
The HTTP client is the jobs experiment's (hiddenoptions.jobs.llm._client), configured by environment variables:

    JOBS_LLM_BASE_URL   e.g. https://go.apis.huit.harvard.edu/ais-openai-direct-comdev/v2
    JOBS_LLM_API_KEY    the HUIT API key (keep it outside the repo; never commit it)
    JOBS_LLM_KEY_HEADER api-key

    python -m experiments.llm_recommend_openai estimate configs/heldout_primary.json      # free: count and cost
    python -m experiments.llm_recommend_openai ping configs/heldout_primary.json          # one real request
    python -m experiments.llm_recommend_openai sync configs/heldout_primary.json --limit 20 --workers 4
    python -m experiments.llm_recommend_openai batch-submit configs/heldout_primary.json  # /batches (if the route allows)
    python -m experiments.llm_recommend_openai batch-fetch configs/heldout_primary.json   # poll; download finished batches
    python -m experiments.llm_recommend_openai collect configs/heldout_primary.json       # cache -> llm_results for run.py
    python -m experiments.llm_recommend_openai spend                                      # tokens and cost across all caches

The config needs an exposure {"name": "gpt_k5", "policy": "llm", "k": 5} and an "llm" block with model, reasoning_effort,
max_tokens and price_per_million. Responses are cached in results/llm/<config>/<arm>/responses.jsonl, so every step resumes.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from experiments.config import load_config
from experiments.llm_recommend import save_recommendations, selected_markets
from hiddenoptions import llm
from hiddenoptions.jobs.llm import _client
from hiddenoptions.markets import Catalog, InsufficientSupply, build_market

ROOT = Path(__file__).resolve().parents[1]
BATCH_CHUNK = 10_000
DEFAULT_ARM = "gpt_k5"


def arm_exposure(cfg, arm):
    for e in cfg["exposures"]:
        if e["name"] == arm and e["policy"] == "llm":
            return e
    raise SystemExit(f"no llm exposure named {arm!r} in {cfg['name']}")


def all_requests(cfg, arm, seeds=None):
    exposure, catalog, out = arm_exposure(cfg, arm), Catalog(), []
    for spec in selected_markets(cfg, seeds):
        try:
            market = build_market(spec, catalog)
        except InsufficientSupply:
            continue
        for b in range(market.inst.n_b):
            r = llm.build_request(market, b, exposure, catalog, exposure.get("rep", 0))
            if r:
                out.append(r)
    return out


def body(req, llm_cfg, json_mode):
    params = {"model": llm_cfg["model"], "max_completion_tokens": llm_cfg.get("max_tokens", 300),
              "messages": [{"role": "system", "content": llm.SYSTEM_PROMPT}, {"role": "user", "content": req["prompt"]}]}
    if json_mode == "schema":
        params["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "recommendation", "strict": True, "schema": llm.output_schema()}}
    else:  # json_object needs the word JSON and the key name in the messages
        params["response_format"] = {"type": "json_object"}
        params["messages"][0]["content"] += ' Respond with a JSON object {"recommended_ids": [...]}.'
    if "reasoning_effort" in llm_cfg:
        params["reasoning_effort"] = llm_cfg["reasoning_effort"]
    return params


def unpack(resp_dict):
    """A chat.completion (as a dict) -> the cached fields."""
    choice = resp_dict["choices"][0]
    status = {"stop": "ok", "length": "truncated", "content_filter": "refused"}.get(choice["finish_reason"], choice["finish_reason"])
    usage = resp_dict.get("usage") or {}
    return {"status": status, "text": choice["message"].get("content"), "model_returned": resp_dict.get("model"),
            "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}


def paths(cfg, arm):
    work = ROOT / "results" / "llm" / cfg["name"] / arm
    work.mkdir(parents=True, exist_ok=True)
    return work, work / "responses.jsonl", work / "openai_batches.json"


def read_cache(cache):
    out = {}
    if cache.exists():
        for line in cache.open():
            row = json.loads(line)
            out[row["custom_id"]] = row
    return out


def append_cache(cache, rows, lock=None):
    text = "".join(json.dumps(r) + "\n" for r in rows)
    if lock:
        with lock, cache.open("a") as f:
            f.write(text)
    else:
        with cache.open("a") as f:
            f.write(text)


def cost(rows, prices):
    tin = sum(r.get("input_tokens") or 0 for r in rows)
    tout = sum(r.get("output_tokens") or 0 for r in rows)
    usd = (tin * prices[0] + tout * prices[1]) / 1e6
    return tin, tout, usd


def sync(cfg, arm, llm_cfg, json_mode, limit, workers):
    _, cache, _ = paths(cfg, arm)
    done = read_cache(cache)
    todo = [r for r in all_requests(cfg, arm) if r["custom_id"] not in done][:limit]
    print(f"{len(todo):,} requests to send ({len(done):,} already cached)")
    client, lock, gap = _client(), threading.Lock(), float(llm_cfg.get("min_interval_s", 0.5))

    def one(r):
        for attempt in range(6):
            try:
                resp = client.chat.completions.create(**body(r, llm_cfg, json_mode))
                out = unpack(resp.model_dump())
                break
            except Exception as e:  # noqa: BLE001 - record and retry, then save as an error
                out = {"status": "error", "text": None, "error": repr(e)[:300]}
                time.sleep(min(60, 2 ** attempt))
        out.update(custom_id=r["custom_id"], model=llm_cfg["model"], reasoning_effort=llm_cfg.get("reasoning_effort"),
                   json_mode=json_mode, sent_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), route="sync")
        if out["status"] != "error":  # errors are not cached, so a rerun retries them
            append_cache(cache, [out], lock)
        time.sleep(gap)
        return out

    t0, n = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for out in pool.map(one, todo):
            n += 1
            if n % 200 == 0 or n == len(todo):
                print(f"  {n:,}/{len(todo):,} sent, {time.time() - t0:,.0f}s", flush=True)


def batch_submit(cfg, arm, llm_cfg, json_mode, limit):
    work, cache, state_path = paths(cfg, arm)
    done = read_cache(cache)
    state = json.loads(state_path.read_text()) if state_path.exists() else {"batches": []}
    pending = {cid for b in state["batches"] for cid in b["custom_ids"] if b.get("status") not in ("failed", "expired", "cancelled")}
    todo = [r for r in all_requests(cfg, arm) if r["custom_id"] not in done and r["custom_id"] not in pending][:limit]
    if not todo:
        return print("nothing to submit")
    client = _client()
    for i in range(0, len(todo), BATCH_CHUNK):
        chunk = todo[i:i + BATCH_CHUNK]
        f = work / f"batch_input_{len(state['batches']):03d}.jsonl"
        with f.open("w") as out:
            for r in chunk:
                out.write(json.dumps({"custom_id": r["custom_id"], "method": "POST", "url": "/v1/chat/completions",
                                      "body": body(r, llm_cfg, json_mode)}) + "\n")
        uploaded = client.files.create(file=f.open("rb"), purpose="batch")
        batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h")
        state["batches"].append({"id": batch.id, "status": batch.status, "custom_ids": [r["custom_id"] for r in chunk],
                                 "model": llm_cfg["model"], "reasoning_effort": llm_cfg.get("reasoning_effort"),
                                 "json_mode": json_mode, "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        state_path.write_text(json.dumps(state, indent=1))
        print(f"submitted {batch.id}: {len(chunk):,} requests ({batch.status})")


def batch_fetch(cfg, arm):
    _, cache, state_path = paths(cfg, arm)
    if not state_path.exists():
        return print("no batches submitted")
    state, client, done = json.loads(state_path.read_text()), _client(), read_cache(cache)
    for b in state["batches"]:
        if b.get("fetched"):
            continue
        info = client.batches.retrieve(b["id"])
        b["status"] = info.status
        counts = info.request_counts
        print(f"{b['id']}: {info.status} ({getattr(counts, 'completed', '?')}/{getattr(counts, 'total', '?')} done, "
              f"{getattr(counts, 'failed', '?')} failed)")
        if info.status == "completed":
            rows = []
            for fid in filter(None, [info.output_file_id, info.error_file_id]):
                for line in client.files.content(fid).text.splitlines():
                    item = json.loads(line)
                    if item["custom_id"] in done:
                        continue
                    resp = item.get("response") or {}
                    if resp.get("status_code") == 200:
                        out = unpack(resp["body"])
                    else:
                        out = {"status": "error", "text": None, "error": json.dumps(item.get("error") or resp)[:300]}
                    out.update(custom_id=item["custom_id"], model=b["model"], reasoning_effort=b["reasoning_effort"],
                               json_mode=b["json_mode"], sent_at=b["submitted_at"], route="batch", batch_id=b["id"])
                    rows.append(out)
            append_cache(cache, rows)
            b["fetched"] = True
            print(f"  saved {len(rows):,} replies")
    state_path.write_text(json.dumps(state, indent=1))


def collect(cfg, arm):
    _, cache, _ = paths(cfg, arm)
    cached, saved = read_cache(cache), []
    for r in all_requests(cfg, arm):
        c = cached.get(r["custom_id"])
        if c is None:
            continue
        row = llm.parse_response(r, c.get("text"), c["status"])
        row.update(model=c.get("model"), model_returned=c.get("model_returned"), reasoning_effort=c.get("reasoning_effort"),
                   json_mode=c.get("json_mode"), sent_at=c.get("sent_at"), prompt_hash=r["prompt_hash"],
                   batch_request_id=r["custom_id"])  # marks provider-run rows, so a partial run is not silently filled
        saved.append(row)
    recs = ROOT / cfg.get("llm_results", f"results/llm/{cfg['name']}/recs.jsonl")
    save_recommendations(recs, saved)
    status = {}
    for row in saved:
        status[row["status"]] = status.get(row["status"], 0) + 1
    full = sum(1 for row in saved if row["status"] == "ok" and len(row["listings"]) == row["count"])
    print(f"saved {len(saved):,} replies to {recs}; status {status}; complete valid lists {full:,}")


def spend(prices):
    rows = []
    for cache in (ROOT / "results" / "llm").glob("*/gpt_*/responses.jsonl"):
        part = list(read_cache(cache).values())
        tin, tout, usd = cost(part, prices)
        print(f"{cache.parent.parent.name}/{cache.parent.name}: {len(part):,} replies, {tin:,} in / {tout:,} out tokens, ${usd:.2f}")
        rows += part
    tin, tout, usd = cost(rows, prices)
    print(f"total: {len(rows):,} replies, ${usd:.2f} at list price (batch requests may be billed at half)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["estimate", "ping", "sync", "batch-submit", "batch-fetch", "collect", "spend"])
    ap.add_argument("config", nargs="?")
    ap.add_argument("--arm", default=DEFAULT_ARM)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json-mode", choices=["schema", "object"], default="schema")
    args = ap.parse_args()
    if args.action == "spend":
        return spend((0.1, 0.5))
    cfg = load_config(args.config)
    llm_cfg = cfg["llm"]
    if args.action == "estimate":
        reqs = all_requests(cfg, args.arm)
        tin = sum(len(r["prompt"]) + len(llm.SYSTEM_PROMPT) for r in reqs) / 4
        p = llm_cfg.get("price_per_million", [0.1, 0.5])
        usd = (tin * p[0] + len(reqs) * 40 * p[1]) / 1e6
        return print(f"{cfg['name']}/{args.arm}: {len(reqs):,} requests, ~{tin / max(len(reqs), 1):,.0f} input tokens each, "
                     f"about ${usd:.2f} at list price (${usd / 2:.2f} if batched at half price)")
    if args.action == "ping":
        r = all_requests(cfg, args.arm)[0]
        resp = _client().chat.completions.create(**body(r, llm_cfg, args.json_mode))
        out = unpack(resp.model_dump())
        row = llm.parse_response(r, out["text"], out["status"])
        return print(json.dumps({k: out[k] for k in ("status", "model_returned", "input_tokens", "output_tokens")}),
                     "| valid listings:", len(row["listings"]), "of", r["count"])
    if args.action == "sync":
        return sync(cfg, args.arm, llm_cfg, args.json_mode, args.limit, args.workers)
    if args.action == "batch-submit":
        return batch_submit(cfg, args.arm, llm_cfg, args.json_mode, args.limit)
    if args.action == "batch-fetch":
        return batch_fetch(cfg, args.arm)
    return collect(cfg, args.arm)


if __name__ == "__main__":
    main()
