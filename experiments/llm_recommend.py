"""Get the fixed AI recommender's picks through the Message Batches API (50% price).

    python -m experiments.llm_recommend estimate configs/core.json          # free: prompt counts and cost, no API call
    python -m experiments.llm_recommend show configs/core.json              # free: print one example prompt
    python -m experiments.llm_recommend submit configs/core.json --yes      # spends money
    python -m experiments.llm_recommend collect configs/core.json           # saves answers once the batches end

Keep the model pinned for the whole study: --model at submit time is saved with the batch IDs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.config import expand_markets, load_config
from hiddenoptions import llm
from hiddenoptions.markets import Catalog, InsufficientSupply, build_market

ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 20_000


def all_requests(cfg: dict, seeds: int | None, arm: str | None = None) -> list[dict]:
    """Prompts for every renter who needs a model call. `arm` keeps one model's exposures (e.g. 'claude_k5')."""
    catalog, out = Catalog(), []
    exposures = [e for e in cfg["exposures"] if e["policy"] == "llm" and (arm is None or e["name"] == arm)]
    if not exposures:
        raise SystemExit("no matching llm exposure in this config")
    starts = sorted({s.seed for s in expand_markets(cfg)})
    for spec in expand_markets(cfg):
        if seeds is not None and spec.seed >= starts[0] + seeds:
            continue
        try:
            market = build_market(spec, catalog)
        except InsufficientSupply:
            continue
        for exposure in exposures:
            for b in range(market.inst.n_b):
                r = llm.build_request(market, b, exposure, catalog, exposure.get("rep", 0))
                if r:
                    out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["estimate", "show", "submit", "collect"])
    ap.add_argument("config")
    ap.add_argument("--model", default=llm.DEFAULT_MODEL)
    ap.add_argument("--arm", help="only this llm exposure, e.g. claude_k5; each model arm is its own exposure name")
    ap.add_argument("--seeds", type=int, help="use only the first N seed blocks")
    ap.add_argument("--limit", type=int, help="use only the first N requests (for a small paid test)")
    ap.add_argument("--yes", action="store_true", help="confirm that submit spends money")
    args = ap.parse_args()
    cfg = load_config(args.config)
    workdir = ROOT / "results" / "llm" / cfg["name"] / (args.arm or "all")
    state_path, recs_path = workdir / "batches.json", ROOT / cfg.get("llm_results", f"results/llm/{cfg['name']}/recs.jsonl")

    if args.action == "collect":
        return collect(state_path, recs_path)
    requests = all_requests(cfg, args.seeds, args.arm)
    if args.limit:
        requests = requests[: args.limit]
    if args.action == "show":
        print("SYSTEM:\n" + llm.SYSTEM_PROMPT + "\n\nUSER:\n" + requests[0]["prompt"])
        return None
    if args.action == "estimate":
        for model in sorted(llm.PRICES):
            e = llm.estimate_cost(requests, model)
            print(f"{model}: {e['requests']:,} requests, about {e['mean_input_tokens']:.0f} input tokens each, "
                  f"${e['cost_usd_low']:,.2f} to ${e['cost_usd_high']:,.2f} at batch prices")
        print("Token counts are estimates (characters / 4); the output range assumes 150-800 tokens per answer including thinking.")
        return None
    if not args.yes:
        raise SystemExit("submit spends money. Run `estimate` first, then rerun with --yes.")
    return submit(requests, args.model, workdir, state_path)


def submit(requests, model, workdir, state_path):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    if state_path.exists():
        raise SystemExit(f"{state_path} exists: this config was already submitted. Run `collect`, or move the file to resubmit.")
    client = anthropic.Anthropic()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    batch_ids = []
    for i in range(0, len(requests), BATCH_SIZE):
        chunk = requests[i: i + BATCH_SIZE]
        batch = client.messages.batches.create(requests=[
            Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(**llm.request_params(r, model))) for r in chunk])
        batch_ids.append(batch.id)
        print(f"submitted {batch.id}: {len(chunk):,} requests")
        state_path.write_text(json.dumps({"model": model, "effort": llm.EFFORT, "system_prompt": llm.SYSTEM_PROMPT,
                                          "instruction": llm.INSTRUCTION, "prompt_hash": llm.prompt_hash(), "batch_ids": batch_ids}, indent=1))


def collect(state_path, recs_path):
    import anthropic

    state = json.loads(state_path.read_text())
    client = anthropic.Anthropic()
    requests = {}
    for line in (state_path.parent / "requests.jsonl").read_text().splitlines():
        r = json.loads(line)
        requests[r["custom_id"]] = r
    pending = [b for b in state["batch_ids"] if client.messages.batches.retrieve(b).processing_status != "ended"]
    if pending:
        raise SystemExit(f"still processing: {pending}. Most batches end within an hour, and all within 24 hours.")
    counts, usage_in, usage_out, saved = {}, 0, 0, []
    for batch_id in state["batch_ids"]:
        for result in client.messages.batches.results(batch_id):
            req = requests[result.custom_id]
            text, status = None, result.result.type
            if result.result.type == "succeeded":
                msg = result.result.message
                usage_in += msg.usage.input_tokens
                usage_out += msg.usage.output_tokens
                if msg.stop_reason == "refusal":
                    status = "refused"
                elif msg.stop_reason == "max_tokens":
                    status = "truncated"
                else:
                    status, text = "ok", next((b.text for b in msg.content if b.type == "text"), None)
            row = llm.parse_response(req, text, status)
            row["model"] = state["model"]
            saved.append(row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
    recs_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge with other arms already saved in the same file; this arm's rows replace any earlier copies.
    key = lambda r: (r["market_id"], r["exposure"], r["renter"], r.get("rep", 0))
    fresh = {key(r) for r in saved}
    kept = [json.loads(line) for line in recs_path.read_text().splitlines()] if recs_path.exists() else []
    merged = [r for r in kept if key(r) not in fresh] + saved
    recs_path.write_text("".join(json.dumps(r) + "\n" for r in merged))
    print(f"saved {len(saved):,} answers to {recs_path}; status counts {counts}; input tokens {usage_in:,}, output tokens {usage_out:,}")


if __name__ == "__main__":
    main()
