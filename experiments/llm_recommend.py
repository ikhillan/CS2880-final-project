"""Get the fixed AI recommender's picks through the Message Batches API (50% price).

    python -m experiments.llm_recommend estimate configs/stage1_pilot.json          # free: prompt counts and cost, no API call
    python -m experiments.llm_recommend show configs/stage1_pilot.json              # free: print one example prompt
    python -m experiments.llm_recommend submit configs/stage1_pilot.json --yes      # spends money
    python -m experiments.llm_recommend collect configs/stage1_pilot.json           # saves answers once the batches end

Keep the model pinned for the whole study: --model at submit time is saved with the batch IDs.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

from experiments.config import expand_labeled, load_config
from hiddenoptions import llm
from hiddenoptions.markets import Catalog, InsufficientSupply, build_market

ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 20_000


def claude_arms(cfg, arm=None):
    """The Anthropic endpoint can populate Claude arms and legacy provider-neutral LLM arms."""
    choices = [e for e in cfg["exposures"] if e["policy"] == "llm"
               and e.get("provider", e["name"].split("_")[0]) in {"anthropic", "claude", "llm"}]
    selected = [e for e in choices if arm is None or e["name"] == arm]
    if not selected:
        raise ValueError(f"No Claude exposure for {arm!r}. Available: {[e['name'] for e in choices]}")
    return selected


def selected_markets(cfg, seeds):
    markets = expand_labeled(cfg)
    if seeds is None:
        return [spec for spec, label in markets]
    if seeds < 1:
        raise ValueError("--seeds must be positive")
    starts = {}
    for spec, label in markets:
        starts[label] = min(starts.get(label, spec.seed), spec.seed)
    return [spec for spec, label in markets if spec.seed < starts[label] + seeds]


def all_requests(cfg: dict, seeds: int | None, arm: str | list[str] | None = None,
                 catalog: Catalog | None = None, *, markets=None) -> list[dict]:
    """Build the shared prompts. A string or no arm uses Claude routing; a list selects explicit arms."""
    if isinstance(arm, list):
        names = set(arm)
        exposures = [e for e in cfg["exposures"] if e["policy"] == "llm" and e["name"] in names]
        if not names or {e["name"] for e in exposures} != names:
            raise ValueError(f"Unknown or empty LLM exposure selection: {arm}")
    else:
        exposures = claude_arms(cfg, arm)
    catalog = Catalog() if catalog is None else catalog
    out = []
    specs = selected_markets(cfg, seeds) if markets is None else markets
    for spec in specs:
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
    ap.add_argument("--effort", default=llm.EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--arm", help="only this llm exposure, e.g. claude_k5; each model arm is its own exposure name")
    ap.add_argument("--seeds", type=int, help="use the first N seed blocks in each configured design")
    ap.add_argument("--limit", type=int, help="use only the first N requests (for a small paid test)")
    ap.add_argument("--yes", action="store_true", help="confirm that submit spends money")
    args = ap.parse_args()
    cfg = load_config(args.config)
    try:
        selected = claude_arms(cfg, args.arm)
    except ValueError as e:
        ap.error(str(e))
    workdir = ROOT / "results" / "llm" / cfg["name"] / (args.arm or "all")
    state_path, recs_path = workdir / "batches.json", ROOT / cfg.get("llm_results", f"results/llm/{cfg['name']}/recs.jsonl")

    if args.action == "collect":
        return collect(state_path, recs_path, {e["name"] for e in selected})
    requests = all_requests(cfg, args.seeds, args.arm)
    if args.limit is not None:
        if args.limit < 1:
            ap.error("--limit must be positive")
        requests = requests[: args.limit]
    print("Claude exposures:", ", ".join(e["name"] for e in selected))
    if not requests:
        print("No renter requires a model call for these markets and arms.")
        return None
    if args.action == "show":
        print("SYSTEM:\n" + llm.SYSTEM_PROMPT + "\n\nUSER:\n" + requests[0]["prompt"])
        return None
    if args.action == "estimate":
        for model in sorted(llm.PRICES):
            e = llm.estimate_cost(requests, model)
            print(f"{model}: {e['requests']:,} requests, about {e['mean_input_tokens']:.0f} input tokens each, "
                  f"${e['cost_usd_low']:,.2f} to ${e['cost_usd_high']:,.2f} at batch prices")
        print("Token counts are estimates (characters / 4); the output range assumes 500-3,000 tokens per answer including high-effort thinking.")
        return None
    if not args.yes:
        raise SystemExit("submit spends money. Run `estimate` first, then rerun with --yes.")
    return submit(requests, args.model, workdir, state_path, args.effort)


def submit(requests, model, workdir, state_path, effort=llm.EFFORT):
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
            Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(**llm.request_params(r, model, effort))) for r in chunk])
        batch_ids.append(batch.id)
        print(f"submitted {batch.id}: {len(chunk):,} requests")
        state_path.write_text(json.dumps({"model": model, "effort": effort, "system_prompt": llm.SYSTEM_PROMPT,
                                          "instruction": llm.INSTRUCTION, "prompt_hash": llm.prompt_hash(), "batch_ids": batch_ids}, indent=1))


def save_recommendations(recs_path, saved, *, generation_hashes=None):
    """Atomically merge complete batches. Parallel collectors must see one another's updates."""
    recs_path.parent.mkdir(parents=True, exist_ok=True)
    def key(row):
        return row["market_id"], row["exposure"], row["renter"], row.get("rep", 0)

    with recs_path.with_suffix(recs_path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fresh = {key(row) for row in saved}
        if len(fresh) != len(saved):
            raise ValueError("Multiple replies for one market, arm, renter and repetition")
        with tempfile.NamedTemporaryFile("w", dir=recs_path.parent, delete=False) as tmp:
            try:
                if recs_path.exists():
                    with recs_path.open() as existing:
                        for line in existing:
                            row = json.loads(line)
                            if key(row) in fresh:
                                continue
                            if generation_hashes and row["exposure"] in generation_hashes:
                                if row.get("generation_sha256") != generation_hashes[row["exposure"]]:
                                    raise ValueError(f"Cannot combine different model settings for {row['exposure']}. "
                                                     "Use another exposure name or results file.")
                            tmp.write(line.rstrip("\n") + "\n")
                for row in saved:
                    tmp.write(json.dumps(row) + "\n")
                tmp.flush()
                os.fsync(tmp.fileno())
                os.replace(tmp.name, recs_path)
            finally:
                Path(tmp.name).unlink(missing_ok=True)


def collect(state_path, recs_path, allowed_arms):
    import anthropic

    state = json.loads(state_path.read_text())
    requests = {}
    for line in (state_path.parent / "requests.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["exposure"] not in allowed_arms:
            raise ValueError(f"The Claude request manifest includes an incompatible arm: {r['exposure']}")
        if r["custom_id"] in requests:
            raise ValueError(f"Duplicate request in manifest: {r['custom_id']}")
        requests[r["custom_id"]] = r
    client = anthropic.Anthropic()
    pending = [b for b in state["batch_ids"] if client.messages.batches.retrieve(b).processing_status != "ended"]
    if pending:
        raise SystemExit(f"Still processing: {pending}")
    counts, usage_in, usage_out, saved = {}, 0, 0, []
    returned = set()
    for batch_id in state["batch_ids"]:
        for result in client.messages.batches.results(batch_id):
            if result.custom_id not in requests or result.custom_id in returned:
                raise ValueError(f"Unknown or repeated batch result ID: {result.custom_id}")
            returned.add(result.custom_id)
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
            row.update(model=state["model"], effort=state.get("effort"), batch_id=batch_id,
                       batch_request_id=req["custom_id"], prompt_hash=req["prompt_hash"],
                       prompt_sha256=hashlib.sha256(req["prompt"].encode()).hexdigest())
            saved.append(row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
    missing = requests.keys() - returned
    if missing:
        raise ValueError(f"{len(missing)} request IDs are absent from ended batches. Check the submission ledger before scoring.")
    save_recommendations(recs_path, saved)
    print(f"Saved {len(saved):,} answers to {recs_path}; status counts {counts}; input tokens {usage_in:,}, output tokens {usage_out:,}")


if __name__ == "__main__":
    main()
