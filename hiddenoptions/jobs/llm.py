"""The fixed AI recommender for job markets: one pinned model and prompt, asked once per seeker, with saved answers.

Mirrors hiddenoptions/llm.py, with two differences. Prompts carry structured fields only: no employer name, no rcid, no job_id
and no posting text, per the Revelio approval. And the client is an OpenAI-format chat endpoint (the HUIT API Platform, or a
local OpenAI-compatible server), called sequentially with a local cache, because that route has no batch API.

    python -m hiddenoptions.jobs.llm estimate configs/jobs_frozen.json   # free: request count and cost estimate
    python -m hiddenoptions.jobs.llm show configs/jobs_frozen.json       # free: print one prompt
    python -m hiddenoptions.jobs.llm ping configs/jobs_frozen.json       # one request built from SYNTHETIC postings (no Revelio data)
    python -m hiddenoptions.jobs.llm run configs/jobs_frozen.json        # sends Revelio-derived prompts; resumable
    python -m hiddenoptions.jobs.llm collect configs/jobs_frozen.json    # cache -> recs.jsonl for experiments_jobs.run

HUIT AI Sandbox route (approved by Revelio; manual, one chat per market):
    python -m hiddenoptions.jobs.llm export-sandbox configs/jobs_frozen.json   # writes one prompt file per market
    #   paste each prompt into a NEW Sandbox chat; save the reply as <market>.answer.txt next to the prompt
    python -m hiddenoptions.jobs.llm import-sandbox configs/jobs_frozen.json   # answers -> recs.jsonl

`run` refuses to send anything unless the config's llm.route appears in license_and_usage.approved_llm_routes of
revelio_2025_manifest.json. Add the route there only after HUIT and Revelio have both confirmed it in writing.

Environment: JOBS_LLM_BASE_URL (endpoint base URL), JOBS_LLM_API_KEY, and optionally JOBS_LLM_KEY_HEADER if the endpoint wants
the key in a header other than Authorization (for example an API gateway's "api-key"). Never commit any of these.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

from hiddenoptions.exposure import show_count
from hiddenoptions.llm import parse_response

ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = ROOT / "revelio_2025_manifest.json"
SYSTEM_PROMPT = ("You are a job-search assistant for people in the New York City area. You receive one job seeker's profile and "
                 "the job postings available to them. Recommend the postings this person should apply to, best first. "
                 "Use only IDs from the list. Respond with a JSON object of the form {\"recommended_ids\": [\"P001\", ...]}.")
SENIORITY_LABELS = {1: "entry level", 2: "junior", 3: "associate", 4: "manager", 5: "director", 6: "executive", 7: "senior executive"}
OCCUPATION_LABELS = {"291141": "registered nurse", "533032": "heavy and tractor-trailer truck driver", "412031": "retail salesperson",
                     "311131": "nursing assistant", "113031": "financial manager", "151252": "software developer"}
ARRANGEMENT = {"Fully In Office": "in person", "Partially In Office": "hybrid"}


# ---- prompts -----------------------------------------------------------------------------------------------------------------
def _postings(market):
    """First slot of each posting, and seeker x posting feasibility."""
    s2p = market.slot_to_posting
    first = np.unique(s2p, return_index=True)[1]
    E_post = np.zeros((market.inst.n_b, len(first)), bool)
    for slot, post in enumerate(s2p):
        E_post[:, post] |= market.inst.E[:, slot]
    return market.listings.iloc[first].reset_index(drop=True), E_post


def seeker_text(row, cell: str) -> str:
    occ = OCCUPATION_LABELS.get(cell, f"occupation code {cell}")
    years = int(row.years_experience)
    return (f"Occupation: {occ}\n"
            f"Experience: about {years} year{'s' if years != 1 else ''}\n"
            f"Education: {'bachelor’s degree or higher' if row.bachelors else 'no bachelor’s degree'}\n"
            f"Lowest acceptable annual pay: ${0.8 * row.wage_month:,.0f}\n"
            f"Location: New York City")


def posting_text(row, pid: str) -> str:
    rng = f"${row.salary_min:,.0f}–${row.salary_max:,.0f}" if np.isfinite(row.salary_min) and np.isfinite(row.salary_max) else f"${row.salary:,.0f}"
    return f"{pid} | {rng} a year | {SENIORITY_LABELS.get(int(row.seniority), 'unspecified level')}"


def build_request(market, b: int, exposure: dict, rep: int = 0) -> dict | None:
    posts, E_post = _postings(market)
    options = np.nonzero(E_post[b])[0]
    if len(options) == 0:
        return None
    count = int(show_count(np.array([len(options)]), exposure.get("k"), exposure.get("q"))[0])
    salt = f"{market.spec.market_id}|{exposure['name']}|{b}|{rep}"
    rng = np.random.default_rng([market.spec.seed, zlib.crc32(salt.encode())])
    shuffled = options[rng.permutation(len(options))]
    id_map = {f"P{i + 1:03d}": int(p) for i, p in enumerate(shuffled)}
    table = "\n".join(posting_text(posts.iloc[p], pid) for pid, p in id_map.items())
    noun = "posting" if count == 1 else "postings"
    prompt = (f"Job seeker:\n{seeker_text(market.renters.iloc[b], market.spec.cell)}\n\nAvailable postings ({len(options)}):\n{table}\n\n"
              f"Recommend exactly {count} {noun} for this person, best first.")
    key = {"market_id": market.spec.market_id, "exposure": exposure["name"], "renter": b, "rep": rep}
    custom_id = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:40]
    return {"custom_id": custom_id, **key, "count": count, "prompt": prompt, "id_map": id_map}


def synthetic_request() -> dict:
    """A request built from invented postings, for testing the endpoint before any Revelio data may be sent."""
    rng = np.random.default_rng(0)
    lines, id_map = [], {}
    for i in range(12):
        lo = int(rng.integers(55, 110)) * 1000
        lines.append(f"P{i + 1:03d} | ${lo:,}–${lo + 15000:,} a year | {SENIORITY_LABELS[int(rng.integers(1, 4))]}")
        id_map[f"P{i + 1:03d}"] = i
    prompt = ("Job seeker:\nOccupation: registered nurse\nExperience: about 4 years\nEducation: bachelor’s degree or higher\n"
              "Lowest acceptable annual pay: $72,000\nLocation: New York City\n\nAvailable postings (12):\n" + "\n".join(lines) +
              "\n\nRecommend exactly 5 postings for this person, best first.")
    return {"custom_id": "synthetic-ping", "market_id": "synthetic", "exposure": "ping", "renter": 0, "rep": 0, "count": 5,
            "prompt": prompt, "id_map": id_map}


def all_requests(cfg: dict) -> list[dict]:
    from experiments_jobs.config import expand_labeled
    from hiddenoptions.jobs.markets import JobCatalog, build_market
    exposures = [e for e in cfg["exposures"] if e["policy"] == "llm"]
    if not exposures:
        raise SystemExit("this config has no llm exposure")
    catalog, out = JobCatalog(months=tuple(cfg["grid"]["months"])), []
    for spec, _ in expand_labeled(cfg):
        market = build_market(spec, catalog)
        for exposure in exposures:
            for b in range(market.inst.n_b):
                r = build_request(market, b, exposure, exposure.get("rep", 0))
                if r:
                    out.append(r)
    return out


# ---- client ------------------------------------------------------------------------------------------------------------------
def _client():
    from openai import OpenAI
    base, key = os.environ.get("JOBS_LLM_BASE_URL"), os.environ.get("JOBS_LLM_API_KEY")
    if not base or not key:
        raise SystemExit("set JOBS_LLM_BASE_URL and JOBS_LLM_API_KEY")
    header = os.environ.get("JOBS_LLM_KEY_HEADER")
    return OpenAI(base_url=base, api_key=key, default_headers={header: key} if header else None)


def _call(client, llm_cfg: dict, prompt: str) -> dict:
    # Current OpenAI models are reasoning models: Chat Completions takes max_completion_tokens (not max_tokens) and a
    # reasoning_effort. The study pins effort in the config so the recommender is fixed; "none" makes it a plain ranker.
    params = {"model": llm_cfg["model"], "max_completion_tokens": llm_cfg.get("max_tokens", 300),
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]}
    if "reasoning_effort" in llm_cfg:
        params["reasoning_effort"] = llm_cfg["reasoning_effort"]
    if "temperature" in llm_cfg:
        params["temperature"] = llm_cfg["temperature"]
    resp = client.chat.completions.create(**params)
    choice = resp.choices[0]
    status = {"stop": "ok", "length": "truncated", "content_filter": "refused"}.get(choice.finish_reason, choice.finish_reason)
    usage = resp.usage
    return {"status": status, "text": choice.message.content, "model_returned": resp.model,
            "input_tokens": getattr(usage, "prompt_tokens", None), "output_tokens": getattr(usage, "completion_tokens", None)}


def _approved(route: str) -> bool:
    if not PROVENANCE.exists():
        return False
    lic = json.loads(PROVENANCE.read_text()).get("license_and_usage", {})
    return route in (lic.get("approved_llm_routes") or [])


def run(cfg: dict, cache: Path, limit: int | None = None, workers: int = 1) -> None:
    """Send uncached requests. Resumable: rerunning skips anything already in the cache. `limit` caps how many are sent this
    time (for a small paid test); `workers` sends that many requests at once (keep it at or under the endpoint's rate limit)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    llm_cfg = cfg["llm"]
    if not _approved(llm_cfg["route"]):
        raise SystemExit(f"route {llm_cfg['route']!r} is not in approved_llm_routes of {PROVENANCE.name}. "
                         "Add it only after HUIT and Revelio have confirmed it in writing.")
    done = set()
    if cache.exists():
        done = {json.loads(l)["custom_id"] for l in cache.read_text().splitlines() if l.strip()}
    requests = [r for r in all_requests(cfg) if r["custom_id"] not in done]
    if limit:
        requests = requests[:limit]
    print(f"{len(done):,} cached, sending {len(requests):,} with {llm_cfg['model']} via {llm_cfg['route']} on {workers} worker(s)", flush=True)
    client, gap, lock = _client(), float(llm_cfg.get("min_interval_s", 0.5)), threading.Lock()
    cache.parent.mkdir(parents=True, exist_ok=True)

    def one(r):
        for attempt in range(6):
            try:
                out = _call(client, llm_cfg, r["prompt"])
                break
            except Exception as e:  # rate limits and transient errors: back off, then record the failure
                if attempt == 5:
                    out = {"status": "error", "text": None, "error": repr(e)[:300]}
                else:
                    time.sleep(min(60, 2 ** attempt))
        time.sleep(gap)
        return {"custom_id": r["custom_id"], "model": llm_cfg["model"], "sent_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **out}

    with open(cache, "a") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, r) for r in requests]
        for i, f in enumerate(as_completed(futures), 1):
            with lock:
                fh.write(json.dumps(f.result()) + "\n")
                fh.flush()
            if i % 200 == 0 or i == len(requests):
                print(f"  {i:,}/{len(requests):,}", flush=True)


def collect(cfg: dict, cache: Path, recs: Path) -> None:
    answers = {}
    for line in cache.read_text().splitlines():
        a = json.loads(line)
        answers[a["custom_id"]] = a  # last attempt wins
    saved, counts, tin, tout = [], {}, 0, 0
    for r in all_requests(cfg):
        a = answers.get(r["custom_id"])
        if a is None:
            continue
        row = parse_response(r, a.get("text"), a.get("status", "error"))
        # row["listings"] are posting indices; JobLlmRecommendations lifts them to slots
        row["model"] = a.get("model")
        saved.append(row)
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        tin += a.get("input_tokens") or 0
        tout += a.get("output_tokens") or 0
    recs.parent.mkdir(parents=True, exist_ok=True)
    recs.write_text("".join(json.dumps(r) + "\n" for r in saved))
    print(f"saved {len(saved):,} answers to {recs}; status {counts}; tokens in {tin:,}, out {tout:,}")


# ---- HUIT AI Sandbox (manual) ------------------------------------------------------------------------------------------------
SANDBOX_INSTRUCTIONS = (
    "You are a job-search assistant for people in the New York City area. Below are job postings and several job seekers. "
    "Treat each seeker independently, as if you were advising only that person: do not try to spread postings across seekers, "
    "and recommending the same posting to several seekers is fine. For each seeker, choose exactly the requested number of "
    "postings from that seeker's own list, best first. Reply with ONLY a JSON object mapping each seeker ID to a list of "
    "posting IDs, for example {\"S01\": [\"P004\", \"P011\"], \"S02\": [...]}. No other text.")


def _sandbox_markets(cfg: dict):
    from experiments_jobs.config import expand_labeled
    sb = cfg["sandbox"]
    keep = [(s, lab) for s, lab in expand_labeled(cfg)
            if lab == sb["variant"] and s.month in sb["months"] and s.seed < sb["seeds"]]
    if not keep:
        raise SystemExit("no markets match the sandbox block (variant, months, seeds)")
    return keep


def export_sandbox(cfg: dict, folder: Path) -> None:
    from hiddenoptions.jobs.markets import JobCatalog, build_market
    sb = cfg["sandbox"]
    exposure = next(e for e in cfg["exposures"] if e["name"] == sb["exposure"])
    catalog = JobCatalog(months=tuple(sb["months"]))
    folder.mkdir(parents=True, exist_ok=True)
    index = {}
    for n, (spec, _) in enumerate(_sandbox_markets(cfg), 1):
        market = build_market(spec, catalog)
        posts, E_post = _postings(market)
        rng = np.random.default_rng([spec.seed, zlib.crc32(f"{spec.market_id}|sandbox|{exposure['name']}".encode())])
        order = rng.permutation(len(posts))
        pid = {int(p): f"P{i + 1:03d}" for i, p in enumerate(order)}
        lines = [posting_text(posts.iloc[p], pid[int(p)]) for p in order]
        seekers, smap = [], {}
        for b in range(market.inst.n_b):
            options = np.nonzero(E_post[b])[0]
            if len(options) == 0:
                continue
            count = int(show_count(np.array([len(options)]), exposure.get("k"), exposure.get("q"))[0])
            sid = f"S{len(smap) + 1:02d}"
            allowed = sorted(pid[int(p)] for p in options)
            smap[sid] = {"renter": b, "count": count, "allowed": allowed}
            seekers.append(f"{sid}\n{seeker_text(market.renters.iloc[b], spec.cell)}\n"
                           f"May apply to: {', '.join(allowed)}\nRecommend exactly {count}.")
        tag = f"m{n:03d}"
        prompt = (SANDBOX_INSTRUCTIONS + f"\n\nJOB POSTINGS ({len(lines)}):\n" + "\n".join(lines) +
                  f"\n\nJOB SEEKERS ({len(seekers)}):\n\n" + "\n\n".join(seekers))
        (folder / f"{tag}.prompt.txt").write_text(prompt)
        index[tag] = {"market_id": spec.market_id, "exposure": exposure["name"], "seekers": smap,
                      "posting_index": {v: k for k, v in pid.items()}}
    (folder / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} prompts to {folder}. For each: open a NEW Sandbox chat, paste mNNN.prompt.txt, and save the "
          f"reply as mNNN.answer.txt in the same folder. Note the model name shown and the date in sandbox_log.csv.")


def import_sandbox(cfg: dict, folder: Path, recs: Path) -> None:
    import re
    index = json.loads((folder / "index.json").read_text())
    rows, counts = [], {}
    for tag, info in index.items():
        path = folder / f"{tag}.answer.txt"
        answer, status = None, "missing"
        if path.exists():
            text = path.read_text().strip()
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
            try:
                answer, status = json.loads(text[text.index("{"): text.rindex("}") + 1]), "ok"
            except (ValueError, json.JSONDecodeError):
                status = "unparseable"
        for sid, s in info["seekers"].items():
            row = {"market_id": info["market_id"], "exposure": info["exposure"], "renter": s["renter"], "rep": 0,
                   "count": s["count"], "status": status, "listings": [], "invalid_ids": 0, "extra_ids": 0, "model": "huit_ai_sandbox"}
            if status == "ok":
                picks = answer.get(sid)
                if picks is None:
                    row["status"] = "missing_seeker"
                else:
                    seen = []
                    for p in picks:
                        if p in s["allowed"] and info["posting_index"][p] not in seen:
                            seen.append(info["posting_index"][p])
                        else:
                            row["invalid_ids"] += 1
                    row["extra_ids"] = max(0, len(seen) - s["count"])
                    row["listings"] = seen[: s["count"]]
            rows.append(row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
    recs.parent.mkdir(parents=True, exist_ok=True)
    recs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"saved {len(rows):,} seeker answers from {len(index)} markets to {recs}; status {counts}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["estimate", "show", "ping", "run", "collect", "export-sandbox", "import-sandbox"])
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, help="run: send at most this many requests this time")
    ap.add_argument("--workers", type=int, default=1, help="run: concurrent requests (stay within the rate limit)")
    args = ap.parse_args()
    from experiments_jobs.config import load_config
    cfg = load_config(args.config)
    workdir = ROOT / "results" / "llm" / cfg["name"]
    cache, recs = workdir / "responses.jsonl", ROOT / cfg.get("llm_results", f"results/llm/{cfg['name']}/recs.jsonl")
    sandbox_dir = workdir / "sandbox"
    if args.action == "export-sandbox":
        return export_sandbox(cfg, sandbox_dir)
    if args.action == "import-sandbox":
        return import_sandbox(cfg, sandbox_dir, recs)
    if args.action == "ping":
        r = synthetic_request()
        out = _call(_client(), cfg["llm"], r["prompt"])
        print(json.dumps(out, indent=1))
        print(parse_response(r, out["text"], out["status"]))
        return
    if args.action == "collect":
        return collect(cfg, cache, recs)
    requests = all_requests(cfg)
    if args.action == "show":
        print("SYSTEM:\n" + SYSTEM_PROMPT + "\n\nUSER:\n" + requests[0]["prompt"])
        return
    if args.action == "estimate":
        chars = sum(len(r["prompt"]) + len(SYSTEM_PROMPT) for r in requests)
        price_in, price_out = cfg["llm"].get("price_per_million", [float("nan"), float("nan")])
        out_tok = cfg["llm"].get("max_tokens", 300) / 3  # JSON of a few IDs; well under max_tokens in practice
        cost = (chars / 4 * price_in + len(requests) * out_tok * price_out) / 1e6
        print(f"{len(requests):,} requests, about {chars / 4 / max(len(requests), 1):.0f} input tokens each, about ${cost:,.2f}")
        print("Estimate only (characters / 4). Run `ping`, then a small `run`, and use the logged token counts.")
        return
    run(cfg, cache, args.limit, args.workers)


if __name__ == "__main__":
    main()
