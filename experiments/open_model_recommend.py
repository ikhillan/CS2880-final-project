"""Recommendations from the open-model arms (protocol sections 5 and 8), run locally with vLLM offline inference.

    python -m experiments.open_model_recommend configs/stage2_main.json --family qwen --dry-run      # no GPU, no vLLM
    python -m experiments.open_model_recommend configs/stage2_main.json --family qwen                # one GPU replica
    python -m experiments.open_model_recommend configs/stage2_main.json --arm mistral_k5 --shard 0 --num-shards 4
    python -m experiments.open_model_recommend configs/stage2_main.json --family mistral --merge-only
    python -m experiments.open_model_recommend configs/stage2_main.json --family qwen --limit 500 --tag profile_h200 --no-merge

Arms are the config's `llm` exposures. The model is picked by the exposure name's prefix (qwen_k5 -> Qwen, mistral_q40 ->
Mistral; see ARM_MODELS) unless --model is given; every arm in one run must use the same model, so one model load serves
e.g. qwen_k5 and qwen_q40.

Identical to the Claude arm (experiments/llm_recommend.py):
  * prompts, listing IDs and presentation order come from hiddenoptions.llm.build_request, so a renter's prompt and its
    permutation are the same for every model family (paired comparisons);
  * the system prompt is hiddenoptions.llm.SYSTEM_PROMPT and the reply schema is hiddenoptions.llm.output_schema(), the same
    generic {"recommended_ids": [string, ...]} schema Claude gets (not restricted to the renter's IDs, so invalid-ID rates
    stay comparable across arms);
  * replies go through hiddenoptions.llm.parse_response and are saved as the same rows (plus "model"), merged into the
    config's llm_results file the way llm_recommend.collect merges: rows of these arms replace earlier copies, other
    arms' rows are kept.
Specific to the open models: the model's own chat template (Hugging Face template, or mistral_common for Mistral as its
model card recommends), a pinned weight/tokenizer revision, JSON-constrained decoding, the model card's recommended sampling
settings, and a per-request seed derived from the request key. Batching and GPU kernels may still affect the sampled reply.

Outputs (results/llm/<config>/open_<family>[_<tag>]/):
  outputs[-<shard>].jsonl  one line per request: raw reply text, finish reason, token counts and the parsed row. Reruns
                           skip requests already there, so an interrupted or requeued job resumes.
  state[-<shard>].json     completion status, exact request digest, model, revision, engine and version, chat template hash,
                           hardware, timings and status counts. A rerun with different settings is refused.
Unrestricted replicas divide the market specs before constructing prompts. With --limit the sample is selected globally.
Sharded runs do not touch the recs file; run --merge-only once every shard has finished.
vLLM is imported only when generating (it does not install on macOS); --dry-run, --merge-only and the tests never need it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import tempfile
import subprocess
import sys
import time
from pathlib import Path

from experiments.config import load_config
from experiments.llm_recommend import all_requests, save_recommendations, selected_markets
from hiddenoptions import llm

ROOT = Path(__file__).resolve().parents[1]
# family -> (model, revision, settings). Revisions are the Hugging Face commit SHAs of the main branch on 2026-11-13
# (huggingface.co/api/models/<model>); they pin both the weights and the tokenizer files. Sampling follows each model card:
# Qwen's generation_config.json at that commit, and Mistral's recommended temperature of 0.15.
ARM_MODELS = {
    "qwen": ("Qwen/Qwen2.5-32B-Instruct", "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
             {"engine": {"tokenizer_mode": "auto"},
              "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}}),
    "mistral": ("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "68faf511d618ef198fef186659617cfd2eb8e33a",
                {"engine": {"tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral",
                            "limit_mm_per_prompt": {"image": 0}},  # text only
                 "sampling": {"temperature": 0.15}}),
}
MAX_TOKENS = 512  # the reply is a short JSON object; neither model thinks before answering
RETRIES = 3  # protocol: retry transient failures up to three times on the same payload


def family_of(arm: str) -> str:
    return arm.split("_", 1)[0]


def resolve_arms(cfg: dict, arms: list[str] | None, family: str | None) -> list[str]:
    llm_arms = [e["name"] for e in cfg["exposures"] if e["policy"] == "llm"]
    if arms:
        missing = [a for a in arms if a not in llm_arms]
        if missing:
            raise SystemExit(f"not llm exposures in this config: {missing}; available {llm_arms}")
        chosen = arms
    elif family:
        chosen = [a for a in llm_arms if family_of(a) == family]
    else:
        raise SystemExit("give --arm (repeatable) or --family")
    if not chosen:
        raise SystemExit(f"no llm exposure for family {family!r} in this config; available {llm_arms}")
    families = {family_of(a) for a in chosen}
    if len(families) > 1:
        raise SystemExit(f"arms {chosen} belong to different model families; run one family at a time")
    if not families.issubset(ARM_MODELS) or (family and families != {family}):
        raise SystemExit(f"Open-model arms must match one configured Qwen or Mistral family: {chosen}")
    if len(chosen) != len(set(chosen)):
        raise SystemExit("Each arm must be specified once")
    return sorted(chosen)


def messages(req: dict) -> list[dict]:
    """The same system prompt and user prompt the Claude arm sends; the model's chat template renders them."""
    return [{"role": "system", "content": llm.SYSTEM_PROMPT}, {"role": "user", "content": req["prompt"]}]


def request_seed(req: dict) -> int:
    if "sampling_seed" in req:
        return req["sampling_seed"]
    return int(req["custom_id"][:8], 16)


def profile_order(request: dict) -> tuple[str, str]:
    """Sample the same renter, repetition and actual budget in every model family."""
    case = [request["market_id"], request["renter"], request["rep"], request["count"]]
    # Different policies can yield the same integer quota on one renter (e.g. top-5 and q40).
    # The family-independent suffix keeps both records in deterministic order in that case.
    policy = request["exposure"].partition("_")[2]
    return digest(case), policy


def select(requests: list[dict], limit: int | None, shard: int | None, num_shards: int | None) -> list[dict]:
    """Select paired profile cases globally, then distribute that common sample over replicas."""
    requests = sorted(requests, key=profile_order)
    if limit:
        requests = requests[:limit]
    if shard is not None:
        requests = requests[shard::num_shards]
    return requests


class VllmGenerator:
    """vLLM offline inference. Call with a list of requests; returns one result dict per request, in order."""

    def __init__(self, model: str, revision: str, engine: dict, sampling: dict, max_tokens: int = MAX_TOKENS,
                 max_model_len: int = 8192, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.90,
                 dtype: str = "bfloat16", allow_unconstrained: bool = False):
        import vllm  # lazy: not installable on macOS, and not needed for --dry-run or tests
        from vllm import LLM, SamplingParams

        self.vllm, self.SamplingParams = vllm, SamplingParams
        self.sampling, self.max_tokens, self.max_model_len = sampling, max_tokens, max_model_len
        self.llm = LLM(model=model, revision=revision, tokenizer_revision=revision, dtype=dtype, seed=0,
                       max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=gpu_memory_utilization, **engine)
        self.tokenizer = self.llm.get_tokenizer()
        self.constraint = self._json_constraint(llm.output_schema())
        if self.constraint is None and not allow_unconstrained:
            raise SystemExit("this vLLM has no JSON-constrained decoding; upgrade, or pass --allow-unconstrained and record it")

    def _json_constraint(self, schema: dict) -> tuple[str, object] | None:
        try:  # vLLM >= 0.11
            from vllm.sampling_params import StructuredOutputsParams
            return "structured_outputs", StructuredOutputsParams(json=schema)
        except ImportError:
            pass
        try:  # vLLM 0.6 - 0.10
            from vllm.sampling_params import GuidedDecodingParams
            return "guided_decoding", GuidedDecodingParams(json=schema)
        except ImportError:
            return None

    def info(self) -> dict:
        template = getattr(self.tokenizer, "chat_template", None)
        out = {"engine": "vllm", "engine_version": self.vllm.__version__,
               "json_constraint": self.constraint[0] if self.constraint else None,
               "tokenizer_class": type(self.tokenizer).__name__,
               "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else None}
        if out["chat_template_sha256"] is None:
            try:
                import mistral_common
                out["chat_template"] = f"mistral_common {mistral_common.__version__} (tokenizer files pinned by the revision)"
            except ImportError:
                out["chat_template"] = "unknown"
        try:
            import torch
            out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            out["torch_version"], out["cuda_version"] = torch.__version__, torch.version.cuda
        except Exception:  # noqa: BLE001 - hardware info is best effort
            pass
        return out

    def prompt_tokens(self, req: dict) -> int | None:
        try:
            ids = self.tokenizer.apply_chat_template(messages(req), tokenize=True, add_generation_prompt=True)
            return len(ids)
        except Exception:  # noqa: BLE001 - length check is best effort; vLLM rejects overlong prompts itself
            return None

    def params(self, req: dict):
        kwargs = dict(self.sampling, max_tokens=self.max_tokens, seed=request_seed(req))
        if self.constraint:
            kwargs[self.constraint[0]] = self.constraint[1]
        return self.SamplingParams(**kwargs)

    def __call__(self, requests: list[dict]) -> list[dict]:
        results, todo = [None] * len(requests), []
        for i, req in enumerate(requests):  # protocol: test request length before sending
            n = self.prompt_tokens(req)
            if n is not None and n + self.max_tokens > self.max_model_len:
                results[i] = {"text": None, "status": "too_long", "finish_reason": None, "input_tokens": n, "output_tokens": 0}
            else:
                todo.append(i)
        if todo:
            outs = self.llm.chat([messages(requests[i]) for i in todo], sampling_params=[self.params(requests[i]) for i in todo],
                                 use_tqdm=False)
            for i, o in zip(todo, outs):
                c = o.outputs[0]
                status = "ok" if c.finish_reason == "stop" else "truncated" if c.finish_reason == "length" else str(c.finish_reason)
                results[i] = {"text": c.text, "status": status, "finish_reason": c.finish_reason,
                              "input_tokens": len(o.prompt_token_ids or []), "output_tokens": len(c.token_ids)}
        return results


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def request_index(requests):
    index = {r["custom_id"]: digest(r) for r in requests}
    if len(index) != len(requests):
        raise ValueError("Duplicate request ID in selected markets")
    return index


def output_records(path, *, resume=False):
    """A crash can interrupt the final append. Only an unterminated tail is safe to retry."""
    if not path.exists():
        return
    with path.open("rb+" if resume else "rb") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                return
            if not line.endswith(b"\n"):
                if not resume:
                    raise ValueError(f"Incomplete output record in {path}")
                stream.truncate(start)
                return
            yield json.loads(line)


def generate(requests: list[dict], generator, outputs_path: Path, model: str, batch_size: int = 256,
             log=print, *, generation_sha256=None, revision=None) -> dict:
    """Resume matching request payloads, then generate the remaining replies in length batches."""
    expected = request_index(requests)
    done = set()
    for record in output_records(outputs_path, resume=True):
        cid = record["custom_id"]
        if (cid in done or cid not in expected or record.get("request_sha256") != expected[cid]
                or record["row"].get("model") != model
                or record["row"].get("generation_sha256") != generation_sha256):
            raise ValueError(f"Unexpected, changed or duplicate saved request {cid} in {outputs_path}")
        done.add(cid)
    pending = sorted((r for r in requests if r["custom_id"] not in done), key=lambda r: -len(r["prompt"]))
    stats = {"requests": len(requests), "already_done": len(requests) - len(pending), "generated": 0, "status": {},
             "input_tokens": 0, "output_tokens": 0, "retries": 0}
    t0 = time.perf_counter()
    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pending), batch_size):
        batch = pending[start: start + batch_size]
        for attempt in range(RETRIES + 1):
            try:
                results = generator(batch)
                break
            except (RuntimeError, OSError) as e:  # transient engine or I/O failure: same payload again
                if attempt == RETRIES:
                    raise
                stats["retries"] += 1
                log(f"  batch failed ({e!r}); retry {attempt + 1}/{RETRIES}")
        if len(results) != len(batch):
            raise RuntimeError(f"generator returned {len(results)} results for {len(batch)} requests")
        with open(outputs_path, "a") as f:
            for req, res in zip(batch, results):
                row = llm.parse_response(req, res["text"], res["status"])
                cid = req["custom_id"]
                row.update(model=model, revision=revision, batch_request_id=cid,
                           prompt_hash=req["prompt_hash"], prompt_sha256=hashlib.sha256(req["prompt"].encode()).hexdigest(),
                           generation_sha256=generation_sha256)
                record = {"custom_id": cid, "request_sha256": expected[cid], "raw": res, "row": row}
                f.write(json.dumps(record) + "\n")
                stats["status"][row["status"]] = stats["status"].get(row["status"], 0) + 1
                stats["input_tokens"] += int(res.get("input_tokens") or 0)
                stats["output_tokens"] += int(res.get("output_tokens") or 0)
        stats["generated"] += len(batch)
        elapsed = time.perf_counter() - t0
        log(f"  {stats['generated']:,}/{len(pending):,} generated, {elapsed:.0f}s, {stats['generated'] / max(elapsed, 1e-9):.1f} req/s")
    stats["seconds"] = time.perf_counter() - t0
    if stats["generated"]:
        stats["requests_per_second"] = stats["generated"] / max(stats["seconds"], 1e-9)
        stats["output_tokens_per_second"] = stats["output_tokens"] / max(stats["seconds"], 1e-9)
        stats["valid_completions"] = stats["status"].get("ok", 0)
    return stats


def read_state(path):
    return json.loads(path.read_text())


def save_state(path, state):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        try:
            json.dump(state, tmp, indent=1)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def merge(workdir: Path, recs_path: Path, requested: dict) -> int:
    """Publish a complete, consistent set of shard receipts, including zero-request shards."""
    states = sorted(workdir.glob("state*.json"))
    if not states:
        raise ValueError(f"No shard receipts in {workdir}")
    first = read_state(states[0])
    if first.get("selection") != requested:
        raise ValueError("Shard selection differs from this command or config. Check arms, seeds and limit.")
    replicas = first["num_shards"]
    suffixes = [""] if replicas is None else [f"-{i:04d}" for i in range(replicas)]
    if set(states) != {workdir / f"state{tag}.json" for tag in suffixes}:
        raise ValueError(f"Incomplete or mixed set of shard receipts in {workdir}")
    paths = {workdir / f"outputs{tag}.jsonl" for tag in suffixes}
    if set(workdir.glob("outputs*.jsonl")) - paths:
        raise ValueError(f"Unexpected output shards in {workdir}")

    rows = []
    seen = set()
    generation_hash = digest(first["pinned"])
    for i, tag in enumerate(suffixes):
        state = read_state(workdir / f"state{tag}.json")
        if (state.get("phase") != "complete" or state.get("pinned") != first["pinned"]
                or state.get("selection") != requested or state.get("num_shards") != replicas
                or state.get("shard") != (None if replicas is None else i)):
            raise ValueError(f"Incomplete or inconsistent shard {tag!r} in {workdir}")
        actual = {}
        for rec in output_records(workdir / f"outputs{tag}.jsonl"):
            cid, row = rec["custom_id"], rec["row"]
            if (cid in seen or row["batch_request_id"] != cid or row["exposure"] not in requested["arms"]
                    or row["model"] != requested["model"] or row.get("generation_sha256") != generation_hash):
                raise ValueError(f"Incompatible or duplicate reply for {cid}")
            seen.add(cid)
            actual[cid] = rec["request_sha256"]
            rows.append(row)
        if len(actual) != state["requests"] or digest(actual) != state["requests_sha256"]:
            raise ValueError(f"Reply set does not match the request receipt for shard {tag!r}")
    if rows:
        save_recommendations(recs_path, rows, generation_hashes={arm: generation_hash for arm in requested["arms"]})
    return len(rows)


def check_state(state_path: Path, pinned: dict) -> None:
    """Refuse to mix settings in one work directory: the arm must stay fixed (protocol section 5)."""
    if state_path.exists():
        old = read_state(state_path).get("pinned", {})
        changed = {k: (old.get(k), v) for k, v in pinned.items() if old.get(k) != v}
        if changed:
            raise SystemExit(f"{state_path} was written with different settings {changed}. Use a new --tag, or move it.")


def _nvidia_smi() -> str | None:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: list[str] | None = None, generator_factory=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--arm", action="append", help="llm exposure name, e.g. qwen_k5 (repeatable)")
    ap.add_argument("--family", choices=sorted(ARM_MODELS), help="every llm exposure whose name starts with this prefix")
    ap.add_argument("--model", help="override the family's model id (record why)")
    ap.add_argument("--revision", help="override the family's pinned revision (a commit SHA)")
    ap.add_argument("--seeds", type=int, help="only the first N seed blocks")
    ap.add_argument("--limit", type=int, help="only the first N paired renter/budget cases (e.g. the 500-request profile)")
    ap.add_argument("--shard", type=int, help="this replica's index, e.g. $SLURM_ARRAY_TASK_ID")
    ap.add_argument("--num-shards", type=int)
    ap.add_argument("--tag", help="suffix for the work directory, e.g. profile_h200")
    ap.add_argument("--batch-size", type=int, default=256, help="requests per vLLM call (vLLM batches continuously inside)")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="1 = single-GPU replica (preferred, protocol section 8)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--allow-unconstrained", action="store_true", help="run even if JSON-constrained decoding is unavailable")
    ap.add_argument("--dry-run", action="store_true", help="build the requests and print counts; no vLLM")
    ap.add_argument("--show", action="store_true", help="with --dry-run, print one rendered request")
    ap.add_argument("--merge-only", action="store_true", help="merge finished outputs into the recs file; no generation")
    ap.add_argument("--no-merge", action="store_true", help="do not merge into the recs file (profiling runs)")
    args = ap.parse_args(argv)
    if (args.shard is None) != (args.num_shards is None) or (args.shard is not None and not 0 <= args.shard < args.num_shards):
        ap.error("--shard and --num-shards go together, with 0 <= shard < num-shards")

    if args.batch_size < 1 or (args.limit is not None and args.limit < 1) or (args.seeds is not None and args.seeds < 1):
        ap.error("--batch-size, --limit and --seeds must be positive")
    if args.model and not args.revision:
        ap.error("A model override requires its matching --revision commit SHA")
    if args.revision and (len(args.revision) != 40 or any(c not in "0123456789abcdefABCDEF" for c in args.revision)):
        ap.error("--revision must be a 40-character commit SHA")
    cfg = load_config(args.config)
    arms = resolve_arms(cfg, args.arm, args.family)
    family = family_of(arms[0])
    if family not in ARM_MODELS and not (args.model and args.revision):
        raise SystemExit(f"no default model for family {family!r}; pass --model and --revision")
    default_model, default_revision, settings = ARM_MODELS.get(family, (None, None, {"engine": {}, "sampling": {}}))
    model, revision = args.model or default_model, args.revision or default_revision
    workdir = ROOT / "results" / "llm" / cfg["name"] / (f"open_{family}" + (f"_{args.tag}" if args.tag else ""))
    tag = f"-{args.shard:04d}" if args.shard is not None else ""
    outputs_path, state_path = workdir / f"outputs{tag}.jsonl", workdir / f"state{tag}.json"
    recs_path = ROOT / cfg.get("llm_results", f"results/llm/{cfg['name']}/recs.jsonl")

    selection = {"arms": arms, "model": model, "revision": revision, "config_sha256": digest(cfg),
                 "seeds": args.seeds, "limit": args.limit, "prompt_hash": llm.prompt_hash()}
    if args.limit is not None:
        selection["profile_order"] = "sha256(market_id,renter,rep,count),policy_suffix"
    if args.merge_only:
        n = merge(workdir, recs_path, selection)
        print(f"merged {n:,} rows from {workdir} into {recs_path}")
        return None

    specs = selected_markets(cfg, args.seeds)
    if args.shard is not None and args.limit is None:
        specs = specs[args.shard::args.num_shards]
        requests = all_requests(cfg, args.seeds, arms, markets=specs) if specs else []
    else:
        requests = select(all_requests(cfg, args.seeds, arms, markets=specs), args.limit, args.shard, args.num_shards)
    by_arm = {a: sum(r["exposure"] == a for r in requests) for a in arms}
    chars = [len(r["prompt"]) + len(llm.SYSTEM_PROMPT) for r in requests]
    print(f"{cfg['name']}{tag}: {len(requests):,} requests for {model}@{revision[:12]} ({by_arm}); "
          f"prompt chars mean {sum(chars) / max(len(chars), 1):,.0f}, max {max(chars, default=0):,} "
          f"(about {max(chars, default=0) / 4:,.0f} tokens at 4 characters per token; context {args.max_model_len:,})")
    if args.dry_run:
        if args.show and requests:
            for m in messages(requests[0]):
                print(f"\n{m['role'].upper()}:\n{m['content']}")
        print(f"dry run: nothing generated. Outputs would go to {outputs_path}; recs file {recs_path}")
        return requests

    sampling = dict(settings["sampling"])
    pinned = {"model": model, "revision": revision, "tokenizer_revision": revision, "engine_settings": settings["engine"],
              "sampling": sampling, "max_tokens": args.max_tokens, "max_model_len": args.max_model_len, "dtype": args.dtype,
              "tensor_parallel_size": args.tensor_parallel_size, "seed_rule": "paired market/exposure/renter/rep ID prefix (32 bits)",
              "prompt_hash": llm.prompt_hash(), "system_prompt": llm.SYSTEM_PROMPT, "instruction": llm.INSTRUCTION,
              "schema": llm.output_schema()}
    workdir.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise SystemExit(f"Another worker owns this shard: {state_path}") from e

        previous = read_state(state_path) if state_path.exists() else {}
        receipt = {"selection": selection, "shard": args.shard, "num_shards": args.num_shards,
                   "requests": len(requests), "requests_sha256": digest(request_index(requests))}
        if previous and any(previous.get(k) != v for k, v in receipt.items()):
            raise ValueError(f"Settings, shard layout or request payloads changed in {state_path}; use a new --tag")
        if not previous and outputs_path.exists():
            raise ValueError(f"Existing outputs without a matching run receipt in {workdir}")
        check_state(state_path, pinned)
        started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        if generator_factory is None:
            gen = VllmGenerator(model, revision, settings["engine"], sampling, args.max_tokens, args.max_model_len,
                                args.tensor_parallel_size, args.gpu_memory_utilization, args.dtype, args.allow_unconstrained)
        else:
            gen = generator_factory(model=model, revision=revision, settings=settings, args=args)
        engine_info = gen.info()
        pinned_engine = {k: engine_info.get(k) for k in ("engine", "engine_version", "json_constraint",
                         "chat_template_sha256", "chat_template", "tokenizer_class", "torch_version", "cuda_version", "gpus")}
        pinned.update(pinned_engine)
        check_state(state_path, pinned)
        state = {**receipt, "pinned": pinned, "engine_info": engine_info, "config": cfg["name"], "tag": args.tag,
                 "phase": "running", "run_started_utc": started, "runs": previous.get("runs", [])}
        save_state(state_path, state)
        stats = generate(requests, gen, outputs_path, model, args.batch_size, generation_sha256=digest(pinned), revision=revision)
        state.update(phase="complete", run_ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                     nvidia_smi=_nvidia_smi(), stats=stats, python=sys.version.split()[0])
        state["runs"].append({k: state[k] for k in ("run_started_utc", "run_ended_utc", "stats", "nvidia_smi")})
        save_state(state_path, state)
    print(f"status counts {stats['status']}; state in {state_path}")
    if args.shard is not None or args.no_merge:
        print("not merged into the recs file" + (" (sharded: run --merge-only after every shard ends)" if args.shard is not None else ""))
    else:
        n = merge(workdir, recs_path, selection)
        print(f"merged {n:,} rows into {recs_path}")
    return stats


if __name__ == "__main__":
    main()
