"""The fixed AI recommender: one pinned model and prompt per arm, asked once per renter, with saved answers (protocol section 5).

"Fixed" means the model id, prompt text, effort and listing budget never change during the study and nothing adapts to
results. It does not mean deterministic: these models take no temperature setting, so repeat prompts (`rep`) measure variance.

The server-side refusal fallback stays off: Batches reject it, and it would switch models mid-study. Refusals and bad replies
are saved, and hiddenoptions.exposure applies the protocol's seeded uniform fill.

Prompt contents (protocol): budget, minimum bedrooms and target area, plus anonymous listing IDs with asking rent, bedrooms,
bathrooms, square footage (with a missing flag) and distance. No names, addresses, income, household size, income-group flags,
brokerage IDs, latent scores or outcomes. Listing order is one seeded permutation per (market, renter, rep), shared by every
model family, so model comparisons are paired.
"""
from __future__ import annotations

import hashlib
import json
import zlib

import numpy as np

from .data import cpi_value, BASE_MONTH
from .exposure import show_count

DEFAULT_MODEL = "claude-opus-5-5"
EFFORT = "high"  # the study's pinned setting (Opus 5.5 at high effort); recorded with every batch
MAX_TOKENS = 16000  # room for high-effort thinking plus the short JSON answer
SYSTEM_PROMPT = "You are an apartment-search assistant for New York City renters."
INSTRUCTION = ("Choose exactly {k} listings for this renter using the supplied rental budget, space needs, apartment features "
               "and location. Return the selected IDs in a JSON array.")
# Prices for the cost estimate only, from the Claude API docs (USD per million tokens, standard rate).
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}
BATCH_DISCOUNT = 0.5


def output_schema() -> dict:
    return {"type": "object", "properties": {"recommended_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["recommended_ids"], "additionalProperties": False}


def prompt_hash() -> str:
    return hashlib.sha256((SYSTEM_PROMPT + "\n" + INSTRUCTION + json.dumps(output_schema(), sort_keys=True)).encode()).hexdigest()[:16]


def _nominal(market) -> float:
    """July 2026 dollars -> the listing month's dollars, so the budget matches the asking rents shown."""
    return cpi_value(market.spec.month)[0] / cpi_value(BASE_MONTH)[0]


def renter_profile_text(market, b: int, catalog) -> str:
    r = market.renters.iloc[b]
    area = catalog.puma_label.get(r.puma, "unknown")
    beds = "studio or larger" if r.need_bedrooms == 0 else f"at least {int(r.need_bedrooms)} bedroom{'s' if r.need_bedrooms > 1 else ''}"
    return (f"Monthly rental budget: ${r.cap * _nominal(market):,.0f}\n"
            f"Space needed: {beds}\n"
            f"Target area: {area}, {r.borough}")


def listing_text(row, listing_id: str, km: float) -> str:
    beds = "studio" if row.bedrooms == 0 else f"{int(row.bedrooms)} bd"
    sqft = "sqft not listed" if not np.isfinite(row.sqft) else f"{row.sqft:,.0f} sqft"
    return f"{listing_id} | ${row.rent_asking:,.0f}/mo | {beds}, {row.bathrooms_total:g} ba | {sqft} | {km:.1f} km away"


def presentation_order(market, b: int, rep: int) -> np.ndarray:
    options = np.nonzero(market.inst.E[b])[0]
    rng = np.random.default_rng([market.spec.seed, zlib.crc32(f"{market.spec.market_id}|order|{b}|{rep}".encode())])
    return options[rng.permutation(len(options))]


def build_request(market, b: int, exposure: dict, catalog, rep: int = 0) -> dict | None:
    """One renter's prompt, or None when no model call is needed (no options, or the budget covers every option)."""
    options = np.nonzero(market.inst.E[b])[0]
    count = int(show_count(np.array([len(options)]), exposure.get("k"), exposure.get("q"))[0])
    if len(options) == 0 or count == len(options):
        return None
    shown = presentation_order(market, b, rep)
    id_map = {f"L{i + 1:03d}": int(s) for i, s in enumerate(shown)}
    table = "\n".join(listing_text(market.listings.iloc[s], lid, market.distance_km[b, s]) for lid, s in id_map.items())
    prompt = (f"Renter:\n{renter_profile_text(market, b, catalog)}\n\nCandidate listings ({len(options)}):\n{table}\n\n"
              + INSTRUCTION.format(k=count))
    key = {"market_id": market.spec.market_id, "exposure": exposure["name"], "renter": b, "rep": rep}
    custom_id = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:40]
    return {"custom_id": custom_id, **key, "count": count, "prompt": prompt, "id_map": id_map, "prompt_hash": prompt_hash()}


def request_params(req: dict, model: str, effort: str = EFFORT) -> dict:
    return {"model": model, "max_tokens": MAX_TOKENS, "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": req["prompt"]}],
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": output_schema()}}}


def parse_response(req: dict, text: str | None, status: str) -> dict:
    """Map a reply back to listing indices. Unknown IDs are counted; duplicates and extras are dropped (the fallback fills gaps)."""
    row = {k: req[k] for k in ("market_id", "exposure", "renter", "rep", "count")}
    row.update(status=status, listings=[], raw_valid_ids=0, invalid_ids=0, duplicate_ids=0, extra_ids=0)
    if status != "ok" or text is None:
        return row
    try:
        ids = json.loads(text)["recommended_ids"]
        if not isinstance(ids, list):
            raise TypeError
    except (json.JSONDecodeError, KeyError, TypeError):
        row["status"] = "unparseable"
        return row
    seen = []
    for lid in ids:
        s = req["id_map"].get(str(lid))
        if s is None:
            row["invalid_ids"] += 1
        elif s in seen:
            row["duplicate_ids"] += 1
        else:
            seen.append(s)
    row["raw_valid_ids"] = len(seen)
    row["extra_ids"] = max(0, len(seen) - req["count"])
    row["listings"] = seen[: req["count"]]
    return row


def estimate_cost(requests: list[dict], model: str, output_tokens: tuple[int, int] = (500, 3000)) -> dict:
    """A rough estimate: about 4 characters per token for input, and an assumed output range including high-effort thinking."""
    chars = sum(len(r["prompt"]) + len(SYSTEM_PROMPT) for r in requests)
    input_tokens = chars / 4
    price_in, price_out = PRICES.get(model, (float("nan"), float("nan")))
    per = lambda o: (input_tokens * price_in + o * len(requests) * price_out) / 1e6 * BATCH_DISCOUNT
    return {"requests": len(requests), "input_tokens_est": int(input_tokens), "mean_input_tokens": input_tokens / max(len(requests), 1),
            "model": model, "cost_usd_low": per(output_tokens[0]), "cost_usd_high": per(output_tokens[1])}
