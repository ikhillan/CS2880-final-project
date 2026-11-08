"""The fixed AI recommender: one pinned model and prompt, asked once per renter, with saved answers.

"Fixed" means the model id, prompt text, effort and listing count never change during the study and nothing adapts to results.
It does not mean deterministic: these models take no temperature setting, so repeat prompts (`rep`) measure the variance.

The server-side refusal fallback is deliberately off. Batches reject it, and it would switch models mid-study, so the
recommender would no longer be fixed. Refusals are saved and reported instead.

Listings get anonymous IDs in a shuffled order, so IDs and order can't leak firm or rank information.
Race and ethnicity are never included. The renter's rank list is never shown, because that list is our model, not data.
"""
from __future__ import annotations

import hashlib
import json
import zlib

import numpy as np

from .exposure import show_count
from .markets import Market

DEFAULT_MODEL = "claude-opus-5-5"
EFFORT = "low"
MAX_TOKENS = 4000
SYSTEM_PROMPT = (
    "You are an apartment-search assistant for New York City renters. You receive one renter's profile and the listings "
    "available to them. Recommend the listings this renter should look at, best first. Use only IDs from the list."
)
BUILDING_LABELS = {"RENTAL": "rental building", "CONDO": "condo", "CO_OP": "co-op", "CONDOP": "condop", "TWOFAMILY": "two-family house",
                   "THREEFAMILY": "three-family house", "FOURFAMILY": "four-family house", "HOUSE": "house", "MIXED_USE": "mixed-use building",
                   "MULTIFAMILY": "multifamily building", "HYBRID": "hybrid building"}
# Prices for the cost estimate only, from the Claude API docs (USD per million tokens, standard rate).
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}
BATCH_DISCOUNT = 0.5


def output_schema() -> dict:
    return {"type": "object", "properties": {"recommended_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["recommended_ids"], "additionalProperties": False}


def _neighborhood_by_puma(catalog) -> dict:
    if not hasattr(catalog, "puma_label"):
        import pandas as pd
        pooled = pd.concat(catalog.listings.values())
        catalog.puma_label = pooled.groupby("puma").neighborhood.agg(lambda s: s.value_counts().index[0]).to_dict()
    return catalog.puma_label


def renter_profile_text(market: Market, b: int, catalog) -> str:
    r = market.renters.iloc[b]
    size = int(r.household_size)
    children = "with children" if r.has_children else "no children"
    area = _neighborhood_by_puma(catalog).get(r.puma, "unknown")
    return (f"Household: {size} {'person' if size == 1 else 'people'}, {children}\n"
            f"Annual household income: ${r.income_month:,.0f}\n"
            f"Most they can spend on rent: ${r.cap_month:,.0f} a month\n"
            f"Minimum bedrooms: {'studio' if r.need_bedrooms == 0 else int(r.need_bedrooms)}\n"
            f"Searching in: {r.borough}, currently near {area}")


def listing_text(row, listing_id: str) -> str:
    beds = "studio" if row.bedrooms == 0 else f"{int(row.bedrooms)} bd"
    return (f"{listing_id} | ${row.rent:,.0f}/mo | {beds}, {row.bathrooms_total:g} ba | {row.neighborhood} | "
            f"{BUILDING_LABELS.get(row.building_type, 'apartment')}")


def build_request(market: Market, b: int, exposure: dict, catalog, rep: int = 0) -> dict | None:
    """One renter's prompt, or None when the renter has nothing to be shown."""
    E = market.inst.E
    options = np.nonzero(E[b])[0]
    if len(options) == 0:
        return None
    count = int(show_count(np.array([len(options)]), exposure.get("k"), exposure.get("q"))[0])
    salt = f"{market.spec.market_id}|{exposure['name']}|{b}|{rep}"
    rng = np.random.default_rng([market.spec.seed, zlib.crc32(salt.encode())])
    shuffled = options[rng.permutation(len(options))]
    id_map = {f"L{i + 1:03d}": int(s) for i, s in enumerate(shuffled)}
    table = "\n".join(listing_text(market.listings.iloc[s], lid) for lid, s in id_map.items())
    noun = "listing" if count == 1 else "listings"
    prompt = (f"Renter profile:\n{renter_profile_text(market, b, catalog)}\n\nAvailable listings ({len(options)}):\n{table}\n\n"
              f"Recommend exactly {count} {noun} for this renter, best first.")
    key = {"market_id": market.spec.market_id, "exposure": exposure["name"], "renter": b, "rep": rep}
    custom_id = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:40]
    return {"custom_id": custom_id, **key, "count": count, "prompt": prompt, "id_map": id_map}


def request_params(req: dict, model: str) -> dict:
    return {"model": model, "max_tokens": MAX_TOKENS, "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": req["prompt"]}],
            "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": output_schema()}}}


def parse_response(req: dict, text: str | None, status: str) -> dict:
    """Map model output back to listing indices. Unknown or repeated IDs are dropped and counted."""
    row = {k: req[k] for k in ("market_id", "exposure", "renter", "rep", "count")}
    row.update(status=status, listings=[], invalid_ids=0, extra_ids=0)
    if status != "ok" or text is None:
        return row
    try:
        ids = json.loads(text)["recommended_ids"]
    except (json.JSONDecodeError, KeyError, TypeError):
        row["status"] = "unparseable"
        return row
    seen = []
    for lid in ids:
        s = req["id_map"].get(lid)
        if s is None:
            row["invalid_ids"] += 1
        elif s not in seen:
            seen.append(s)
    row["extra_ids"] = max(0, len(seen) - req["count"])
    row["listings"] = seen[: req["count"]]  # the study holds the number shown fixed
    return row


def estimate_cost(requests: list[dict], model: str, output_tokens: tuple[int, int] = (150, 800)) -> dict:
    """A rough estimate: about 4 characters per token for input, and an assumed range for output including thinking."""
    chars = sum(len(r["prompt"]) + len(SYSTEM_PROMPT) for r in requests)
    input_tokens = chars / 4
    price_in, price_out = PRICES.get(model, (float("nan"), float("nan")))
    per = lambda o: (input_tokens * price_in + o * len(requests) * price_out) / 1e6 * BATCH_DISCOUNT
    return {"requests": len(requests), "input_tokens_est": int(input_tokens), "mean_input_tokens": input_tokens / max(len(requests), 1),
            "model": model, "cost_usd_low": per(output_tokens[0]), "cost_usd_high": per(output_tokens[1])}
