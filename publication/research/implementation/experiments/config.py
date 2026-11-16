"""Read a config file and expand it into labeled market specs."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from hiddenoptions.markets import MarketSpec


def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg.setdefault("name", Path(path).stem)
    return cfg


def expand_labeled(cfg: dict) -> list[tuple[MarketSpec, str]]:
    """Cross product of boroughs, months, sizes and seeds, once per entry in `variants`.

    A variant can override a grid key, any MarketSpec field, or `tightness` (listings per renter). A market used by several
    variants runs once, under the first label. `seed_start` in the grid offsets the seeds, so stages never share markets.
    """
    labeled: dict[MarketSpec, str] = {}
    for variant in cfg.get("variants") or [{"label": "primary"}]:
        grid = {**cfg["grid"], **{k: v for k, v in variant.items() if k in cfg["grid"]}}
        fixed = {**cfg.get("market", {}), **{k: v for k, v in variant.items() if k not in cfg["grid"] and k not in ("label", "tightness")}}
        tightness = variant.get("tightness", cfg.get("tightness", 1.0))
        start = grid.get("seed_start", 0)  # pilot and evaluation stages use disjoint seed ranges (protocol section 6)
        for borough, month, n, seed in itertools.product(grid["boroughs"], grid["months"], grid["sizes"], range(start, start + grid["seeds"])):
            spec = MarketSpec(borough=borough, month=month, seed=seed, n_renters=n, n_listings=round(n * tightness), **fixed)
            labeled.setdefault(spec, variant.get("label", "primary"))
    return list(labeled.items())


def expand_markets(cfg: dict) -> list[MarketSpec]:
    return [spec for spec, _ in expand_labeled(cfg)]
