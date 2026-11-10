"""Read a config file and expand it into labeled market specs."""
from __future__ import annotations

import itertools
import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg.setdefault("name", Path(path).stem)
    cfg.setdefault("domain", "housing")
    return cfg


def domain_modules(cfg: dict):
    """The market, exposure and LLM modules for this config's domain. run.py and llm_recommend.py use these instead of
    importing the housing modules directly."""
    if cfg["domain"] == "housing":
        from hiddenoptions import exposure, llm, markets
        return markets, exposure, llm
    if cfg["domain"] == "jobs":
        from hiddenoptions.jobs import exposure, markets
        try:
            from hiddenoptions.jobs import llm
        except ImportError:  # the jobs LLM client is written once the API route is approved
            llm = None
        return markets, exposure, llm
    raise ValueError(f"unknown domain {cfg['domain']}")


def expand_labeled(cfg: dict) -> list[tuple[object, str]]:
    """Cross product of cells, months, sizes and seeds, once per entry in `variants`.

    Housing configs key the grid on `boroughs`; jobs configs on `cells` (occupation groups). A variant can override a grid key,
    any spec field, or `tightness` (listings or postings per proposer). A market used by several variants runs once, under
    the first label.
    """
    markets, _, _ = domain_modules(cfg)
    if cfg["domain"] == "housing":
        Spec, cell_key, n_keys = markets.MarketSpec, "boroughs", ("n_renters", "n_listings")
    else:
        Spec, cell_key, n_keys = markets.JobMarketSpec, "cells", ("n_seekers", "n_postings")
    labeled: dict = {}
    for variant in cfg.get("variants") or [{"label": "primary"}]:
        grid = {**cfg["grid"], **{k: v for k, v in variant.items() if k in cfg["grid"]}}
        fixed = {**cfg.get("market", {}), **{k: v for k, v in variant.items() if k not in cfg["grid"] and k not in ("label", "tightness")}}
        tightness = variant.get("tightness", cfg.get("tightness", 1.0))
        for cell, month, n, seed in itertools.product(grid[cell_key], grid["months"], grid["sizes"], range(grid["seeds"])):
            first = {"borough": cell} if cfg["domain"] == "housing" else {"cell": cell}
            spec = Spec(**first, month=month, seed=seed, **{n_keys[0]: n, n_keys[1]: round(n * tightness)}, **fixed)
            labeled.setdefault(spec, variant.get("label", "primary"))
    return list(labeled.items())


def expand_markets(cfg: dict) -> list:
    return [spec for spec, _ in expand_labeled(cfg)]
