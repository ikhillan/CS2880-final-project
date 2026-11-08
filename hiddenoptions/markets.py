"""Sampled markets: renters from PUMS, listings from FirstMover, modeled preferences and firms.

A market is a sample from one cell (borough x month), not a slice of the city's map. The same seed gives the same renters,
listings, preferences and firms, so every exposure and audit method is compared on identical markets.
"""
from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import MONTHS, cell_frames, load_listings, load_renters
from .matching import NOT_EDGE, Instance

KM_PER_DEG_LAT, KM_PER_DEG_LON = 111.0, 84.5  # at NYC's latitude
SCREEN_RATIO = 40 / 12  # landlords stop caring about income above the common "income >= 40 x monthly rent" screen


@dataclass(frozen=True)
class MarketSpec:
    borough: str | None  # None = citywide search (sensitivity check)
    month: str
    seed: int
    n_renters: int = 24
    n_listings: int = 24
    share: float = 0.40  # max rent as a share of income
    design: str = "stratified"  # "stratified" or "natural"
    lower_share: float = 0.5  # stratified: share of renters from G+
    cheap_share: float = 0.5  # stratified: share of listings from the G+ rent band
    band_quantile: float = 0.75  # stratified: the rent band stops at this quantile of G+ rent caps
    # "clustered" keeps real firm concentration: a PUMA-month has about 2.6 listings per firm, and clustered draws give 2.5-2.8.
    # "uniform" keeps catalog shares but thins portfolios to about 1.1 listings per firm at n=24.
    firm_sampling: str = "clustered"
    firm_cluster: int = 3  # clustered: listings drawn per sampled firm
    renter_firm_size: int = 6  # renters per synthetic renter-side firm
    renter_firm_mode: str = "random"  # "random" or "by_income"
    w_quality: float = 1.0  # renter: taste for the market-wide quality index (log rent + common shock)
    common_shock: float = 0.5
    w_price: float = 2.0  # renter: disutility of rent / own max rent
    w_size: float = 0.5  # renter: value of each spare bedroom, up to 2
    w_distance: float = 0.5  # renter: disutility per 5 km from home PUMA
    renter_noise: float = 0.75
    w_income: float = 2.5  # landlord: value of income / annual rent, up to SCREEN_RATIO
    landlord_noise: float = 1.0

    @property
    def market_id(self) -> str:
        key = "|".join(f"{k}={v}" for k, v in asdict(self).items() if k != "seed")
        return f"{self.borough or 'Citywide'}-{self.month}-n{self.n_renters}x{self.n_listings}-s{self.seed}-{zlib.crc32(key.encode()):08x}"


@dataclass
class Market:
    spec: MarketSpec
    inst: Instance
    renters: pd.DataFrame
    listings: pd.DataFrame
    n_firms_listing: int
    n_firms_renter: int


class Catalog:
    """Lazily loaded data shared by all markets in one process."""

    def __init__(self):
        self.renters = load_renters()
        self.listings = {m: load_listings(m) for m in MONTHS}
        self.frames: dict = {}

    def frame(self, borough, month, share):
        key = (borough, month, share)
        if key not in self.frames:
            self.frames[key] = cell_frames(self.renters, self.listings[month], borough, month, share)
        return self.frames[key]


def _rng(spec: MarketSpec, stream: str) -> np.random.Generator:
    """Streams ignore `share` and the preference weights, so sensitivity runs stay paired with the primary run."""
    key = f"{spec.borough}|{spec.month}|{spec.n_renters}|{spec.n_listings}|{stream}"
    return np.random.default_rng([spec.seed, zlib.crc32(key.encode())])


def _weighted_sample(rng, frame: pd.DataFrame, k: int, weight: str | None = None) -> pd.DataFrame:
    """Sample PUMS records by weight. Very large markets can need more renters than the frame has records, and then records
    repeat: each simulated renter copies a Census record, and one record stands for many real households."""
    p = None if weight is None else (frame[weight].to_numpy(float) / frame[weight].sum())
    return frame.iloc[rng.choice(len(frame), size=k, replace=k > len(frame), p=p)]


def _sample_listings(rng, pool: pd.DataFrame, k: int, spec: MarketSpec) -> pd.DataFrame:
    if spec.firm_sampling == "uniform" or k == 0:
        return pool.iloc[rng.choice(len(pool), size=min(k, len(pool)), replace=False)]
    # Clustered: pick a listing at random (so firms are size-biased), then take more from the same firm.
    remaining, picks = pool.copy(), []
    while k > 0 and len(remaining):
        seed_row = remaining.iloc[int(rng.integers(len(remaining)))]
        same = remaining[remaining.firm_id == seed_row.firm_id]
        take = same.iloc[rng.choice(len(same), size=min(spec.firm_cluster, k, len(same)), replace=False)]
        picks.append(take)
        remaining = remaining.drop(take.index)
        k -= len(take)
    return pd.concat(picks)


def sample_people(spec: MarketSpec, catalog: Catalog) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, pool = catalog.frame(spec.borough, spec.month, spec.share)
    r_rng, l_rng = _rng(spec, "renters"), _rng(spec, "listings")
    if spec.design == "natural":
        renters = _weighted_sample(r_rng, frame, spec.n_renters, "weight")
        listings = _sample_listings(l_rng, pool, spec.n_listings, spec)
    elif spec.design == "stratified":
        n_low = round(spec.lower_share * spec.n_renters)
        low = frame[frame.group == "G+"]
        renters = pd.concat([_weighted_sample(r_rng, low, n_low, "weight"),
                             _weighted_sample(r_rng, frame[frame.group == "higher"], spec.n_renters - n_low, "weight")])
        band = float(low.cap_month.quantile(spec.band_quantile))
        n_cheap = round(spec.cheap_share * spec.n_listings)
        cheap_pool = pool[pool.rent <= band]
        cheap = _sample_listings(l_rng, cheap_pool, n_cheap, spec)
        rest = _sample_listings(l_rng, pool.drop(cheap.index), spec.n_listings - len(cheap), spec)
        listings = pd.concat([cheap, rest])
    else:
        raise ValueError(f"unknown design {spec.design}")
    return renters.reset_index(drop=True), listings.reset_index(drop=True)


def _home_points(catalog: Catalog, renters: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Renter location: centroid of listings in the renter's PUMA (PUMS gives no finer geography)."""
    if not hasattr(catalog, "puma_centroids"):
        pooled = pd.concat(catalog.listings.values())
        catalog.puma_centroids = pooled.groupby("puma")[["latitude", "longitude"]].mean()
    c = catalog.puma_centroids.reindex(renters.puma)
    return c.latitude.to_numpy(float), c.longitude.to_numpy(float)


def _to_ranks(score: np.ndarray, E: np.ndarray, axis: int) -> np.ndarray:
    """Dense 0-based ranks along `axis` (higher score = better); NOT_EDGE outside E."""
    s = np.where(E, score, -np.inf)
    order = np.argsort(-s, axis=axis, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(s.shape[axis]).reshape((-1, 1) if axis == 0 else (1, -1)), s.shape), axis=axis)
    return np.where(E, ranks, NOT_EDGE)


def build_market(spec: MarketSpec, catalog: Catalog) -> Market:
    renters, listings = sample_people(spec, catalog)
    cap = renters.cap_month.to_numpy(float)[:, None]
    income = renters.income_month.to_numpy(float)[:, None]
    need = renters.need_bedrooms.to_numpy()[:, None]
    rent = listings.rent.to_numpy(float)[None, :]
    beds = listings.bedrooms.to_numpy()[None, :]
    E = (rent <= cap) & (beds >= need) & (income > 0)

    # Renter side: everyone values the same quality index; rent stings more the closer it is to one's own limit.
    prefs = _rng(spec, "preferences")
    log_rent = np.log(rent)
    quality = (log_rent - log_rent.mean()) / (log_rent.std() + 1e-9) + spec.common_shock * prefs.standard_normal(rent.shape)
    lat, lon = _home_points(catalog, renters)
    km = np.hypot((lat[:, None] - listings.latitude.to_numpy()[None, :]) * KM_PER_DEG_LAT,
                  (lon[:, None] - listings.longitude.to_numpy()[None, :]) * KM_PER_DEG_LON)
    km = np.nan_to_num(km, nan=float(np.nanmean(km)) if np.isfinite(km).any() else 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        burden = np.where(cap > 0, rent / cap, 1.0)
    u_renter = (spec.w_quality * quality - spec.w_price * burden + spec.w_size * np.clip(beds - need, 0, 2)
                - spec.w_distance * km / 5 + spec.renter_noise * prefs.standard_normal(E.shape))

    # Listing side: income matters only up to the usual screen. Above it, landlords differ idiosyncratically.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.minimum(income / (12 * rent), SCREEN_RATIO)
    u_landlord = spec.w_income * ratio + spec.landlord_noise * prefs.standard_normal(E.shape)

    # Firms: real brokerages on the listing side, synthetic agents on the renter side (FirstMover has no renter-side firms).
    firm_rng = _rng(spec, "firms")
    k = max(2, int(np.ceil(spec.n_renters / spec.renter_firm_size)))
    if spec.renter_firm_mode == "by_income":
        order = np.argsort(renters.income_month.to_numpy(), kind="stable")
    else:
        order = firm_rng.permutation(spec.n_renters)
    firm_b = np.empty(spec.n_renters, dtype=np.int64)
    firm_b[order] = np.arange(spec.n_renters) * k // spec.n_renters
    names, firm_s = np.unique(listings.firm_id.astype(str).to_numpy(), return_inverse=True)
    firm_names = [f"agent:{i}" for i in range(k)] + names.tolist()

    inst = Instance(_to_ranks(u_renter, E, axis=1), _to_ranks(u_landlord, E, axis=0), firm_b, firm_s + k,
                    np.ones(len(firm_names), dtype=np.int64), firm_names)
    renters = renters.assign(n_feasible_market=E.sum(1))
    return Market(spec, inst, renters, listings, len(names), k)


def maximum_matching_size(E: np.ndarray) -> int:
    """Kuhn's algorithm: the most renters that could be housed under E, whatever the preferences."""
    match_s = [-1] * E.shape[1]
    adj = [np.nonzero(row)[0].tolist() for row in E]

    def try_renter(b, seen):
        for s in adj[b]:
            if s not in seen:
                seen.add(s)
                if match_s[s] == -1 or try_renter(match_s[s], seen):
                    match_s[s] = b
                    return True
        return False

    return sum(try_renter(b, set()) for b in range(E.shape[0]))
