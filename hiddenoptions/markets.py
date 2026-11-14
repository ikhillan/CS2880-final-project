"""Sampled markets following Todd's protocol: renters from PUMS, listings from FirstMover, modeled preferences and firms.

A market is one borough x one month. The same spec and seed always give the same renters, listings, preferences and firms,
so every policy and audit method is compared on identical markets.

Designs
    uniform   (primary) renters drawn by survey weight, with replacement, from the cell's donor frame; listings drawn
              uniformly without replacement from the borough-month catalog. Kept even if nobody can afford anything.
    enriched  (second panel) an affordability band: half the listings from the band, half uniformly from the listings
              above it; half the renters from lower-income donors with a feasible edge to the drawn listings, half from
              higher-income donors (both survey-weighted, with replacement). The band is every cell listing whose rent
              is at or below the `band_quantile` (default 0.75) survey-weighted quantile of the budgets (`cap`) of the
              cell's G+ donors, i.e. lower-income donors with at least one feasible listing in the whole cell.
    enriched_quartile  the earlier enriched rule, kept for comparison: the band is the cell's bottom asking-rent
              quartile. It reuses the old "enriched" random streams, so it reproduces earlier enriched markets.
An enriched market is unestimable (InsufficientSupply) if the cell has no G+ donor, the band or the remainder has too
few listings, or no lower-income donor has an edge to the drawn listings.

Renter score for listing j (protocol section 4):
    u_bj = w_q (Q_j + 0.5 xi_j) - w_p r_j / budget_b + w_s min(beds_j - minbeds_b, 2) - w_d distance_bj / 5 + sigma_r eps_bj
Listing score for renter b:
    v_jb = w_income min(I_b / (12 r_j), 40/12) + landlord_noise eta_jb
"""
from __future__ import annotations

import warnings
import zlib
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from .data import (BASE_MONTH, DEV_MONTHS, MONTHS, cell_frames, heldout_period, load_heldout_listings, load_listings, load_renters,
                   to_base_factor, weighted_quantile)
from .firms import load_alias_map
from .matching import NOT_EDGE, Instance

KM_PER_DEG_LAT, KM_PER_DEG_LON = 111.0, 84.5  # at NYC's latitude
SCREEN_RATIO = 40 / 12  # landlords stop caring about income above the common "income >= 40 x monthly rent" screen
MIN_REFERENCE_LISTINGS = 30  # below this, a borough x bedroom cell uses the citywide same-bedroom reference
CLIP = 3.0
ENRICHED_DESIGNS = ("enriched", "enriched_quartile")


class InsufficientSupply(ValueError):
    """The cell does not have enough unique listings (or donors) for the requested market."""


@dataclass(frozen=True)
class MarketSpec:
    borough: str | None  # None = citywide search (sensitivity check)
    month: str
    seed: int
    n_renters: int = 64
    n_listings: int = 64
    share: float = 0.40  # max rent as a share of income
    design: str = "uniform"  # "uniform" (primary), "enriched" (affordability band) or "enriched_quartile" (old rule)
    donors: str = "evaluation"  # "evaluation", "calibration" or "all"
    bedroom_rule: str = "standard"  # "standard" or "strict"
    rent_column: str = "rent"  # "rent" (asking, July 2026 $) or "net_effective_rent" (sensitivity)
    firm_sampling: str = "uniform"  # "uniform" (protocol) or "clustered" (draw listings a few at a time per firm)
    firm_cluster: int = 3
    renter_firm_size: int = 1  # 1 = each renter has a private firm (protocol); larger values group renters under agents
    renter_firm_mode: str = "random"  # "random" or "by_income" (only used when renter_firm_size > 1)
    quality: str = "features"  # "features" (sqft + baths, primary) or "rent_premium" (sensitivity)
    w_q: float = 1.0
    w_p: float = 4.0  # calibrated over {1, 2, 4} on 2026-11-11 (selected 4.0); see results/calibration/calibration.json
    w_s: float = 0.5
    w_d: float = 0.5
    sigma_r: float = 0.5  # calibrated over {0.5, 0.75, 1.25} on 2026-11-11 (selected 0.5); see results/calibration/calibration.json
    common_shock: float = 0.5
    w_income: float = 2.5
    landlord_noise: float = 1.0
    common_priority: bool = False  # mathematical control: every listing ranks renters by the same income order
    # Audit institutions (protocol section 7). Costs stay integers because the solver relies on integer objectives, so
    # "portfolio_quarter" (c_f = 1 + 0.25 (m_f - 1)) is stored x4; divide reported costs by `cost_scale`.
    audit_costs: str = "unit"  # "unit", "brokerage_only", "portfolio_quarter" or "portfolio_linear"
    band_quantile: float = 0.75  # enriched design: band = rent <= this weighted quantile of G+ donors' budgets
    # Listing-side firms from the frozen brokerage alias map (data/reference/brokerage_aliases.json, hiddenoptions.firms):
    # alias variants share a firm and unresolved sources become singleton firms. Off by default (firms = slug of source_group).
    firm_aliases: bool = False
    heldout_dedup: bool = True  # remove cross-period unit repeats in the prespecified held-out months

    @property
    def market_id(self) -> str:
        # Preserve pre-dedup raw market IDs and every existing development ID.
        fields = asdict(self)
        del fields["seed"], fields["heldout_dedup"]
        if self.design != "enriched":
            del fields["band_quantile"]
        if self.firm_aliases:
            fields["alias_map_sha256"] = load_alias_map().sha256
        else:
            del fields["firm_aliases"]
        if self.heldout_dedup and heldout_period(self.month):
            fields["unit_dedup"] = "dev-and-within-period-v1"
        key = "|".join(f"{k}={v}" for k, v in fields.items())
        return f"{self.borough or 'Citywide'}-{self.month}-n{self.n_renters}x{self.n_listings}-s{self.seed}-{zlib.crc32(key.encode()):08x}"


@dataclass
class Market:
    spec: MarketSpec
    inst: Instance
    renters: pd.DataFrame
    listings: pd.DataFrame
    n_firms_listing: int
    n_firms_renter: int
    u_observable: np.ndarray  # renter score without latent shocks; the transparent policy ranks on it
    distance_km: np.ndarray
    sampling_log: dict


class Catalog:
    """Data shared by all markets in one process. Listing months load on first use.

    heldout_dedup (default True) applies only to held-out months (data.HELDOUT_PRIMARY and HELDOUT_REPLICATION): it removes
    identifiable physical units already seen in development and keeps each unit's earliest arrival within its held-out
    period (data.dedup_heldout). Development and other months load unchanged. Counts per month are in `dedup_counts`.
    """

    def __init__(self, heldout_dedup: bool = True):
        self.heldout_dedup = heldout_dedup
        self.dedup_counts: dict[str, dict] = {}
        self.renters = load_renters()
        self._listings: dict[tuple, pd.DataFrame] = {}
        self._alias_firms: dict[tuple, pd.Series] = {}
        self._warned_unreviewed = False
        self.frames: dict = {}
        dev = pd.concat([self.listings(m) for m in DEV_MONTHS])
        self.feature_reference = _feature_reference(dev)
        self.rent_reference = _rent_reference(dev)
        self.puma_centroids = dev.groupby("puma")[["latitude", "longitude"]].mean()
        self.puma_label = dev.groupby("puma").neighborhood.agg(lambda s: s.value_counts().index[0]).to_dict()

    def alias_firms(self, month: str, *, heldout_dedup: bool = True) -> pd.Series:
        """Listing firm labels for the selected catalog, cached across market draws."""
        key = (month, self.heldout_dedup and heldout_dedup and heldout_period(month) is not None)
        if key not in self._alias_firms:
            amap = load_alias_map()
            if not amap.reviewed and not self._warned_unreviewed:
                warnings.warn('brokerage alias map is not reviewed yet ("reviewed": false); see scripts/brokerage_aliases.py')
                self._warned_unreviewed = True
            listings = self.listings(month, heldout_dedup=heldout_dedup)
            self._alias_firms[key] = pd.Series(amap.firm_ids(listings), index=listings.listing_uid.to_numpy())
        return self._alias_firms[key]

    def listings(self, month: str, *, heldout_dedup: bool = True) -> pd.DataFrame:
        dedup = self.heldout_dedup and heldout_dedup and heldout_period(month) is not None
        key = (month, dedup)
        if key not in self._listings:
            if month not in MONTHS:
                raise KeyError(f"no FirstMover month {month}")
            if dedup:
                self._listings[key], self.dedup_counts[month] = load_heldout_listings(month)
            else:
                self._listings[key] = load_listings(month)
        return self._listings[key]

    def frame(self, spec: MarketSpec):
        key = (spec.borough, spec.month, spec.share, spec.donors, spec.bedroom_rule, spec.rent_column, spec.heldout_dedup)
        if key not in self.frames:
            listings = self.listings(spec.month, heldout_dedup=spec.heldout_dedup)
            if spec.rent_column == "net_effective_rent":
                listings = listings.assign(net_effective_rent=listings.net_effective_rent_asking * to_base_factor(spec.month))
            self.frames[key] = cell_frames(self.renters, listings, spec.borough, spec.month, spec.share,
                                           donors=None if spec.donors == "all" else spec.donors,
                                           bedroom_rule=spec.bedroom_rule, rent_column=spec.rent_column)
        return self.frames[key]


# ------------------------------------------------------------------------------------------------- quality
def _feature_reference(dev: pd.DataFrame) -> dict:
    """Mean, SD and median of log sqft and bathrooms per (borough, bedroom bin), citywide when a cell is thin."""
    feats = pd.DataFrame({"borough": dev.borough, "bedbin": dev.bedbin, "log_sqft": np.log(dev.sqft), "baths": dev.bathrooms_total})
    ref = {}
    for f in ("log_sqft", "baths"):
        city = feats.groupby("bedbin")[f].agg(["count", "mean", "std", "median"])
        cell = feats.groupby(["borough", "bedbin"])[f].agg(["count", "mean", "std", "median"])
        ref[f] = {"city": city, "cell": cell}
    return ref


def _feature_z(values: pd.Series, borough: pd.Series, bedbin: pd.Series, ref: dict) -> np.ndarray:
    out = np.zeros(len(values))
    for i, (v, b, k) in enumerate(zip(values, borough, bedbin)):
        stats = ref["cell"].loc[(b, k)] if (b, k) in ref["cell"].index and ref["cell"].loc[(b, k), "count"] >= MIN_REFERENCE_LISTINGS \
            else (ref["city"].loc[k] if k in ref["city"].index else None)
        if stats is None or not np.isfinite(stats["std"]) or stats["std"] <= 0:
            continue  # a zero-variance or missing reference contributes zero
        x = stats["median"] if pd.isna(v) else v
        out[i] = np.clip((x - stats["mean"]) / stats["std"], -CLIP, CLIP)
    return out


def listing_quality(listings: pd.DataFrame, catalog: Catalog, kind: str) -> np.ndarray:
    if kind == "features":
        z_sqft = _feature_z(np.log(listings.sqft), listings.borough, listings.bedbin, catalog.feature_reference["log_sqft"])
        z_bath = _feature_z(listings.bathrooms_total, listings.borough, listings.bedbin, catalog.feature_reference["baths"])
        return (z_sqft + z_bath) / 2
    if kind == "rent_premium":
        means, sd = catalog.rent_reference
        expected = [means.get(key, means.get(key[1:], np.nan)) for key in zip(listings.puma, listings.borough, listings.bedbin)]
        return np.clip((np.log(listings.rent.to_numpy(float)) - np.array(expected, float)) / sd, -CLIP, CLIP)
    raise ValueError(f"unknown quality spec {kind}")


def _rent_reference(dev: pd.DataFrame) -> tuple[dict, float]:
    """Mean log rent per (PUMA, bedroom bin), falling back to (borough, bedroom bin), and the residual SD, from development months."""
    log_rent = np.log(dev.rent)
    means = {}
    by_borough = log_rent.groupby([dev.borough, dev.bedbin]).mean()
    for (b, k), v in by_borough.items():
        means[(b, k)] = v
    counts = log_rent.groupby([dev.puma, dev.borough, dev.bedbin]).agg(["count", "mean"])
    for (p, b, k), row in counts.iterrows():
        if row["count"] >= MIN_REFERENCE_LISTINGS:
            means[(p, b, k)] = row["mean"]
    expected = np.array([means.get((p, b, k), means.get((b, k))) for p, b, k in zip(dev.puma, dev.borough, dev.bedbin)], float)
    return means, float(np.nanstd(log_rent.to_numpy() - expected))


# ------------------------------------------------------------------------------------------------- sampling
def _rng(spec: MarketSpec, stream: str) -> np.random.Generator:
    """Streams ignore `share`, `design`-independent weights and preference settings, so sensitivity runs stay paired.

    enriched_quartile uses the "enriched" streams, so it reproduces markets drawn before the affordability band existed.
    """
    design = "enriched" if spec.design == "enriched_quartile" else spec.design
    key = f"{spec.borough}|{spec.month}|{spec.n_renters}|{spec.n_listings}|{design}|{stream}"
    return np.random.default_rng([spec.seed, zlib.crc32(key.encode())])


def _draw_renters(rng, frame: pd.DataFrame, k: int) -> pd.DataFrame:
    """Survey-weighted draws with replacement. Each draw is a separate simulated household."""
    if k == 0:
        return frame.iloc[[]]
    if frame.empty or frame.weight.sum() <= 0:
        raise InsufficientSupply("no donors in this frame")
    p = frame.weight.to_numpy(float) / frame.weight.sum()
    return frame.iloc[rng.choice(len(frame), size=k, replace=True, p=p)]


def _draw_listings(rng, pool: pd.DataFrame, k: int, spec: MarketSpec) -> pd.DataFrame:
    if k > len(pool):
        raise InsufficientSupply(f"asked for {k} listings from a pool of {len(pool)}")
    if spec.firm_sampling == "uniform" or k == 0:
        return pool.iloc[rng.choice(len(pool), size=k, replace=False)]
    remaining, picks = pool.copy(), []
    while k > 0:
        seed_row = remaining.iloc[int(rng.integers(len(remaining)))]
        same = remaining[remaining.firm_id == seed_row.firm_id]
        take = same.iloc[rng.choice(len(same), size=min(spec.firm_cluster, k, len(same)), replace=False)]
        picks.append(take)
        remaining = remaining.drop(take.index)
        k -= len(take)
    return pd.concat(picks)


def _affordability_cap(frame: pd.DataFrame, spec: MarketSpec) -> float:
    """The enriched band's rent cap: the weighted `band_quantile` of G+ donors' budgets (lower-income, >= 1 cell option)."""
    g_plus = frame[frame.group == "G+"]
    if g_plus.empty or g_plus.weight.sum() <= 0:
        raise InsufficientSupply("enriched design: no lower-income donor can afford any listing in the cell (cell unestimable)")
    return weighted_quantile(g_plus.cap.to_numpy(float), g_plus.weight.to_numpy(float), spec.band_quantile)


def _feasible(frame: pd.DataFrame, listings: pd.DataFrame, rent_column: str) -> np.ndarray:
    rent = listings[rent_column].to_numpy(float)[None, :]
    return (rent <= frame.cap.to_numpy(float)[:, None]) & (listings.bedrooms.to_numpy()[None, :] >= frame.need_bedrooms.to_numpy()[:, None]) \
        & (frame.income.to_numpy(float)[:, None] > 0)


def sample_people(spec: MarketSpec, catalog: Catalog) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frame, pool = catalog.frame(spec)
    r_rng, l_rng = _rng(spec, "renters"), _rng(spec, "listings")
    log = {"frame_records": len(frame), "pool_listings": len(pool)}
    if spec.heldout_dedup and catalog.heldout_dedup and spec.month in catalog.dedup_counts:  # held-out months only, so development logs are unchanged
        log["heldout_dedup"] = {k: v for k, v in catalog.dedup_counts[spec.month].items() if k.startswith(("removed", "ambiguous"))}
    if spec.design == "uniform":
        listings = _draw_listings(l_rng, pool, spec.n_listings, spec)
        renters = _draw_renters(r_rng, frame, spec.n_renters)
    elif spec.design in ENRICHED_DESIGNS:
        rent = pool[spec.rent_column]
        if spec.design == "enriched":
            band_cap = _affordability_cap(frame, spec)
            band = f"affordability band (rent <= ${band_cap:,.0f}, the {spec.band_quantile:g} quantile of G+ budgets)"
        else:
            band_cap = float(rent.quantile(0.25))
            band = f"bottom rent quartile (rent <= ${band_cap:,.0f})"
        in_band = (rent <= band_cap).to_numpy()
        n_band = round(spec.n_listings / 2)
        log.update(band_cap=band_cap, band_listings=int(in_band.sum()), above_band_listings=int((~in_band).sum()))
        if spec.design == "enriched":
            log.update(band_quantile=spec.band_quantile, g_plus_donors=int((frame.group == "G+").sum()))
        if in_band.sum() < n_band:
            raise InsufficientSupply(f"{spec.design} design: the {band} has {in_band.sum()} listings, need {n_band} (cell unestimable)")
        if (~in_band).sum() < spec.n_listings - n_band:
            raise InsufficientSupply(f"{spec.design} design: only {(~in_band).sum()} listings above the {band}, "
                                     f"need {spec.n_listings - n_band} (cell unestimable)")
        listings = pd.concat([_draw_listings(l_rng, pool[in_band], n_band, spec),
                              _draw_listings(l_rng, pool[~in_band], spec.n_listings - n_band, spec)])
        all_lower = frame[frame.lower_income]
        lower = all_lower[_feasible(all_lower, listings, spec.rent_column).any(1)]
        log.update(lower_with_edge_donors=len(lower),
                   lower_with_edge_weight_share=float(lower.weight.sum() / max(all_lower.weight.sum(), 1e-9)))
        if lower.empty:
            raise InsufficientSupply(f"{spec.design} design: no lower-income donor can afford any drawn listing (cell unestimable)")
        higher = frame[~frame.lower_income]
        n_low = round(spec.n_renters / 2)
        renters = pd.concat([_draw_renters(r_rng, lower, n_low), _draw_renters(r_rng, higher, spec.n_renters - n_low)])
    else:
        raise ValueError(f"unknown design {spec.design}")
    if spec.firm_aliases:
        # Keep even clustered draws paired. The alias rule changes the audit institution on the sampled listings.
        firm_ids = catalog.alias_firms(spec.month, heldout_dedup=spec.heldout_dedup)
        listings = listings.assign(firm_id_raw=listings.firm_id, firm_id=listings.listing_uid.map(firm_ids))
        amap = load_alias_map()
        log.update(firm_aliases=True, alias_map_reviewed=amap.reviewed, alias_map_sha256=amap.sha256)
    return renters.reset_index(drop=True), listings.reset_index(drop=True), log


# ------------------------------------------------------------------------------------------------- the market
def _to_ranks(score: np.ndarray, E: np.ndarray, axis: int) -> np.ndarray:
    """Dense 0-based ranks along `axis` (higher score = better); NOT_EDGE outside E."""
    s = np.where(E, score, -np.inf)
    order = np.argsort(-s, axis=axis, kind="stable")
    ranks = np.empty_like(order)
    shape = (-1, 1) if axis == 0 else (1, -1)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(s.shape[axis]).reshape(shape), s.shape), axis=axis)
    return np.where(E, ranks, NOT_EDGE)


def build_market(spec: MarketSpec, catalog: Catalog) -> Market:
    if not catalog.heldout_dedup and spec.heldout_dedup and heldout_period(spec.month):
        # A diagnostic raw catalog must also report the raw ID to downstream cache consumers.
        spec = replace(spec, heldout_dedup=False)
    renters, listings, log = sample_people(spec, catalog)
    cap = renters.cap.to_numpy(float)[:, None]
    income = renters.income.to_numpy(float)[:, None]
    need = renters.need_bedrooms.to_numpy()[:, None]
    rent = listings[spec.rent_column].to_numpy(float)[None, :]
    beds = listings.bedrooms.to_numpy()[None, :]
    E = (rent <= cap) & (beds >= need) & (income > 0)

    prefs = _rng(spec, "preferences")
    xi = prefs.standard_normal(rent.shape)  # one shared shock per listing
    eps = prefs.standard_normal(E.shape)  # one taste shock per pair
    eta = prefs.standard_normal(E.shape)  # one listing-side shock per pair
    tie = prefs.random(E.shape) * 1e-9  # fixed seeded strict order for exact ties
    quality = listing_quality(listings, catalog, spec.quality)[None, :]
    home = catalog.puma_centroids.reindex(renters.puma)
    lat, lon = home.latitude.to_numpy(float), home.longitude.to_numpy(float)
    km = np.hypot((lat[:, None] - listings.latitude.to_numpy()[None, :]) * KM_PER_DEG_LAT,
                  (lon[:, None] - listings.longitude.to_numpy()[None, :]) * KM_PER_DEG_LON)
    km = np.where(np.isfinite(km), km, np.nanmedian(km) if np.isfinite(km).any() else 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        budget_share = np.where(cap > 0, rent / cap, 1.0)
    u_observable = spec.w_q * quality - spec.w_p * budget_share + spec.w_s * np.minimum(beds - need, 2) - spec.w_d * km / 5
    u_renter = u_observable + spec.w_q * spec.common_shock * xi + spec.sigma_r * eps + tie

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.minimum(np.where(rent > 0, income / (12 * rent), 0.0), SCREEN_RATIO)
    if spec.common_priority:
        order = np.argsort(np.argsort(renters.income.to_numpy(), kind="stable"), kind="stable")[:, None] + 0.0
        u_landlord = np.broadcast_to(order, E.shape).copy()
    else:
        u_landlord = spec.w_income * ratio + spec.landlord_noise * eta + tie

    # Firms: real brokerages on the listing side; renter-side firms are private unless grouped as a sensitivity check.
    firm_rng = _rng(spec, "firms")
    n = spec.n_renters
    k = n if spec.renter_firm_size <= 1 else max(2, int(np.ceil(n / spec.renter_firm_size)))
    order = np.argsort(renters.income.to_numpy(), kind="stable") if spec.renter_firm_mode == "by_income" else firm_rng.permutation(n)
    firm_b = np.empty(n, dtype=np.int64)
    firm_b[order] = np.arange(n) * k // max(n, 1)
    names, firm_s = np.unique(listings.firm_id.astype(str).to_numpy(), return_inverse=True)
    renter_names = [f"renter:{i}" for i in range(k)] if k == n else [f"agent:{i}" for i in range(k)]

    inst = Instance(_to_ranks(u_renter, E, axis=1), _to_ranks(u_landlord, E, axis=0), firm_b, firm_s + k,
                    audit_cost_vector(spec.audit_costs, firm_b, firm_s, k, len(names)), renter_names + names.tolist())
    renters = renters.assign(n_feasible_market=E.sum(1))
    return Market(spec, inst, renters, listings, len(names), k, np.where(E, u_observable, -np.inf), km, log)


COST_SCALE = {"unit": 1, "brokerage_only": 1, "portfolio_quarter": 4, "portfolio_linear": 1}


def audit_cost_vector(kind: str, firm_b: np.ndarray, firm_s: np.ndarray, n_renter_firms: int, n_listing_firms: int) -> np.ndarray:
    """Integer audit costs per firm (renter-side firms first). m_f = number of clients the firm represents in this market."""
    clients = np.concatenate([np.bincount(firm_b, minlength=n_renter_firms), np.bincount(firm_s, minlength=n_listing_firms)])
    if kind == "unit":
        return np.ones(len(clients), dtype=np.int64)
    if kind == "brokerage_only":
        # Renter-side audits cost more than auditing every brokerage, which always succeeds, so no optimal plan uses them.
        return np.concatenate([np.full(n_renter_firms, n_listing_firms + 1), np.ones(n_listing_firms)]).astype(np.int64)
    if kind == "portfolio_quarter":
        return (4 + (clients - 1)).astype(np.int64)
    if kind == "portfolio_linear":
        return np.maximum(clients, 1).astype(np.int64)
    raise ValueError(f"unknown audit cost structure {kind}")


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


__all__ = ["BASE_MONTH", "Catalog", "InsufficientSupply", "Market", "MarketSpec", "build_market", "maximum_matching_size"]
