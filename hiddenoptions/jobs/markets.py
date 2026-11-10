"""Sampled job markets: seekers from ACS person PUMS, postings from Revelio, modeled preferences and firms.

Mirrors hiddenoptions/markets.py. A market is a sample from one cell (occupation group x month). The same seed gives the
same seekers, postings, preferences and firms. The audit layer sees only an Instance, so audit.py and matching.py are unchanged.

Capacity: a posting with q openings becomes q slot columns with identical employer rankings; a seeker ranks the slots of one
posting consecutively. Stable matchings of the many-to-one market correspond to those of this one-to-one market, and DA runs the
same way. Every slot shares the posting's firm, so firm audits expose all slots together.

Placeholders: every Revelio and PUMS column name is read from COLUMNS below. Fill them from the WRDS data dictionary and the
manifest, not from memory.
"""
from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field

from pathlib import Path

import numpy as np
import pandas as pd

from hiddenoptions.matching import NOT_EDGE, Instance

ROOT = Path(__file__).resolve().parents[2]  # repo root, as in hiddenoptions/data.py

# ---- column names: fill from the data dictionary / manifest -------------------------------------------------------------
COLUMNS = {
    # postings: revelio_job_postings.postings_cosmos, the 2025 NY-metro extract in revelio_2025_manifest.json
    # (already filtered to country, metro, post_date in 2025, salary_predicted == false, onet_code not null)
    "posting_id": "job_id",
    "post_date": "post_date",
    "employer_id": "rcid",
    "employer_name": "company",
    "occupation": "onet_code",        # O*NET code; truncated to 6-digit SOC below
    "salary": "salary",
    "salary_min": "salary_min",
    "salary_max": "salary_max",
    "seniority": "seniority",          # mapped to an ordinal 1..7 in load_postings
    "remote_type": "remote_type",
    "expected_hires": "expected_hires",  # predicted; capacity sensitivity only
    # seekers (ACS 2020-2024 person PUMS, metro PUMAs)
    "serialno": "SERIALNO",
    "sporder": "SPORDER",
    "weight": "PWGTP",
    "age": "AGEP",
    "esr": "ESR",
    "wage": "WAGP",
    "adjinc": "ADJINC",
    "soc": "SOCP",
    "schl": "SCHL",
    "puma": "PUMA",
}
POSTINGS_FILE = str(ROOT / "data" / "job_postings" / "raw" / "revelio_job_postings.csv")  # local_file in revelio_2025_manifest.json; git-ignored
# From jobs_manifest.json month_split. Load only what a run needs: a Catalog built for the development months must not
# touch the frozen or held-out months.
MONTHS_DEVELOPMENT = ("2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07")
MONTHS_FROZEN = ("2025-08", "2025-09", "2025-10")
MONTHS_HELD_OUT = ("2025-11", "2025-12")
# remote_type values in the 2025 extract: Fully In Office 2,954,475; Fully Remote 283,853; Partially In Office 32,931;
# Remote In Territory 556; Temporarily Remote 140. Keep the two local ones; the rest are remote and leave the local market.
REMOTE_TYPES_KEPT = ("Fully In Office", "Partially In Office")
RESERVATION_SHARE = 0.8  # feasibility (iii): salary midpoint >= 0.8 * prior wage
MAX_LEVEL_GAP = 2  # feasibility (iv): posting seniority <= seeker experience + 2 on the 1..7 scale (one-sided)
CAPACITY_CAP = 5
HOURS_PER_YEAR = 2080


@dataclass(frozen=True)
class JobMarketSpec:
    cell: str | None  # SOC prefix of any length: "29" = a 2-digit group, "2911" = a 4-digit minor group, "291141" = one occupation. None = all
    month: str
    seed: int
    n_seekers: int = 24
    n_postings: int = 24
    share: float = RESERVATION_SHARE  # salary floor as a share of prior wage
    level_gap: int = MAX_LEVEL_GAP  # a posting more than this many levels ABOVE the seeker is infeasible; large value = off
    design: str = "stratified"  # "stratified" or "natural"
    lower_share: float = 0.5
    band_share: float = 0.5  # stratified: share of postings drawn from the salary band below
    posting_band: str = "top"  # "top" = top salary quartile of the cell (primary), "bottom" = bottom quartile (the housing
    # mirror), "none" = no posting oversample. The oversample should favor the group the feasibility cap binds: in housing the
    # rent cap binds lower-income renters, so housing oversamples cheap listings; here the pay floor binds higher-wage seekers.
    capacity: str = "one"  # "one" (primary) or "expected_hires" (sensitivity, capped)
    firm_sampling: str = "clustered"
    firm_cluster: int = 3
    seeker_firm_size: int = 6
    seeker_firm_mode: str = "random"
    w_wage: float = 2.0  # seeker: value of log(salary / prior wage); calibrated
    w_quality: float = 1.0  # seeker: taste for the posting-quality index
    common_shock: float = 0.5
    w_fit: float = 0.5  # seeker: penalty per level of seniority mismatch, capped at 2
    seeker_noise: float = 0.75  # calibrated
    w_payband: float = 1.0  # employer: prior wage inside [0.8 min, 1.2 max]; sweep {0, 1, 2.5}
    w_match: float = 0.5  # employer: experience at or above seniority
    employer_noise: float = 1.0

    @property
    def market_id(self) -> str:
        key = "|".join(f"{k}={v}" for k, v in asdict(self).items() if k != "seed")
        return f"{self.cell or 'Metro'}-{self.month}-n{self.n_seekers}x{self.n_postings}-s{self.seed}-{zlib.crc32(key.encode()):08x}"


@dataclass
class JobMarket:
    """Same shape as hiddenoptions.markets.Market, plus what metrics/exposure need to stop hard-coding housing names."""
    spec: JobMarketSpec
    inst: Instance
    renters: pd.DataFrame  # seekers; keeps the housing attribute name so run.py/metrics.py work unchanged
    listings: pd.DataFrame  # one row per SLOT; `posting_row` maps a slot to its posting
    n_firms_listing: int
    n_firms_renter: int
    price_col: str = "salary"
    groups: dict = field(default_factory=dict)
    slot_to_posting: np.ndarray | None = None


# ---- data ------------------------------------------------------------------------------------------------------------------
def soc6(onet: pd.Series) -> pd.Series:
    """O*NET '15-1252.00' -> '151252', the six-digit form PUMS SOCP uses (no hyphen)."""
    return onet.astype(str).str.replace("-", "", regex=False).str.slice(0, 6)


def soc_in_cell(codes: pd.Series, cell: str) -> pd.Series:
    """True where a SOC code belongs to the cell prefix, treating 'X' in the code as a wildcard: PUMS aggregates some
    occupations (truck drivers are '5330XX'), and such a seeker belongs to every 6-digit cell under '5330'."""
    c = _pums_wild(codes.astype(str))
    ok = pd.Series(True, index=codes.index)
    for i, ch in enumerate(cell[:6]):
        ok &= (c.str[i] == "X") | (c.str[i] == ch)
    return ok


def _pums_wild(codes: pd.Series) -> pd.Series:
    """Normalize PUMS SOCP aggregation to 'X' wildcards. PUMS uses 'X' ('5330XX', '29112X') and, for broad occupations, a
    trailing 0 ('533030' = driver/sales workers and truck drivers, '412010' = cashiers). Detailed SOC codes never end in 0,
    so a sixth-digit 0 is a wildcard."""
    c = codes.str.strip().str.ljust(6, "X")
    return c.where(c.str[5] != "0", c.str[:5] + "X")


def soc_match(seeker_soc: np.ndarray, posting_soc: np.ndarray) -> np.ndarray:
    """PUMS SOCP is six characters and may end in 'X' for aggregated occupations (e.g. '1191XX'). A seeker matches a posting
    when every non-X character of the seeker's code equals the posting's code at that position."""
    s = np.array([list(c) for c in _pums_wild(pd.Series(seeker_soc))], dtype="<U1").reshape(-1, 6)  # n_b x 6
    p = np.array([list(c.ljust(6, "?")[:6]) for c in posting_soc], dtype="<U1").reshape(-1, 6)  # n_s x 6
    ok = np.ones((len(s), len(p)), bool)
    for i in range(6):
        ok &= (s[:, i][:, None] == "X") | (s[:, i][:, None] == p[:, i][None, :])
    return ok


_POSTINGS_CACHE: dict = {}


def load_postings(month: str) -> pd.DataFrame:
    """One month's postings (by post_date) with salary midpoint, SOC, seniority ordinal and capacity."""
    c = COLUMNS
    if "all" not in _POSTINGS_CACHE:
        if POSTINGS_FILE.endswith(".parquet"):
            _POSTINGS_CACHE["all"] = pd.read_parquet(POSTINGS_FILE)
        else:  # the 3.27M-row CSV takes a while to read; convert once to Parquet if this becomes a bottleneck
            _POSTINGS_CACHE["all"] = pd.read_csv(POSTINGS_FILE, usecols=[c[k] for k in ("posting_id", "post_date", "employer_id", "employer_name",
                                                 "occupation", "salary", "salary_min", "salary_max", "seniority", "remote_type", "expected_hires")], low_memory=False)
    d = _POSTINGS_CACHE["all"]
    d = d[pd.to_datetime(d[c["post_date"]]).dt.strftime("%Y-%m") == month]
    d = d[d[c["remote_type"]].isin(REMOTE_TYPES_KEPT)]
    # Revelio's `salary` is its midpoint estimate; recompute from the posted range so the number is the posting's, not the model's.
    mid = (d[c["salary_min"]].astype(float) + d[c["salary_max"]].astype(float)) / 2
    mid = mid.where(mid.notna(), d[c["salary"]].astype(float))
    # TODO check whether any salaries are hourly in this extract; if so annualize with HOURS_PER_YEAR and flag the rows
    out = pd.DataFrame({
        "posting_id": d[c["posting_id"]].astype(str), "firm_id": d[c["employer_id"]].astype(str), "employer": d[c["employer_name"]],
        "soc": soc6(d[c["occupation"]]), "salary": mid, "salary_min": d[c["salary_min"]].astype(float), "salary_max": d[c["salary_max"]].astype(float),
        "seniority": pd.to_numeric(d[c["seniority"]], errors="coerce").map(SENIORITY_ORDINAL).fillna(3).astype(int),
        "expected_hires": pd.to_numeric(d.get(c["expected_hires"]), errors="coerce").fillna(1.0),
        "month": month,
    })
    out["cell"] = out.soc.str.slice(0, 2)
    return out[out.salary > 0].reset_index(drop=True)


# Revelio seniority is already an integer 1 (entry) .. 7 (executive); counts in the extract run from 1.41M at level 1 to 5,397 at 7.
SENIORITY_ORDINAL = {i: i for i in range(1, 8)}


def load_seekers() -> pd.DataFrame:
    """ACS person records in NYC: age 22-64, in the labor force, positive wages, SOC present. Wages in 2024 dollars."""
    c = COLUMNS
    # From scripts/fetch_census_pums.py: the 55 NYC PUMAs (five boroughs). Postings span the NY metro; seekers do not.
    p = pd.read_csv(ROOT / "census_pums" / "acs_pums_2020_2024_nyc_persons.csv", dtype=str, usecols=[c[k] for k in
                    ("serialno", "sporder", "weight", "age", "esr", "wage", "adjinc", "soc", "schl", "puma")])
    for k in ("age", "esr", "wage", "adjinc", "weight", "schl"):
        p[c[k]] = pd.to_numeric(p[c[k]], errors="coerce")
    p = p[(p[c["age"]].between(22, 64)) & (p[c["esr"]].isin([1, 2, 3])) & (p[c["wage"]] > 0) & p[c["soc"]].notna()]
    wage = p[c["wage"]] * p[c["adjinc"]] / 1e6
    out = pd.DataFrame({
        "serialno": p[c["serialno"]], "sporder": p[c["sporder"]], "weight": p[c["weight"]].astype(float), "puma": p[c["puma"]],
        "wage_2024": wage, "soc": p[c["soc"]].astype(str).str.strip().str.slice(0, 6), "schl": p[c["schl"]].astype(int),  # kept only for the bachelor's-or-not sensitivity grouping
        "experience": np.clip((p[c["age"]] - 22) // 6, 0, 6).astype(int),  # proxy on the seniority scale; modeled
    })
    out["cell"] = out.soc.str.slice(0, 2)
    cut = weighted_median(out.wage_2024.to_numpy(), out.weight.to_numpy())
    out["lower_wage"] = out.wage_2024 <= cut
    out["lower_income"] = out.lower_wage  # alias so metrics.py works unchanged; drop once metrics reads market.groups
    return out.reset_index(drop=True)


def weighted_median(x, w) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return float(x[o][min(np.searchsorted(cw, 0.5), len(x) - 1)])


CPI_CSV = ROOT / "data" / "reference" / "cpi_u_us_city_average.csv"  # same file as the housing side; covers 2024 M13 through 2026-09


def cpi_factor(month: str) -> float:
    """Multiply 2024 dollars (ACS WAGP * ADJINC) by this to get dollars of `month` (YYYY-MM), the posting month.
    Identical to hiddenoptions.data.cpi_factor, so seeker wages and posted salaries are compared in the same month's dollars."""
    t = pd.read_csv(CPI_CSV, dtype={"period": str})
    base = t.loc[(t.year == 2024) & (t.period == "M13"), "value"].iloc[0]
    now = t.loc[(t.year == int(month[:4])) & (t.period == f"M{int(month[5:]):02d}"), "value"].iloc[0]
    return float(now) / float(base)


def feasible(seekers: pd.DataFrame, postings: pd.DataFrame, share: float, level_gap: int = MAX_LEVEL_GAP) -> np.ndarray:
    """E: same 6-digit SOC, salary midpoint >= share * prior wage, and posting seniority at most level_gap above the seeker's
    experience. The level condition is one-sided: an experienced nurse can take a staff-nurse job, a new graduate cannot take a
    level-6 one. A symmetric version was tried first and left 47% of higher-wage (older) seekers with no feasible posting,
    because the experience proxy is age-based and most postings are level 1-2. (No education field in the extract.)"""
    same_occ = soc_match(seekers.soc.to_numpy(str), postings.soc.to_numpy(str))
    pay = postings.salary.to_numpy(float)[None, :] >= share * seekers.wage_month.to_numpy(float)[:, None]
    level = (postings.seniority.to_numpy()[None, :] - seekers.experience.to_numpy()[:, None]) <= level_gap
    return same_occ & pay & level


class JobCatalog:
    def __init__(self, months: tuple[str, ...] = MONTHS_DEVELOPMENT):
        self.seekers = load_seekers()
        self.postings = {m: load_postings(m) for m in months}
        self.frames: dict = {}

    def frame(self, cell, month, share, level_gap=MAX_LEVEL_GAP):
        key = (cell, month, share, level_gap)
        if key not in self.frames:
            # A cell is a SOC prefix. Six-digit cells (one occupation) give dense feasibility; 2-digit groups are too sparse
            # for 24x24 markets because feasibility needs a 6-digit match.
            s = self.seekers if cell is None else self.seekers[soc_in_cell(self.seekers.soc, cell)]
            p = self.postings[month] if cell is None else self.postings[month][self.postings[month].soc.str.startswith(cell)]
            if len(s) == 0 or len(p) == 0:
                raise ValueError(f"cell {cell!r} month {month}: {len(s)} seekers and {len(p)} postings; pick another cell or month")
            s = s.copy()
            s["wage_month"] = s.wage_2024 * cpi_factor(month)
            s["income_month"] = s.wage_month  # alias for metrics.py
            E = feasible(s, p, share, level_gap)
            s["n_feasible_cell"] = E.sum(1)
            s["group"] = np.select([s.lower_wage & (s.n_feasible_cell >= 1), s.lower_wage], ["G+", "G0"], "higher")
            self.frames[key] = (s.reset_index(drop=True), p.reset_index(drop=True))
        return self.frames[key]


# ---- sampling and preferences ---------------------------------------------------------------------------------------------
def _rng(spec: JobMarketSpec, stream: str) -> np.random.Generator:
    key = f"{spec.cell}|{spec.month}|{spec.n_seekers}|{spec.n_postings}|{stream}"
    return np.random.default_rng([spec.seed, zlib.crc32(key.encode())])


def _weighted_sample(rng, frame, k, weight=None):
    p = None if weight is None else frame[weight].to_numpy(float) / frame[weight].sum()
    return frame.iloc[rng.choice(len(frame), size=k, replace=k > len(frame), p=p)]


def _sample_postings(rng, pool, k, spec):
    if spec.firm_sampling == "uniform" or k == 0:
        return pool.iloc[rng.choice(len(pool), size=min(k, len(pool)), replace=False)]
    remaining, picks = pool.copy(), []
    while k > 0 and len(remaining):
        seed_row = remaining.iloc[int(rng.integers(len(remaining)))]
        same = remaining[remaining.firm_id == seed_row.firm_id]
        take = same.iloc[rng.choice(len(same), size=min(spec.firm_cluster, k, len(same)), replace=False)]
        picks.append(take)
        remaining = remaining.drop(take.index)
        k -= len(take)
    return pd.concat(picks)


def sample_people(spec: JobMarketSpec, catalog: JobCatalog):
    frame, pool = catalog.frame(spec.cell, spec.month, spec.share, spec.level_gap)
    s_rng, p_rng = _rng(spec, "seekers"), _rng(spec, "postings")
    if spec.design == "natural":
        seekers = _weighted_sample(s_rng, frame, spec.n_seekers, "weight")
        postings = _sample_postings(p_rng, pool, spec.n_postings, spec)
    elif spec.design == "stratified":
        n_low = round(spec.lower_share * spec.n_seekers)
        for g in ("G+", "higher"):
            if ((frame.group == g) & (frame.n_feasible_cell >= 1)).sum() == 0:
                raise ValueError(f"cell {spec.cell!r} month {spec.month}: no {g} seekers with a feasible posting; "
                                 f"groups present: {frame.group.value_counts().to_dict()}")
        # Both groups are drawn from seekers with at least one feasible posting in the cell. In housing only G+ carried that
        # condition, because higher-income renters nearly always had options; here the pay floor binds higher earners, and an
        # unconditional draw left 47% of them with nothing. The excluded share is in describe_markets.
        higher_ok = frame[(frame.group == "higher") & (frame.n_feasible_cell >= 1)]
        seekers = pd.concat([_weighted_sample(s_rng, frame[frame.group == "G+"], n_low, "weight"),
                             _weighted_sample(s_rng, higher_ok, spec.n_seekers - n_low, "weight")])
        if spec.posting_band == "none":
            postings = _sample_postings(p_rng, pool, spec.n_postings, spec)
        else:
            if spec.posting_band == "top":
                in_band = pool[pool.salary >= float(pool.salary.quantile(0.75))]
            elif spec.posting_band == "bottom":
                in_band = pool[pool.salary <= float(pool.salary.quantile(0.25))]
            else:
                raise ValueError(f"unknown posting_band {spec.posting_band}")
            banded = _sample_postings(p_rng, in_band, round(spec.band_share * spec.n_postings), spec)
            rest = _sample_postings(p_rng, pool.drop(banded.index), spec.n_postings - len(banded), spec)
            postings = pd.concat([banded, rest])
    else:
        raise ValueError(spec.design)
    return seekers.reset_index(drop=True), postings.reset_index(drop=True)


def _to_ranks(score, E, axis):
    s = np.where(E, score, -np.inf)
    order = np.argsort(-s, axis=axis, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(s.shape[axis]).reshape((-1, 1) if axis == 0 else (1, -1)), s.shape), axis=axis)
    return np.where(E, ranks, NOT_EDGE)


def _expand_slots(postings: pd.DataFrame, spec: JobMarketSpec) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per slot. Returns (slot table, slot -> posting index)."""
    if spec.capacity == "one":
        q = np.ones(len(postings), dtype=int)
    else:
        q = np.clip(np.round(postings.expected_hires.to_numpy(float)), 1, CAPACITY_CAP).astype(int)
    slot_to_posting = np.repeat(np.arange(len(postings)), q)
    slots = postings.iloc[slot_to_posting].reset_index(drop=True)
    slots["slot"] = np.concatenate([np.arange(k) for k in q])
    return slots, slot_to_posting


def build_market(spec: JobMarketSpec, catalog: JobCatalog) -> JobMarket:
    seekers, postings = sample_people(spec, catalog)
    slots, slot_to_posting = _expand_slots(postings, spec)
    wage = seekers.wage_month.to_numpy(float)[:, None]
    salary = slots.salary.to_numpy(float)[None, :]
    E = feasible(seekers, slots, spec.share, spec.level_gap)

    prefs = _rng(spec, "preferences")
    # Posting-level shocks are drawn per posting, then spread to slots, so slots of one posting are ranked together.
    post_shock = prefs.standard_normal(len(postings))[slot_to_posting][None, :]
    quality = _standardize(np.log(salary)) + spec.common_shock * post_shock  # TODO add log employer headcount if a company table exists
    fit = np.clip(np.abs(slots.seniority.to_numpy()[None, :] - seekers.experience.to_numpy()[:, None]), 0, 2)
    pair_shock = prefs.standard_normal((len(seekers), len(postings)))[:, slot_to_posting]
    slot_tiebreak = -1e-6 * slots.slot.to_numpy()[None, :]  # consecutive slots, strict order
    u_seeker = (spec.w_wage * np.log(salary / wage) + spec.w_quality * quality - spec.w_fit * fit
                + spec.seeker_noise * pair_shock + slot_tiebreak)

    in_band = (wage >= 0.8 * slots.salary_min.to_numpy(float)[None, :]) & (wage <= 1.2 * slots.salary_max.to_numpy(float)[None, :])
    match = seekers.experience.to_numpy()[:, None] >= slots.seniority.to_numpy()[None, :]
    emp_shock = prefs.standard_normal((len(seekers), len(postings)))[:, slot_to_posting]  # identical across a posting's slots
    u_employer = spec.w_payband * in_band + spec.w_match * match + spec.employer_noise * emp_shock

    firm_rng = _rng(spec, "firms")
    k = max(2, int(np.ceil(spec.n_seekers / spec.seeker_firm_size)))
    order = np.argsort(seekers.wage_month.to_numpy(), kind="stable") if spec.seeker_firm_mode == "by_income" else firm_rng.permutation(spec.n_seekers)
    firm_b = np.empty(spec.n_seekers, dtype=np.int64)
    firm_b[order] = np.arange(spec.n_seekers) * k // spec.n_seekers
    names, firm_s = np.unique(slots.firm_id.to_numpy(), return_inverse=True)
    firm_names = [f"agent:{i}" for i in range(k)] + names.tolist()

    inst = Instance(_to_ranks(u_seeker, E, axis=1), _to_ranks(u_employer, E, axis=0), firm_b, firm_s + k,
                    np.ones(len(firm_names), dtype=np.int64), firm_names)
    seekers = seekers.assign(n_feasible_market=E.sum(1), has_children=False)  # has_children: alias so metrics.py runs
    slots = slots.assign(rent=slots.salary)  # alias for metrics.py / exposure.rent_max; drop once they read price_col
    groups = {"all": np.ones(len(seekers), bool), "lower_wage": seekers.lower_wage.to_numpy(bool), "higher_wage": ~seekers.lower_wage.to_numpy(bool)}
    market = JobMarket(spec, inst, seekers, slots, len(names), k, "salary", groups, slot_to_posting)
    # The transparent policy ranks on the seeker score with both shocks removed, at posting level (one column per posting).
    first_slot = np.unique(slot_to_posting, return_index=True)[1]
    sal_p = slots.salary.to_numpy(float)[first_slot][None, :]
    fit_p = np.clip(np.abs(slots.seniority.to_numpy()[first_slot][None, :] - seekers.experience.to_numpy()[:, None]), 0, 2)
    market.score_deterministic = spec.w_wage * np.log(sal_p / wage) + spec.w_quality * _standardize(np.log(sal_p)) - spec.w_fit * fit_p
    return market


def _standardize(x):
    return (x - x.mean()) / (x.std() + 1e-9)
