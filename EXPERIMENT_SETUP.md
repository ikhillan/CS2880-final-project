# Experiment setup: stable matching with hidden options (NYC rentals)

**Goal:** use real NYC renters and listings to test whether hidden options hurt lower-income renters
more, and what audits cost and change.

## Data
- **Renters:** 10,302 recent-mover records (about 299,000 households when weighted) from the
  ACS 2020–24 Census microdata, out of 71,908 renter records (2.18M households). Incomes are moved
  to 2026 dollars with CPI.
- **Listings:** FirstMover NYC listings, June–August 2026 (about 20k a month). Each listing's
  brokerage is its firm.

## What a renter looks like
Each simulated renter copies one real Census household:
```json
{"household_size": 4, "has_children": true, "income": 56035, "max_rent": 1868,
 "min_bedrooms": 1, "borough": "Brooklyn", "group": "lower income, has options"}
```

## What a listing looks like
```json
{"listing_uid": "fm:5134399", "rent": 1700, "bedrooms": 1, "bathrooms": 1,
 "neighborhood": "Midwood", "building_type": "RENTAL", "firm": "Elite Realty of USA"}
```

## Assumptions: who can rent what (E)
A renter–listing pair is feasible if all of these hold:
- **Affordable:** rent ≤ 40% of income. That's a middle ground: the common landlord screen
  ("earns 40× the monthly rent") equals 30% of income, while lower-income NYC movers actually pay a
  median of 52%. We test 30% and 50% as well.
- **Big enough:** at most two people per bedroom, and a studio works for 1–2 people (3–4 people
  need 1 bedroom, 5–6 need 2).
- **Same borough and same month:** about 78% of renters who move within NYC stay in their borough.
  A citywide search is a sensitivity check.

## Preferences (modeled, since neither dataset has real preferences)
Each side gives every feasible partner a score, and a higher score means a better rank. Only pairs
in E get ranked.

**Renter b's score for listing l:**
```
u(b, l) = w_q × quality(l)                          quality
        − w_p × rent(l) / max_rent(b)               price relative to own budget
        + w_s × min(spare bedrooms, 2)              extra space
        − w_d × (km from b's home area) / 5         distance
        + σ_r × ε(b, l)                             personal taste

quality(l) = rent premium(l) + 0.5 × ξ(l)
rent premium(l) = log rent − average log rent of same-size listings in the same neighborhood area
                  (standardized)
```

**Listing l's (landlord's) score for renter b:**
```
v(l, b) = w_i × min( income(b) / (12 × rent(l)), 40/12 )    income vs. rent, capped at the 40× screen
        + σ_l × η(l, b)                                     things we can't see, like credit
```

- **Noise:** ε, ξ and η are standard normal random draws. ξ is shared by all renters, so some
  listings are generally more appealing; ε and η differ for each pair.
- **What landlords ignore:** race, children and household size.
- **Starting weights:** w_q = 1, w_p = 2, w_s = 0.5, w_d = 0.5, σ_r = 0.75, w_i = 2.5, σ_l = 1.
- **Calibration:** renter weights (w_q, w_p, σ_r) are tuned so matched renters' rent burdens
  resemble what real NYC movers pay. Landlord weights have no data behind them, so we vary them in
  sensitivity checks.

## Building markets
- **Why small:** the exact solver can only prove the cheapest audit plan up to about 128 renters,
  so we sample many markets of 24 or 64 per side from each borough and month.
- **Why not uniform:** uniform draws leave 76–98% of lower-income renters with nothing affordable.
  Instead, half the renters are lower income with at least one affordable option, and half are
  above the median. Half the listings come from a cheap rent band.
- **Reproducible and paired:** draws are seeded, so every method runs on identical markets.

## What renters see (E0)
Each renter sees the same number of listings under every policy, either a top 5 or 40% of their
options:
- **Random:** listings picked at random, as the baseline.
- **Steered:** the priciest affordable listings, as a stand-in for commission-driven steering.
- **AI:** a fixed Claude model picks from the renter's profile and affordable listings. The model,
  prompt and settings stay the same for the whole study.

## Matching and audits (from Todd's presentation)
1. **Deferred acceptance on E0:** renters propose down their rankings, but only to listings they
   were shown.
2. **Hidden blocking pairs:** check the result against the full market E. A blocking pair is a
   renter and listing who both prefer each other, but the renter was never shown it.
3. **Audits:** auditing a firm opens every feasible pair for its clients. Brokerages are the
   listing-side firms, and each renter has their own firm. Every audit costs 1.
4. **Cheapest audit plan:**
   - **Exact:** Todd's exact algorithm, which learns from failed DA runs and proves the plan is
     cheapest (or reports a gap at a time limit).
   - **Heuristics:** reverse-delete, blocker-greedy, prefix cover, and auditing everyone.

   Every returned plan is checked to be stable against the full market.

## Comparisons
**Lower vs. higher income** (cutoff: the NYC renter median, $68,808):
- **Matches:** who gets matched, before and after audits.
- **Hidden options:** how many feasible options each renter wasn't shown.
- **Rank:** how good each renter's match is by their own ranking, before and after audits.
- **Who's worse off:** who ends up with a worse match after the audit.
- **Equally cheap plans:** whether different cheapest plans favor different groups.

Renters with no affordable option are reported separately as structural, never blamed on hidden
information.

**Is a stable matching possible?**
- **On the full market:** always, by Gale–Shapley.
- **What we measure:**
  - how often DA on what renters see is unstable against the full market
  - the cost of making a stable matching *available* (C_support) vs. getting DA to *pick* one (C_DA)
  - whether adding audits ever breaks stability

## Scale and compute
- **Size:** about 1,050 markets across 5 experiment sets: core, solver comparison, sensitivity
  checks, size limits and very large markets.
- **Compute:** no cluster or GPU needed. Claude estimates about 2–2.5 hours on a laptop (worst case
  about 6), almost all of it the audit solver rather than the data.

## Status
**Done**
- **Profiles:** code that turns Census households into renter profiles and loads the cleaned
  listings.
- **Markets:** the sampler, feasibility rules and preference model.
- **Matching:** deferred acceptance and the full-market stability check.
- **Audits:** the exact solver and the heuristics.
- **Recommender:** the AI recommender pipeline.
- **Runner and analysis:** an experiment runner (resumable, parallel) and analysis scripts.
- **Tests:** 68 tests pass, including every worked example from the slides.

**Before running**
- **Renter firms:** give each renter their own firm. Shared agents let "audit every agent" open the
  whole market cheaply.
- **Preferences:** recalibrate, because renters currently prefer higher rent overall.

**Next:** run the experiments tonight, then go through metrics tomorrow.
