# Experiment setup: NYC rental shortlists and the cost of a stable repair

*Updated 11 November 2026 to follow Todd's protocol (`FINAL_EXPERIMENT_PROTOCOL_2.md`).*

**Goal:** a renter can see five affordable apartments and still miss the one where both sides would prefer to
match. We ask whether the recommender changes how often this happens to lower-income renters, and what it costs a
platform to repair the matching while protecting renters from avoidable losses.

## Main questions
| Question | What we measure | Comparison |
|---|---|---|
| Which listings does the AI show? | How expensive the shown listings are within each renter's options (rent percentile), by income group | Claude vs. random picks of the same length |
| Does that change the matching? | **T** (main statistical result): change in the lower- minus higher-income gap in the share of renters with a hidden blocking pair | Claude vs. random on held-out months, 64 renters and 64 listings |
| What does protecting renters cost? | **Δ_L** (main optimization result): extra audits needed so lower-income renters lose only what they unavoidably must | Protected repair vs. ordinary repair, on the same Claude shortlists |
| Do results depend on our assumptions? | Both results under other preferences, shortlist rules and months | Same markets, one change at a time |

The price result comes straight from the IDs the AI returns, so it doesn't depend on our modeled preferences.

## Data
**Renters:** ACS 2020–24 Census microdata.
- **Records:** 10,302 recent-mover records (about 299,000 households weighted), out of 71,908 renter records (2.18M
  households).
- **Split:** households are split 70/30 by a fixed hash into **calibration** donors (used to tune preferences) and
  **evaluation** donors (used in the experiments).
- **Lower income:** at or below the median income of all NYC renting households.
- **Dollars:** everything is converted to July 2026 dollars with the Census adjustment factors and CPI.
- **Zero income:** households with zero or negative income have no affordable listings. They stay in the counts.

**Listings:** FirstMover NYC listings, **21 months, February 2025 to October 2026**, about 371,000 NYC listings in the
analysis catalog. All are downloaded and cleaned.

| Role | Months | Why |
|---|---|---|
| Development | Jun–Aug 2026 | Building, tuning and the pilot |
| Held-out, primary | Sep–Oct 2026 | Todd's choice. Both have collection gaps, which we report openly |
| Held-out, replication | Jun–Aug 2025 | Complete months, same season as development |
| Timing robustness | All 21 months | Do results hold across seasons and years? Non-AI policies only |

Nobody looks at held-out outcomes until the design is frozen.

**Collection gaps**, from `scripts/firstmover_coverage.py`. A low day has under 40% of a typical weekday's postings;
isolated holiday and weekend lows are normal.

| Month | NYC listings | Problem |
|---|---|---|
| Feb 2025 | 10,977 | Collection starts around the 9th |
| Oct 2025 | 14,179 | 27th–31st missing |
| Nov 2025 | 9,832 | 1st–8th missing |
| Sep 2026 | 13,776 | 12 low days, including the 10th–13th and 27th–30th |
| Oct 2026 | 13,838 | 1st–5th low |

The other 16 months are complete. Gappy months are kept, not dropped. Each market draws only 24–64 listings from
about 10,000–24,000, so a few missing days rarely change a market. A sensitivity check reruns without the gap days,
and compares listings on complete days with the rest.

**Listing cleaning:**
- **Duplicates:** repeated listings and repeat postings of the same unit are collapsed, with an exclusion ledger.
- **Brokerages:** names are matched with a fixed alias map, and sources that can't be resolved get their own single firm.
- **Rent:** asking rent is primary, with net-effective rent as a sensitivity check.

**Boroughs:** all five. Staten Island has only about 20 listings a month, so its markets are marked as unestimable.

**Months without CPI:** October 2025 has no published value (the 2025 funding lapse), so it uses the average of September
and November 2025. October 2026 isn't published yet, so it carries September 2026 forward until the value appears.

## What a renter looks like
Each simulated renter copies one real Census household:
```json
{"household_size": 4, "has_children": true, "income": 56035, "max_rent": 1868,
 "min_bedrooms": 1, "borough": "Brooklyn", "group": "lower income"}
```

## What a listing looks like
```json
{"listing_uid": "fm:5134399", "rent": 1700, "bedrooms": 1, "bathrooms": 1,
 "neighborhood": "Midwood", "building_type": "RENTAL", "firm": "Elite Realty of USA"}
```

## Assumptions: who can rent what (E)
A renter–listing pair is feasible if all of these hold:
- **Affordable:** rent ≤ 40% of income. The common landlord screen, "earns 40× the monthly rent", equals 30% of
  income, while lower-income NYC movers actually pay a median of 52%. 30% and 50% are sensitivity checks.
- **Big enough:** at most two people per bedroom, and a studio works for 1–2 people. A stricter rule (⌈people/2⌉
  bedrooms for 3 or more people) is a sensitivity check.
- **Same borough and month:** renters search their own borough, starting from the center of their Census area.
  Citywide search is a sensitivity check.

## Preferences (modeled, since neither dataset has real preferences)
Each side scores every feasible partner, and a higher score ranks higher.

**Renter b's score for listing j:**
```
u(b, j) = w_q × (Q_j + 0.5 × ξ_j)                quality, plus a shared listing shock
        − w_p × rent_j / budget_b                price relative to own budget
        + w_s × min(spare bedrooms, 2)           extra space
        − w_d × distance_km / 5                  distance
        + σ_r × ε_bj                             personal taste
```
- **Quality Q_j:** the average of standardized log square footage and standardized bathrooms. Both are standardized
  within borough × bedroom group (studio, 1, 2, 3+), using citywide references when a group has under 30 listings.
  Missing values are filled with the median, and values are clipped to ±3. Square footage is reported for only
  25–31% of listings, so in practice quality mostly reflects bathrooms. Rent premium (rent after accounting for size
  and neighborhood) is a sensitivity check.
- **Fixed weights:** w_q = 1, w_s = 0.5, w_d = 0.5.
- **Calibrated weights:**
  - **The grid:** w_p ∈ {1, 2, 4} and σ_r ∈ {0.5, 0.75, 1.25}.
  - **The target:** the 25th, 50th and 75th percentile rent burdens of matched renters under full disclosure,
    matched to Census recent movers whose burden fits under the 40% cap, with each income group weighted equally.
  - **The setup:** calibration donors and development months only.
  - **Sensitivity:** all nine settings are kept for the sensitivity checks.

**Listing (landlord) j's score for renter b:**
```
v(j, b) = 2.5 × min( income_b / (12 × rent_j), 40/12 ) + η_jb
```
Landlords value income relative to rent up to the 40× screen, plus noise for what we can't see, like credit.

**Noise and ties:** ξ, ε and η are standard normal draws. ξ is shared by all renters, while ε and η differ for each
pair. Remaining ties are broken by a fixed, seeded order.

## Building markets
- **Size and scope:** each market is one borough × one month, at 24 or 64 per side. The exact solver can only prove
  the cheapest plan up to about 128 per side.
- **Primary design, uniform:** households are drawn by Census weight from evaluation donors in that borough, and
  listings are drawn uniformly from that borough-month. The draw is kept even if nobody can afford anything.
- **Second panel, enriched:** built so lower-income renters actually have options to lose.
  - **Affordability band:** take the lower-income households in the borough that can afford at least one listing that
    month, and find the budget that 75% of them (by Census weight) are at or below. Every listing at or under that
    rent is in the band. In August 2026 the band tops out at about $1,980–2,090 in every large borough.
  - **Listings:** half come from the band, and half at random from the listings above it.
  - **Renters:** half are lower-income with a feasible option in the draw, and half are higher-income.
  - **Unestimable:** a market is marked unestimable if the band or the rest of the month has too few listings, or no
    lower-income household can afford anything in the draw. Staten Island's band has only 4–8 listings.
  - **Old rule, kept for comparison:** the first version took half the listings from the cheapest quarter of rents.
    In Manhattan that quarter runs up to about $3,700, above every lower-income budget, so it often failed. It stays
    available as `enriched_quartile`.
- **Scarcity report:** every group is split into four categories:
  1. no affordable listing in the whole month
  2. affordable listings exist, but none in the draw
  3. has options but unmatched even with full information
  4. matched with full information
- **Seed blocks:** each block has one market per borough-month cell. Every policy and method runs on identical
  markets.

## What renters see (E0)
- **Shortlist length:** two budgets, and both run in the pilot and the protection stages:
  - **Top 5:** k = min(5, d), where d is the number of feasible listings. Only renters with more than 5 options can
    have something hidden.
  - **40% of options:** ⌈0.4d⌉ listings. Anyone with 2 or more options has something hidden, so this budget reaches
    lower-income renters, who usually have few options.

| Policy | How it picks |
|---|---|
| Uniform | k feasible listings at random: the baseline |
| Transparent | Top k by the renter's score, without the random parts |
| Expensive-first | The k highest rents |
| Claude | A pinned Claude model with a fixed prompt and settings |
| Qwen 2.5 32B and Mistral Small 24B | Open models, run locally on GPUs |
| Full visibility | Everything feasible: the benchmark |

- **What the AI sees:** budget, minimum bedrooms and search area, plus each listing's anonymous ID, rent, bedrooms,
  bathrooms, square footage (with missing flags) and distance.
- **What it doesn't see:** names, income group or brokerage.
- **Order:** listings are shuffled the same way for every model.
- **Bad replies:** we keep valid IDs, fill any shortfall with seeded random picks, and report the fill rate by group.

## Matching, blocking pairs and audits
1. **DA on E0:** renters propose down their rankings, only to listings they were shown.
2. **Hidden blocking pairs:** a renter counts as affected (Y_b = 1) if some listing they weren't shown would prefer
   them, and they'd prefer it, over their current matches.
3. **Audits:**
   - **Who's audited:** each renter has their own firm, and each listing its brokerage.
   - **What an audit does:** it opens every feasible pair for that firm's clients.
   - **Cost:** every audit costs 1. Brokerage-only audits and costs that grow with portfolio size are sensitivity
     checks.
4. **Four cheapest-plan objectives**, which always satisfy C_support ≤ C_DA ≤ C_L ≤ C_all ≤ full disclosure:
   - **C_support:** a stable matching is *available* in what's visible.
   - **C_DA:** DA actually *picks* a stable matching (the ordinary repair).
   - **C_L:** the ordinary repair, plus lower-income renters lose only what's unavoidable.
   - **C_all:** the same protection for every renter.
5. **Unavoidable losses:** M_E is DA with full information, and it gives every renter their best possible stable
   match. A renter worse off under M_E than in their initial match loses under *every* repair; anyone else can be
   protected.
6. **Protection premium:** Δ_G = C_G − C_DA, the extra audits needed to protect group G.
7. **Methods:**
   - **Exact:** Todd's algorithm, with one extra constraint per protected renter (final match at least as good as
     the initial one).
   - **Heuristics:** reverse-delete (also a protected version), blocker-greedy, prefix cover (which reproduces M_E,
     so it's already protected), and full disclosure.
   - **Checking:** every plan is replayed and checked for stability and protection.
   - **Timeouts:** they report verified upper and lower bounds.

## Analyses
- **Price of shortlists:** the average rent percentile of shown listings, by group. Random picks average exactly 0.5.
- **T:** (lower-income Claude − lower-income uniform) − (higher-income Claude − higher-income uniform), in
  hidden-blocker rates. We show all four rates. We also show q, the share of renters with more options than shown:
  only they can have a hidden option of their own.
- **Group-switch experiment:** DA runs four ways per market, where each income group sees Claude or uniform picks.
  - **S_L:** the effect of switching lower-income renters' lists, averaged over both switching orders.
  - **S_H:** the same for higher-income renters.
  - **Check:** S_L + S_H = T. This separates a group's own shortlists from competition with the other group.
- **Protection premium:** the distribution of Δ_L and Δ_all, including how often it's positive, zero or unresolved.
- **Rank severity:** each renter's rank after the repair minus their rank under M_E, alongside how many lose.
- **Equally cheap plans:** whether different cheapest plans favor different groups.
- **Is a stable matching possible?**
  - **On the full market:** always, by Gale–Shapley, and every successful repair matches the same set of renters.
  - **What we measure:**
    - how often DA on what renters see is unstable
    - C_support vs. C_DA
    - whether adding audits ever breaks stability

## Stages and scale
| Stage | Runs | Where |
|---|---|---|
| 0. Correctness | Known examples, Todd's 160 archived markets, and 200 tiny markets checked by brute force | Laptop |
| 1. Calibration and pilot | Calibrate preferences, then 300 uniform and 300 enriched markets at 64 per side under every policy, at both budgets. The pilot's variance sets the main sample size. | Laptop, plus GPUs for the open models |
| 2. Main comparison | 100–400 held-out seed blocks = 1,000–4,000 markets, every policy | Cluster for the open models; up to 256,000 Claude requests |
| 3. Audit and protection costs | 400 uniform + 200 enriched markets per size (24 and 64), 3 policies × 4 objectives | Laptop for short solver limits; cluster for long ones |
| 4. Sensitivity and solvers | 100 markets with one change at a time, plus the three solver variants | Laptop |
| 5. Larger markets | 100 markets at each of 128, 256 and 1,024 per side | Cluster |

## Validation with people
Preferences and audit costs are modeled, which is the main limitation. Two things would strengthen the case:
- **A renter study:** 100 recent NYC renters each rank 10 pairs of listings.
- **Interviews:** housing referral staff explain what an "audit" would mean in practice.

Both need IRB approval. Without them, we describe the work as a semi-synthetic NYC matching experiment.

## Status
**Done**
- **Data:** all 21 FirstMover months downloaded and cleaned, with a daily coverage check (`scripts/firstmover_coverage.py`).
- **Renters:** Census profiles with the 70/30 calibration/evaluation donor split, July 2026 dollars, and contract rent for the
  calibration target.
- **Markets:** uniform and enriched (affordability-band) designs, plus the old quartile rule for comparison, one firm per renter, Todd's preference formula with the square-footage and
  bathroom quality measure, and unestimable cells (like Staten Island) recorded instead of crashing.
- **Shortlists:** uniform, transparent, expensive-first, full visibility and AI arms, with the fallback fill for bad replies.
  The prompt follows the protocol.
- **Audits:** the four objectives (C_support, C_DA, C_L, C_all), protected reverse-delete and prefix cover, and the
  no-support solver variant. Todd's two constructions are regression tests: protecting everyone costs N in the
  shared-brokerage family and (n+1)/2 in the ring.
- **Analyses:** T with cell weights, month-by-month and conditional versions, and seed-block intervals; the group-switch
  split (S_L + S_H = T is checked); protection premiums with solver bounds; rank severity; shortlist price percentile; the
  scarcity categories.
- **Running:** configs for every stage, `--shard` for Slurm array jobs, and the calibration script. 115 tests pass.

**Early observations** (from small smoke runs, not results)
- **Uniform design:** about 70–99% of lower-income renters have no affordable listing in a 64-listing market, so their
  hidden-blocker rate is mostly zero. T will mostly reflect higher-income renters unless the enriched panel carries the
  lower-income comparison.
- **Enriched design:** 10 markets per borough (August 2026, 64 per side, calibration donors):

  | Lower-income renters | Uniform | Old quartile rule | Affordability band |
  |---|---|---|---|
  | Markets unestimable (of 40) | 0 | 2 (Manhattan) | 0 |
  | No affordable listing | 76–100% | 0% | 0% |
  | Median number of options | 0 | 1 (Manhattan, Brooklyn) to 7 (Bronx) | 9–16 |
  | More than 5 options (can lose one under top 5) | 0–12% | 0–2% (Manhattan, Brooklyn), 40–65% (Queens, Bronx) | 55–69% |
  | 2 or more options (can lose one under 40%) | 0–21% | 16–88% | 76–89% |

  Higher-income renters barely change: a median of 41–52 options, and 87–100% have more than 5, in every design.
  Manhattan's band is thin (32–45 listings for the 32 needed at 64 per side), so the band half of a Manhattan market is
  nearly the same listings every time.
- **Protection groups:** the lower-income protection group is often empty, because few lower-income renters are matched
  initially.

**Still to do**
- **Brokerage names:** a reviewed alias map. Firms are currently matched as written.
- **Listings:** removing physical units that appear in both development and held-out months.
- **Open models:** running Qwen and Mistral on FASRC GPUs, which needs a separate script.
- **Model order and repeatability:** the repeat-prompt panel.
- **Solver check:** comparing our solver with Todd's on his 160 archived markets.
- **Stages 0–1:** calibration (20 seeds), the 300-market pilot and solver timing, followed by the AI test and FASRC setup.

**Pending decisions**
- **Scope:** course project or IJCAI 2027 paper (abstracts due January 4, papers January 11).
- **AI:** the model and budget.
- **Compute:** GPU and cluster access.
- **People:** IRB for the renter study.
- **Design:** whether the uniform design should stay primary, given how few lower-income renters have options under it.
