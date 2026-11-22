# Headline findings: LLM shortlists in NYC rentals, NYC jobs and Chilean schools

Short version. Full numbers, intervals and caveats are in [`README.md`](README.md).

**Setup.** `gpt-6-luna` picks 5 options per person; we compare against 5 random options on the same markets and clear the market with deferred acceptance. There are two simulation studies:
- **NYC rentals:** [`gpt-luna-rentals/`](gpt-luna-rentals/README.md), 3 windows.
- **NYC job postings:** [`jobs-gpt-frozen/`](jobs-gpt-frozen/README.md), Revelio data, frozen design plus a held-out replication.

Chilean school admissions ([`publication/research/observed/school/`](../publication/research/observed/school/README.md)) is an observed-data check that uses logistic classifiers, not an LLM.

1. **GPT's lists place fewer people in both NYC markets.** Placement falls 2–4 pp in rentals and 15–18 pp in jobs, in every window. The cause is concentration: lists pile onto the same few options, so they can't hold everyone. The capacity of the shown lists falls by 9–11 pp in rentals and about 20 pp in jobs. In rentals, fewer collisions win back about two-thirds of that, so the net loss is small; in jobs only about a fifth comes back. Reserving one slot per list fixes it: it erases the whole rental loss (anchored GPT matches or slightly beats random) and recovers 13–16 pp in jobs. This is the most constructive result, and it's a concrete version of the paper's capacity argument.

2. **The placement loss falls on whoever has the most options.** That's higher-income renters (46 feasible listings against 14), who lose about 6 pp. In jobs it's lower-wage seekers (about 21 feasible jobs against 11), who lose about 23 pp.

3. **"More accurate but places fewer" depends on the comparison.**
   - **Rentals:** it holds, with about +8 pp accuracy and −2 to −4 pp placed.
   - **Jobs:** GPT's 5-job lists are no more accurate than random, and the small edge disappears under calibration and in the replication. Only half-length lists show the pattern.
   - **Chile:** both classifiers beat random lists on accuracy *and* placement. The pattern appears only when stated rank is added to the model: more accurate, about 520 fewer placed per year.

4. **GPT widens the lower- vs higher-group gap in missing a better option (T), but for different reasons.**
   - **Rentals:** about +7 to +8 pp. GPT *helps* higher-income renters (misses 86% → 80%) and does nothing for lower-income renters.
   - **Jobs:** about +4 pp in markets drawn without oversampling, and not significant in the pre-specified stratified design. The group switch shows GPT *hurts both groups* through their own lists (lower-wage misses 75% → 81%, higher-wage 39% → 45%), so the harms roughly cancel in the gap.

5. **Repairs are cheap in jobs.**
   - **Repair cost:** the cheapest stable repair costs about the same under GPT as under random, and slightly less.
   - **Protection:** protecting lower-wage seekers from ending up worse off costs about 0 extra.
   - **Losers:** about 17% of seekers still end up worse off after the repair.

   These hold with one firm per seeker (the housing convention). The rental runs had no audits.

**Caveats.**
- All NYC outcomes are simulated.
- Timing of the jobs analyses (`jobs_manifest.json`, `later_additions`):
  - **T, group switch, protection premium:** named in the shared protocol before the jobs freeze.
  - **Accuracy and capacity/anchor:** added after the frozen-month results, but fixed before the held-out replication, which they then passed.
  - **Calibrated T and one firm per seeker:** fully post hoc.
  - **Rentals:** the accuracy and capacity analyses are equally post hoc there.
- Jobs T intervals are about ±3 pp, so small effects could be missed.
- "Missed a better option" and "placed" can move in opposite directions, so report both.
