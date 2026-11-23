# Faculty presentation review

I would present [Who gets a place?](presentation-narrated.mp4) to the course. The result to lead with is that renters who saw every feasible apartment could still gain or lose a match when other people’s lists changed. The NYC simulation isolates this effect. The Chilean replay shows why a school district would care about it. Jobs supplies a useful challenge to the idea that better AI predictions drive the gains.

## What the three reviewers challenged

| Review | What a professor could reasonably ask | What changed |
| --- | --- | --- |
| Research evidence | A doubled rate can sound like a real housing intervention, or an improvement for every renter. | Slide 2 identifies the simulation, subgroup, policies and sample size. Slide 8 shows 349 entrants **and 15 exits**. The appendix reports all six design-by-window comparisons. |
| Course requirements | The earlier opening posed a retention problem and made the audience wait through definitions for a different result. | Slide 1 asks whether changing someone else's recommendations can affect you. Slide 2 answers it. The two-seat example follows the evidence. |
| First-time viewer | Three graphs plus several tables competed with fast narration. The Chilean field trial and the authors' replay could sound like a single study. | The main talk uses one empirical chart, two small comparison tables and two diagrams. The field-trial slide names the external authors, intervention, outcomes and uncertainty. The seven reveals are timed to the spoken argument. |

## Where the course requirements appear

| Requirement | Slide and video time | What the audience hears or sees |
| --- | --- | --- |
| Motivation and problem | 1 · 0:00 | Two renters share a market. If your rental list stays the same, can someone else's new recommendations change whether you get an apartment? |
| Main finding immediately afterward | 2 · 0:14 | In the NYC simulation, the subgroup saw all its feasible listings under both policies. Its match rate rose from 13.1% to 27.5%. Both numbers appear by 0:30. |
| Mechanism and solution approach | 3 · 0:55, 6 · 3:17, 12–13 · 9:36 | A two-seat example, two implementable coordination rules, and a reduction explaining why selecting many compatible edits with zero exits can be hard. |
| Evaluation | 7–11 · 4:42, 14–17 · 11:52 | Paired comparisons, recipient-level transitions, counterfactual admissions, previously unseen lottery draws, and external randomized evidence on real families' behavior. |
| Ethical implications | 18 · 15:50 | Every previous recipient kept a seat under the structural rule. In the paired tests nobody previously unplaced gained one. A family can also keep a seat at a less preferred school. The slide proposes monitoring entrants, preserving choice and allowing corrections and appeals. |
| Narrated video with few graphs | 19 slides · 17:00 | The team’s script is read by three disclosed American synthetic voices at the selected 1.45× source tempo. The PDF can be used as a handout. The video uses builds and toggleable English captions. |

## Where the evidence ends

- **Rentals.** Across the three studied windows, the fixed-list subgroup gains 14.4–14.9 percentage points under enriched sampling and 20.9–24.5 points under ordinary sampling. The rates average within seeds. In the focal cell, the pooled counts reconcile as 304 + 349 − 15 = 638 among 2,328 appearances. The comparison gives balancing more freedom to edit lists than the anchor. The windows reuse properties, and the preferences and assignments are simulated. [Six comparisons and intervals, appendix slide 9](technical-appendix.pdf#page=9)
- **Jobs.** Applying coordination to GPT raises placement from 63.5% to 79.4%. Coordinated random reaches 81.4%. The second comparison uses released point means. An outer bound permits between 0 and 308 prior recipients to lose access. The data release leaves their identities and post-anchor coverage open. [Released jobs audit](../../findings/jobs-anchor-access/README.md)
- **Chilean admissions.** In the linked 2021 entry grades, roughly 97% of pupils had unchanged lists and that group accounted for 237 of 309 net lost placements. Following only pupils whose lists changed would miss most of the net change in this cohort. In a separate analysis of the 2022–25 cohorts, 7.4% of linked applicant-years had no historical initial offer, and that group accounted for 76% of the net placement decline. Changing the comparator reverses the national placement result. [Chilean comparisons](../../findings/chile/README.md)
- **Theory and deployment.** The NP-completeness result concerns choosing shortlists for an explicit panel of strict priority draws. Deferred acceptance evaluates a chosen policy in polynomial time. The prospective structural guarantee covers permitted ordinary orders with fixed inputs in the reconstructed mechanism. Its 28 extra historical enrollment hits across about 1.5 million records are a small coverage gain. [Proof](../../publication/two-studies/retention-theory.tex)
- **Actual behavior.** In a separate randomized Chilean study, qualifying second choices rose by 5.4 percentage points. The any-assignment estimate was +0.2 points, with an approximate 95% interval from −2.0 to +2.4. Families responded to advice. The field result leaves the sign of the assignment effect open. A constructed three-pupil market explains one way individual treatment and full rollout can diverge. [Study](https://doi.org/10.3386/w32670) · [Contrasts and standard errors](../../publication/two-studies/field-study/trials/explorer-contrasts.csv)

The most important unanswered question is how a coordinated policy would affect applications and placements under a real market-wide rollout. That is the empirical limit to keep in view in faculty Q&A.

The finished video was decoded in full. All 78 sampled frames matched the expected Beamer pages, and 352 captions were aligned to all 136 spoken passages. A separate speech-recognition check of 26 passages recovered the main numerical findings at final playback speed. [Verification manifest](narration-manifest.json) · [Technical appendix](technical-appendix.pdf)
