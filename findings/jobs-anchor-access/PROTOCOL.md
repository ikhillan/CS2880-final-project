# Jobs anchor: coverage, access, and coordinated comparisons

Version 1, 22 November 2026. This is a **post hoc follow-up** to both released jobs windows. The new endpoints and baselines were specified after their original placement results were known. They are not a preregistered confirmation or a new independent replication.

## Question

Does adding one coordinated option preserve reference-job coverage and individual access? Does an anchored GPT menu outperform coordinated alternatives with the same menu sizes and access to the same feasibility graph?

## Fixed evaluation panel

- Preserve `configs/jobs_frozen.json` and `configs/jobs_replication.json`, including market IDs, seeds, data order, preference coefficients, and cached GPT responses. Do not call the LLM again or replace missing responses with a new model's output.
- Primary: 240 August–October 2025 markets and 160 November–December 2025 markets, 24 seekers and 24 unit-capacity postings each, 20 seed blocks per window.
- Report the natural-sampling variant separately as a sensitivity. Never pool it with the stratified primary variant. Additional preference settings, if run, are separately labeled sensitivities.
- Construct the random, GPT, and deterministic-score menus once per market. Each method keeps each person's original posting count, normally min(5, feasible degree). Reuse the same modeled preferences for all allocations.
- Verify original GPT, random, anchored GPT and anchored random placement means against the released tables before interpreting any new result. Abort publication of new estimates if reproduction fails.

## Information and baselines

The policy constructors receive feasibility, posting capacities and menu information. They do **not** receive simulated preference shocks, employer rankings, full-information assignments, or baseline realized recipients. Those are available only to the evaluator. Keeping a seat for a realized recipient by inspecting that recipient's assignment would be an oracle in this experiment.

| Arm | Construction | Information regime |
|---|---|---|
| Random | Original seeded random menu | Feasibility and capacities |
| GPT | Original cached GPT menu | Original public structured prompt |
| Score | Original deterministic-score menu | Knows the simulator's deterministic coefficients; model-informed benchmark |
| Anchor + Random | Released maximum-cardinality, maximum-reuse anchor | Random menu plus full feasibility/capacities |
| Anchor + GPT | Same released anchor and GPT's last-pick deletion | GPT menu/order plus full feasibility/capacities |
| Anchor + Score | Same released anchor and lowest-score deletion | Model-informed benchmark |
| Balanced + GPT | Exact fixed-quota menu optimization: minimize sum of squared posting exposure loads, then maximize cached GPT rank scores | Same feasibility/capacity/GPT menu information as Anchor + GPT; unlisted options tied, not treated as known bad |
| Score anchor + GPT | Maximum-cardinality anchor with deterministic-score secondary objective; GPT deletion rule | Model-informed benchmark; not an equal-information improvement over GPT |
| Balanced + Score | Same fixed-quota load optimization with deterministic-score ranks | Model-informed benchmark |

For unit-capacity markets the load objective is `sum_j load_j^2`. It coordinates exposure, not assignment, and is not a guarantee of DA placement or welfare. The lexicographic objective is fixed before the new results; there is no outcome-tuned penalty. These are transparent optimization controls, not claimed reproductions of a published recommender whose information and interaction model differ.

Tie sensitivity: repeat the anchor for GPT and random under 20 deterministic, market-seeded permutations of seeker and posting indices. In this sensitivity only, random deletion uses a fixed per-person seeded ordering so earlier recipients' edits cannot alter a later recipient's random deletion. The main anchor reproduces the original sequential deletion RNG exactly. Average policy randomization within each market/seed block. Do not count permutations as independent data. Stratified sampling puts lower-wage seekers first, so index tie-breaking requires this explicit check.

## Outcomes and denominators

Report all seekers, lower/higher wage groups, and prespecified feasible-degree groups (0; 1–5; 6–10; 11 or more). Lower-wage status and degree are fixed before recommendation.

1. **Conditional coverage:** reference full-information DA posting shown / seekers placed by full-information DA. This matches the released jobs coverage endpoint.
2. **Unconditional coverage:** reference posting shown / all seekers. Retain this separately for comparison with the older rental paper; never silently mix denominators.
3. **Placement:** recipients / all seekers.
4. **Gross exits:** placed under comparator but not treatment, reported both per all seekers and per comparator recipient. Gross entries use the analogous definitions. Check `entries - exits = treatment placements - comparator placements` exactly.
5. **Individual outcome changes:** same posting, different posting while retained, and worse modeled rank, including unmatched as worse than every feasible option. Also report rank worsening among jointly matched recipients.
6. **Normalized rank cost:** feasible posting rank divided by feasible posting count, with zero for the best option and one for unmatched. This is an ordinal diagnostic under modeled preferences, not measured utility.
7. **Top-choice inclusion, displayed matching capacity, number of changed menus, and exact menu-budget preservation.**

Primary contrasts: Anchor + GPT minus GPT, Random, and Anchor + Random; Balanced + GPT minus Anchor + GPT. Report the model-informed controls and tie sensitivity alongside them. Access comparisons must include both own-policy recipients and the common random baseline.

## Inference

Compute counts within markets, pool within a seed block, then average the 20 block rates. Use paired, shared 10,000-draw seed-block bootstrap resamples within each window and variant; never bootstrap individual seekers or algorithm tie draws as independent observations. Display pointwise 95% intervals and a simultaneous familywise interval for the four primary contrasts across coverage and placement, if all endpoints are defined. Report unavailable denominators and block counts explicitly. Transition rates are descriptive safety endpoints: even one observed gross exit refutes perfect retention on the tested panel. Zero observed exits does not prove all-market protection.

No equivalence or noninferiority claim follows from an interval spanning zero. Report the coverage difference and interval directly; “preserves coverage” requires a stated tolerance and a corresponding test, not an insignificant p-value. No tolerance is selected after observing outcomes.

## Reproducibility and compute

Export an anonymous replay packet from the teammate's authorized data environment: only modeled rank arrays, feasibility, saved menu indices, synthetic groups, score ranks, and market metadata. No names, Census identifiers, posting IDs, employer identifiers or salaries. The packet remains an input artifact, not a public row-level data release. Store its SHA-256 and the original inputs' hashes in the run manifest.

Run allocation and optimization on Slurm CPUs. They do not benefit from reserving H200 GPUs or generating new LLM answers. Use scratch for temporary files and preserve source, logs, hashes and aggregate results in project storage. FASRC's netscratch retention changes to 60 days on 7 December 2026; see https://www.rc.fas.harvard.edu/blog/netscratch-retention-change/.

Before real evaluation: exhaustive tiny-instance objective checks; a regression counterexample where the anchor increases displayed capacity but reduces DA placement; empty-menu and repeated-slot checks; transition accounting; denominator and bootstrap tests; a synthetic end-to-end pipeline test. Synthetic validation is labeled separately and cannot be reported as a jobs empirical result.

## Current evidence status

As of this protocol version, GitHub publishes code and aggregate jobs tables, but not the jobs input extract or saved GPT menu cache. PR8 head is `7d5a1e99d753d76367d9fc8eb139f189a769f754`; no Actions artifacts or releases supplied those inputs when checked. The published compressed GPT caches are rental caches. `findings/jobs-gpt-frozen/README.md`, section “Reproduce,” explicitly requires the separate Revelio and ACS files. New empirical coverage/access estimates are therefore pending the replay packet or the authorized original inputs. Existing anchored-GPT minus anchored-random placement differences can be calculated from aggregate tables; their paired uncertainty cannot.
