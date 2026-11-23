# What this talk can honestly claim

Reviewed 22 November 2026 against the aggregate evidence and the current proof statements. This review concerns the presentation. The paper and experiments were not edited.

## Overall judgment

This can be a strong research-course talk. The current evidence does not justify describing it as an award-winning breakthrough, a universally superior AI system, or measured social benefit. The best contribution is a precise distinction: coordinated recommendations can improve aggregate outcomes while changing individual access, including for people whose own complete menus never change. The finite-panel retention problem and the gap between testing and proof give that empirical observation a mathematical counterpart.

The earlier deck was accurate in many details but fragmented. It moved from rentals to jobs to Chile to theory and back to rentals. It also left the prospective guarantee in the appendix. The revision keeps one argument throughout and gives both the theoretical promise and its modest realized gain space in the main talk.

## Findings worth emphasizing

- **Unchanged-shortlist spillovers:** in the September–October enriched rental replay, placement rises from 13.12% to 27.53% for renters with one to five feasible options under balancing versus the anchor. The main talk rounds these to **13.1% and 27.5%**. Their own shortlists cannot change. This is a large effect within the specified simulation, not an estimated field effect. The subgroup has 2,328 modeled appearances and also has 15 exits.
- **Individual protection differs from a higher total:** anchoring yields 449 entrants and 37 exits relative to GPT, or 412 net additional placements among 10,240 modeled appearances. The 37 exits affect about 0.57% of original recipients. Balancing causes 171 exits relative to the anchor despite larger total gains.
- **The comparator matters:** balanced menus improve both placement and reference coverage over plain random; Chile's rank-informed menus improve both aggregate outcomes over first-five. The evidence does not imply an unavoidable prediction–placement tradeoff.
- **A coupled design problem:** optimizing coverage with retention across an explicit priority panel is NP-complete in the stated fixed-priority model, although checking a chosen menu policy with deferred acceptance is polynomial. The reduction and assumptions belong in the appendix as well as the spoken explanation.
- **A test panel is not a guarantee:** the prospective structural Chile rule gains 28 historical hits under its all-order guarantee. The learned rule gains 50 but has exits on fresh draws. The structural result is mathematically meaningful and quantitatively modest; its reported paired test panels have no new entrants.

## What is not a new breakthrough on its own

Recommendation congestion, gross-transition accounting, maximum matching, and convex-load min-cost flow are established ideas. Their use here should be described as methods and controls. A very large computation count, a model name, or millions of administrative rows does not establish novelty or causal validity.

The stronger score controls and coordinated random results prevent a general claim of LLM superiority. The main narration explicitly acknowledges the stronger model-informed score control; detailed jobs comparisons stay in the appendix. The social-impact claim is an evaluation and design lesson about access. It is not evidence that actual households or families benefited from receiving these recommendations.

## Decisions in the revised talk

The main deck has 18 slides and approximately 17 minutes of narration. Slide 4 states the 13.1% to 27.5% finding immediately after the problem definition. The rental story continues through slide 10; slides 11–12 introduce the Chilean distributional question. Slide 13 states the mathematical problem and theorem, slide 14 explains the proof intuition, slide 15 gives the hindsight bounds, and slide 16 presents the prospective test. Slide 17 addresses ethics and exclusion; slide 18 closes. Three empirical figures appear in the timed talk.

There is no dedicated limits slide. The main discussion still identifies modeled and counterfactual outcomes, reused calendar windows, unequal edit budgets, the stronger score control, losses hidden by net gains, the wide optimization bracket, and the small prospective guarantee. These facts change the interpretation and cannot be omitted to make the results look stronger. Detailed jobs results, secondary panels, full controls, uncertainty settings, mechanism assumptions and proof details remain in the appendix.

Ishan's newest findings were already included. His older physical-unit deduplication work exposes a limit worth stating: rental calendar windows have property overlap. No input catalogs or cached experiments were changed. See LATEST_GITHUB_REVIEW.md for verified commits and counts.

The narrated MP4 is included for team sharing. Its voice is synthetic and identified as such; the speech model and intermediate assets remain ignored. A student recording can use the same script. The generated narration is not a recording of any named author, and its measured duration is recorded in the narration manifest.
