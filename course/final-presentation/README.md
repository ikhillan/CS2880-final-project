# Final presentation

Start with [presentation.pdf](presentation.pdf). It has 16 slides, one empirical figure for each of the three settings, and a planned running time of 17 minutes.

- [Speaker notes](speaker-notes.pdf) give a full narration with checkpoints for each slide. The editable script is [presenter-guide.txt](presenter-guide.txt). Rehearse aloud before recording; the duration is a target rather than an audio measurement.
- [Technical appendix](technical-appendix.pdf) contains the detailed estimates, denominators, optimization formulation, hardness proof sketch, bound construction, institutional limitations and related work. It is separate from the timed talk.
- [Beamer source](main.tex) and [appendix source](technical-appendix.tex) remain editable. [make_figures.py](make_figures.py) generates the three empirical figures, a supporting retention-bound figure, and the new access tables. [figure-sources.json](figure-sources.json) records hashes for all 11 aggregate source files.

The problem statement is slide 3. Slide 4 states the findings before the methods and detailed evaluation. Slide 14 reports the completed rental access audit. Slide 15 discusses individual losses, constrained choices and recourse. The talk uses the [verified Chile results](../../findings/chile/README.md), the team's [NYC findings](../../findings/README.md), and the [new rental replay](../../findings/rental-anchor-access/README.md).

The main deck reports simulations and counterfactual replays. It does not present them as observed effects of deploying these recommenders. Coverage targets, baselines, retention guarantees and remaining limitations are visible in the slides and explained in the narration.

## Reading the three figures

- [Rentals](figures/rentals.pdf): GPT minus random across all three enriched-design windows. Both endpoints use the released 95% market-bootstrap intervals. Coverage is conditional on having a full-information match, while placement includes everyone. The repeated sign pattern is the result; these are simulated preferences and allocations.
- [Jobs](figures/jobs-anchor.pdf): original GPT, anchored GPT, and anchored random placement relative to plain random, with 95% seed-block intervals in both windows. Coordination recovers 13–16 percentage points for GPT, but coordinated random remains ahead by 2.8 and 2.0 points. Those between-anchor gaps are point estimates; the displayed intervals use plain random as comparator. The original jobs inputs are unavailable, so post-anchor coverage and individual transitions remain unidentified.
- [Chile](figures/chile-incidence.pdf): each historical group's net placement change divided by that group's own size. The group with no initial offer loses 0.873 percentage points, compared with 0.022 points among those with an offer. Its 76% share concerns the net loss across the linked sample; it is not the proportion of pupils harmed. The next slide includes the first-five comparator, which improves aggregate placement while still reducing this group's placement.

The main talk presents the retention bounds as a short table. The 642–4,620 interval remains wide. The appendix retains the prospective structural rule's gain of 28 historical hits and its limits. These results do not establish an optimal repair or a deployed social benefit.

## The completed access test

Slide 14 uses the original held-out enriched rental panel: 160 markets and 10,240 modeled renter appearances. Anchor versus GPT gains 4.0 placement points but causes 37 exits; balancing versus anchor gains another 12.6 points, loses 3.0 coverage points, and causes 171 exits. Balancing can replace more than one option per person. The appendix gives both designs and all windows, simultaneous intervals, exact flow formulation, model-informed controls, and outer bounds for the missing jobs transitions.

Renters with one to five feasible listings cannot have their menus changed under these fixed list lengths. Their 14.4-point placement gain under balancing therefore illustrates an effect of other people's menus. None of these transitions is an observed housing outcome. The follow-up is post hoc on reused windows; the original `heldout_primary` name does not make the new policies independently validated.

To rebuild from this folder, with Python, Matplotlib, NumPy and a TeX distribution containing Beamer:

```sh
python3 build.py
```

This regenerates the figures and all three PDFs. Compilation stops on errors or overflowing boxes. The intermediate `.build/` folder and page renders stay out of version control.
