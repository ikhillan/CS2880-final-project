# Final presentation

Start with [presentation.pdf](presentation.pdf). It has 16 slides, one empirical figure for each of the three settings, and a planned running time of 17 minutes.

- [Speaker notes](speaker-notes.pdf) give a full narration with checkpoints for each slide. The editable script is [presenter-guide.txt](presenter-guide.txt). Rehearse aloud before recording; the duration is a target rather than an audio measurement.
- [Technical appendix](technical-appendix.pdf) contains the detailed estimates, denominators, optimization formulation, hardness proof sketch, bound construction, institutional limitations and related work. It is separate from the timed talk.
- [Beamer source](main.tex) and [appendix source](technical-appendix.tex) remain editable. [make_figures.py](make_figures.py) generates the three empirical figures and a supporting retention-bound figure as vector PDFs and PNGs. [figure-sources.json](figure-sources.json) records hashes for its seven aggregate source files.

The problem statement is slide 3. Slide 4 states the findings before the methods and detailed evaluation. Slide 15 discusses exclusion, historical labels and recourse. The talk uses the [verified Chile results](../../findings/chile/README.md) and the team's [NYC findings](../../findings/README.md).

The main deck reports simulations and counterfactual replays. It does not present them as observed effects of deploying these recommenders. Coverage targets, baselines, retention guarantees and remaining limitations are visible in the slides and explained in the narration.

## Reading the three figures

- [Rentals](figures/rentals.pdf): GPT minus random across all three enriched-design windows. Both endpoints use the released 95% market-bootstrap intervals. Coverage is conditional on having a full-information match, while placement includes everyone. The repeated sign pattern is the result; these are simulated preferences and allocations.
- [Jobs](figures/jobs-anchor.pdf): original and anchored GPT placement relative to random, with 95% seed-block intervals in both windows. Coordination recovers 13–16 percentage points, leaving a smaller negative gap. The released anchor table does not report coverage, rank changes or gross exits, so it does not establish that the repair preserves relevance or individual access.
- [Chile](figures/chile-incidence.pdf): each historical group's net placement change divided by that group's own size. The group with no initial offer loses 0.873 percentage points, compared with 0.022 points among those with an offer. Its 76% share concerns the net loss across the linked sample; it is not the proportion of pupils harmed. The next slide includes the first-five comparator, which improves aggregate placement while still reducing this group's placement.

The main talk presents the retention bounds as a short table. The 642–4,620 interval remains wide, and the prospective structural gain is 28 historical hits. These results motivate further optimization and access tests. They do not establish an optimal repair or a deployed social benefit.

To rebuild from this folder, with Python, Matplotlib, NumPy and a TeX distribution containing Beamer:

```sh
python3 build.py
```

This regenerates the figures and all three PDFs. Compilation stops on errors or overflowing boxes. The intermediate `.build/` folder and page renders stay out of version control.
