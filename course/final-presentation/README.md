# Final presentation

Start with [presentation.pdf](presentation.pdf). It has 16 slides, three result figures, and a planned running time of 17 minutes.

- [Speaker notes](speaker-notes.pdf) give a full narration with checkpoints for each slide. The editable script is [presenter-guide.txt](presenter-guide.txt). Rehearse aloud before recording; the duration is a target rather than an audio measurement.
- [Technical appendix](technical-appendix.pdf) contains the detailed estimates, denominators, optimization formulation, hardness proof sketch, bound construction, institutional limitations and related work. It is separate from the timed talk.
- [Beamer source](main.tex) and [appendix source](technical-appendix.tex) remain editable. [make_figures.py](make_figures.py) generates the three vector figures from committed aggregate tables. [figure-sources.json](figure-sources.json) records hashes for the tables it consumes.

The problem statement is slide 3. Slide 4 states the findings before the methods and detailed evaluation. Slide 15 discusses exclusion, historical labels and recourse. The talk uses the [verified Chile results](../../findings/chile/README.md) and the team's [NYC findings](../../findings/README.md).

The main deck reports simulations and counterfactual replays. It does not present them as observed effects of deploying these recommenders. Coverage targets, baselines, retention guarantees and remaining limitations are visible in the slides and explained in the narration.

To rebuild from this folder, with Python, Matplotlib, NumPy and a TeX distribution containing Beamer:

```sh
python3 build.py
```

This regenerates the figures and all three PDFs. Compilation stops on errors or overflowing boxes. The intermediate `.build/` folder and page renders stay out of version control.
