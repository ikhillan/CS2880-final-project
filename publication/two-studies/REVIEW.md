# Build the anonymous review copy

The working paper and the review copy use identical manuscript text, including the AI-use appendix. The review copy uses the EC 2026 class and style supplied by the conference. The last official call we could verify on November 13, 2026 was EC 2026. Its 18-page main-text allowance excludes the title page and bibliography and permits an appendix. The review copy omits the old conference's submission footer and leaves the submission number unassigned. [SIGecom events](https://sigecom.hosting.acm.org/events.html) · [EC 2026 call and style kit](https://ec26.sigecom.org/call-for-contributions-acm/papers/)

From a repository checkout, run this in the paper directory. In the standalone source distribution the class, style, and bibliography style are also at the top level.

```sh
TEXINPUTS=.:..: BSTINPUTS=.:..: latexmk -pdf -halt-on-error review.tex
```

The cover says “Anonymous review draft.” The same source can accept an assigned number using the macro `\ECPaperNumber` before `\input{review.tex}`. Check the target year's call when preparing an actual submission. The named source archive contains bylines, branch links and project history. The anonymous output is the review PDF.

## AI-assisted work

The group proposed the intermediation question, planned and implemented the original housing simulation, and reported the scarcity of affordable listings in its pilot. OpenAI Codex (GPT-6 family) assisted extensively with later proof construction and counterexamples, implementation and verification of model comparisons, analysis of the public cohorts, and drafting and revision of the manuscript. The project record identifies the model family. It does not preserve a per-turn checkpoint identifier for earlier work.

Representative research requests included “Can we test whether the prediction-versus-placement tradeoff holds on a real matching dataset with observed preferences, then fold that evidence into the paper?” and “Make sure the theory, experiments, and NYC–Chile connection are clear and defensible.” Revisions narrowed the school claims to the historical labels and specified replays, tested alternative pursuit orders, and traced the changing assignments. Program outputs and independent implementations support the reported numerical claims. The latest revision also includes AI-assisted algorithm design, new reductions, exhaustive checks and evaluation of risk under held-out simulated lotteries.

The model in the rental experiment was a separate system. The saved batch records use Anthropic's model identifier `claude-haiku-5-5`. The prompt asked the model to pick five apartment listings from the renter's feasible candidates using the presented budget, household needs, target area and listing attributes. Exact selections were reused throughout the replays. The pilot records supply the prompt template and response indices, with the cache-authentication limitation described in the paper.

The institutional appendix reports the four-year D1 fit, the reconstructed menu comparisons, the allocation rules and the finite lottery certificates. Only aggregate national results are distributed.
