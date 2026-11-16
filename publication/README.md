# Who Keeps an Offer? Predicting Partners and Choosing Shortlists in Matching Markets

Harvard CS 2880, AI for Social Impact. EC-oriented research draft, November 13, 2026. Build from the archive root with latexmk -pdf -halt-on-error main.tex. No network access is required. The manuscript uses the most recent EC single-column submission kit verified for this study (2026). Check the 2027 call before submitting. This is the team's working archive, with experiment and compute provenance. Review its metadata separately for double-blind external sharing.

## Which results are new in this version?

The paper now tests an explicit directional hypothesis on two subsequent, previously unexamined Chilean school cohorts. Models fitted on 2019 were locked before retrieving the 2022 and 2023 application files. Use research/policy/equity for the new cohorts, actual offer acceptance, priority-designation spillovers, frozen protocol and independent implementation of reserved-seat sensitivity. The original 2020/21 comparison is exploratory relative to this confirmation.

We report historical university enrollment and train six more models on that target. Adding submitted rank to the enrollment-trained network covered 10,891 more actual 2026 destinations and placed 141 fewer applicants in the fixed-report replay. Public files from Chile's Ministry of Education allow local linkage of 136,751 regular applicants to a recorded registration in 2026. Reported destinations supply coverage outcomes; the placement effects of the proposed policy continue to require a replay. Source checksums and aggregate linkage results are in research/policy/enrollment. The published archive excludes every participant-level administrative row, join key, feature panel and prediction from the field studies. The simulated NYC panels are in the archive.

Before tentative offers, research/policy/frontier gives the national minimum-cost flow, exact integer dual checks, alternative upper capacity assumptions, and plotting code. research/policy/theory contains the proof of the capacity frontier, exhaustive small-instance checks, responsive-seat extensions and results for 2,304 simulated housing markets. After tentative offers, research/policy/design describes the maximum residual referral rule, take-up, forecast errors, affordable-stock constraints, alternatives, Hall witnesses, rank limits and when fixing every provisional contract costs more potential placements than permitting strictly preferred moves. Each component includes its own commands and versions. New modeling and empirical work, including the policy design, proofs, test protocols and manuscript revisions, used OpenAI Codex extensively. This working archive is intended for the team and has not been submitted.

## New observed-preference tests

The main paper and its data appendix now report four settings with real choice information. The primary five-option model vs uniform placement inversion in rental simulation did not reproduce in Chilean university admissions, the two K–12 test cohorts or the one held-out mentoring cohort. The small speed-dating study was inconclusive. Within admissions, stronger comparators produce order reversals. For example, in the same 2026 university cohort the rank-informed offer-trained model covers 11,256 more historical offers than its equal-size rank-blind model while placing 267 fewer people in counterfactual DA. Six H200 fits used official regular offers from 2025, with individual test outcomes excluded from the inference inputs.

The new reproducibility materials are under research/observed. Each subdirectory contains a study-specific protocol, evaluation code, primary-source acquisition instructions and aggregate results. University includes the six frozen network checkpoints. School compares modeled initial assignments with administrative offers on complete national cohorts. Ordinal tests submitted bilateral mentoring ranks; reciprocal tests pre-event profiles against observed mutual choices. Capacity verifies the rental display-capacity decomposition against the earlier simulation archive. Underlying field records, administrative identifiers and individual-level neural predictions are excluded from the distributable package. Earlier sections of the original housing paper and its replay remain below. These field studies do not establish the causal effect of actually deploying these policies or of changing how people report their preferences.

## Read the evidence

All housing placements and rankings are simulated. Asking rents and recent-mover household inputs are based on public sources. The 2,304 test markets reuse household donors; lower-income households with no feasible unit remain in ordinary-market placement denominators. The 835,584 conditional Monte Carlo draws in the precision comparison reuse 384 observed feature panels. They do not add boroughs, households, or months. The split designs and follow-ups are dated in the corresponding protocol files; refinements in response to results are identified in the manuscript.

| Question | Evidence in the archive |
|---|---|
| Does an accurate shortlist lead to a placement? | research/ai/learned, including nine checkpoints, frozen predictions, training traces, splits, counts, paired contrasts, capacity control and CPU replay |
| How closely does the model approximate the simulator's partner probabilities? | research/ai/mc-teacher, including the H200 scripts, original and follow-up protocols, assignments, precision counts, seed keys and receipts |
| Does a renter keep their tentative placement, rank, or rent level after competitors' lists change? | research/ai/proof-audit, including theorem checks, market-level results, ordered-partner outcomes and affordability checks |
| How sensitive are policy results to listing priorities? | research/ai/published-baseline, including the fixed-menu paired sensitivity, serial DA verification, original WWW benchmark and housing adaptation |
| How often could an unguarded policy be tried? | research/ai/risk, with the finite-sample bound, frozen rollout protocol and all risk targets |
| Where does the 27.78% floor come from? | research/ec/welfare, with feasible assignments, threshold counts, flow and LP certificates; refers to 217 lower-income placements in 240 simulation markets |

## Replay from the saved panels

The archive includes the numeric panels, frozen scores and benchmark arrays used for the reported results. Panels are integer-indexed simulator outputs. They contain no names, addresses or donor identifiers. Python 3.11+ with NumPy, SciPy and pandas is sufficient for the main assignment replays. PyTorch is required to fit the networks or regenerate their scores. Matplotlib is needed for figures, C++17 for the omission verifier. See individual component READMEs for tested versions. The archive root contains the PDF, LaTeX and pre-rendered figures.

~~~sh
shasum -a 256 -c MANIFEST.sha256 > /dev/null
mkdir -p replay/data replay/counts replay/guard replay/risk
A=research/ai
cp "$A"/panels/* replay/data/
cp "$A"/learned/predictions/*.npz replay/data/
cp "$A"/learned/receipts/*scores.json replay/data/
python -B "$A/learned/evaluate.py" --data replay/data --out replay/counts
OPENBLAS_NUM_THREADS=1 python -B "$A/learned/summarize.py" --data replay/data --counts replay/counts
python -B "$A/proof-audit/evaluate_guardrail.py" --base replay/data --out replay/guard
python -B "$A/proof-audit/stayer_welfare.py" --base replay/data --out replay/guard
python -B "$A/proof-audit/rent_veto.py" --base replay/data --out replay/guard
python -B "$A/risk/normalize.py" replay/counts replay/risk/normalized.csv.gz
python -B "$A/risk/calibrate.py" replay/risk/normalized.csv.gz replay/data/cal.json replay/risk/results.csv
python -B "$A/mc-teacher/evaluate_mc.py" "$A/learned" replay/data "$A/mc-teacher/draw-partners.npz" "$A/mc-teacher/subset-keys.json" replay/mc-counts.csv.gz --precision "$A/mc-teacher/precision-counts.npz"
c++ -std=c++17 -O2 "$A/mc-teacher/omission.cpp" -o replay/check-omission
replay/check-omission
python -B figures/accuracy_placement.py replay/counts/contrasts.csv.gz replay/accuracy_placement.pdf
~~~

To regenerate the conditional-simulation inputs use subset.py replay/data replay/inputs.npz. The GPU runners take that file and an output .npz filename. To regenerate a frozen network's predictions, run research/ai/learned/train.py infer --data replay/data --out research/ai/learned/models --device cuda. The reported scores are from H200 CUDA. The CPU portability experiment quantifies how floating-point differences near ties affect particular renters. Replaying DA on the supplied scores gives the reported assignments.

To rebuild panels from the underlying public sources, follow BASELINE_REPLAY.md and MATCHING_REPLAY.md for inputs, cleaning, household donor splits and the three winter CSVs. From the archive root the panel builder accepts:

~~~sh
python -B research/ai/learned/prepare.py --root . --repo research/implementation --summer-raw nyc-inputs/data/firstmover --winter-raw replay/winter-inputs --winter-script research/ec/winter/run.py --out replay/regenerated
~~~

The authors' original two-sided recommender is obtained from the pinned public release in research/ai/published-baseline/README.md. Synthetic market input generators, previous experiments, and per-market evidence for the interference and burden analyses are included in the earlier replay guides.

## Research provenance

The team proposed studying conflicts in intermediation, planned the housing experiments, implemented the rental simulator and reported the scarcity pilot. OpenAI Codex (GPT-6 family) was used extensively for subsequent theory, counterexamples, benchmark implementation, statistical analysis, manuscript drafting and revision. In particular, the team asked to keep AI for Social Impact central, test a trained mechanism, expand the proofs, reproduce published comparisons and use kkl accelerators. Follow-up computational work checked the reduction and welfare guarantees, compared simulation and learned scores, isolated model capacity, reran alternative listing priorities and traced individual rank and rent changes. The class project initially proposed intermediary friendship as a mechanism. These data do not measure broker relationships or identify commission-driven steering. The dated working draft is available for the team's review and has not been submitted.
