# Who Loses a Match? Individual and Collective Effects of Shortlists

**EC-oriented working paper, November 13, 2026.** The anonymous PDF is a working draft. No submission or registration has been made. We verified the official EC 2026 call and its accompanying single-column style. We could not verify official EC 2027 submission instructions on this date. The bundled class, style and bibliography format are taken from the last verified EC kit, with its original files preserved. A blank first-page footer avoids claiming submission to a past conference. Venue-specific dates, formatting and the submission number should be checked against the 2027 call when it is available. [EC call and linked style kit](https://ec26.sigecom.org/call-for-contributions-acm/papers/)

## Paper and evidence

Build the paper from the extracted archive root with **latexmk -pdf -halt-on-error main.tex**. A normal full TeX Live with the ACM fonts and BibTeX is sufficient. The figures are included, and compiling requires no raw data or network access. The LaTeX main text presents the labeled-menu result, representation size, nonadditivity, randomized adoption, sharp one-slot bound, summer/winter counterfactuals and the cost of stability for lower-income burden. The appendix gives the low-saturation bound, full audit-cost proofs, solver checks and further controls.

| New component | Evidence directory |
|---|---|
| Coalition restriction, attainable listings, displacement labels, worst-case encoding and saturation | research/ec/theory-spillover |
| Sharp one-slot interactions and three-editor example | research/ec/interaction-example |
| Exact individualized effects in 2,400 summer markets and stratified simulated rollout | research/ec/marginal |
| Fixed-preference winter price-score transfer, with cohort exclusions and pointwise/simultaneous uncertainty | research/ec/winter |
| Exact burden minima at increasing levels of protection, stable extrema, witnesses and LP replay | research/ec/welfare |

The original calibrated generator, summer controls, earlier degree-boundary experiments and detailed input preparation are also included. BASELINE_REPLAY.md describes the earlier implementation and archived datasets. Its old manuscript build command refers to that prior paper. The current build uses the root main.tex. The research record and experimental methods disclose substantive AI assistance with theory, analysis, code and writing. This source archive includes internal provenance useful to the team. Its metadata would need separate anonymization for double-blind external sharing.

## Reproduce

The housing replays use Python 3.13.6, NumPy 2.2.6, pandas 2.3.3 and SciPy 1.17.1. New experiments use CPU code. The prior summer grids used NVIDIA H200 jobs and have independent C++/CPU verification. PyTorch is needed only to rerun those CUDA routines. Individual-displacement enumeration and rollout use a C++17 compiler. Source files, designs and statistical estimands are documented in each folder. No LLM calls or private brokerage records are needed.

1. Run the official-input download and calibrated development preparation in BASELINE_REPLAY.md. These create the summer listing files, required census cache and summer-fitted score features under research/implementation.
2. Rebuild the base summer rank tensor from the archive root. The following commands make the two new summer menu outputs and graphic. Use only one BLAS thread for the bootstrap linear algebra.

~~~sh
mkdir -p replay
OPENBLAS_NUM_THREADS=1 python -B research/competition-stress/prepare.py --repo research/implementation --out replay/ranks.npz --workers 4
OPENBLAS_NUM_THREADS=1 python -B research/ec/marginal/run.py --ranks replay/ranks.npz --meta replay/ranks.json --gpu-counts research/competition-stress/crossed/counts.csv.gz --out replay/menus
OPENBLAS_NUM_THREADS=1 python -B research/ec/marginal/rollout.py --ranks replay/ranks.npz --meta replay/ranks.json --menus replay/menus/counts.csv.gz --out replay/menus
python -B figures/rollout.py replay/menus replay/rollout.pdf
~~~

3. Run the welfare comparisons and their independent checks on the prepared calibrated generator.

~~~sh
OPENBLAS_NUM_THREADS=1 python -B research/ec/welfare/check.py --repo research/implementation --out replay/welfare
OPENBLAS_NUM_THREADS=1 python -B research/ec/welfare/verify.py --repo research/implementation --evidence replay/welfare --out replay/welfare/verification.json
OPENBLAS_NUM_THREADS=1 python -B research/ec/welfare/grid.py --repo research/implementation --calibration research/pilot/calibration.json --out replay/welfare/grid_results.json
~~~

4. For the calibrated January–March experiment, use the three measured, checksum-pinned winter files listed in research/ec/winter/sources.json and the three summer files already downloaded. Keep winter and summer raw CSV directories separate. The winter script uses summer reference features and calibration profiles. The former simplest-model winter run and the calibrated winter replication use different cleaning, rankings and studies.

~~~sh
OPENBLAS_NUM_THREADS=1 python -B research/ec/winter/run.py --repo research/implementation --summer-raw nyc-inputs/data/firstmover --winter-raw replay/winter-inputs --out replay/winter
python -B research/ec/theory-spillover/verify.py --out replay/small-theory.json
python -B research/ec/interaction-example/verify.py --out replay/interaction-example.json
~~~

The graph-order and stability bounds are finite-market certificates. Policy effects and bootstrap intervals are conditional on simulated rankings, survey and catalogue inputs; they do not estimate actual lease responses. The primary fresh summer and price-crossing studies reused the same market draws. The winter hypotheses were locally frozen before computing that policy comparison; winter dates had already been used for an earlier experiment with a different policy. The official public inputs are pinned in the corresponding manifests, and no raw household identifiers, addresses or private communications are packaged.
