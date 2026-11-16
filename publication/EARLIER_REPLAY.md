# The Cost of Protecting Applicants When Hidden Matches Become Visible

Open **IJCAI_2027_MATCHING_PAPER.pdf** for the anonymous conference-format manuscript. To change the argument, edit **main.tex**. The experiments are in **experiments.tex**, with the earlier-cohort replication in **replication.tex**. The bibliography is **paper.bib**. The unchanged IJCAI style and bibliography format are included with the author-kit download URL and checksums.

## Build the paper

Extract the archive before running these commands. Use a TeX distribution with BibTeX, latexmk and the packages in the preamble. The chart is included as a vector PDF, so compiling the paper requires no data downloads, Python packages or network access.

~~~sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
~~~

## What the comparisons mean

An audit opens the feasible edges incident to a firm's clients. A successful repair reruns renter-proposing deferred acceptance on the newly available edges and checks for blockers in the full graph. A group's premium is the difference between the minimum audit costs with and without that group's avoidable-loss protection. Zero premium says a cheapest plan can meet the thresholds. The price of the plan itself can be substantial.

The NYC advertisements supply asking prices, bedroom counts and listing sources. Census households supply incomes, household sizes and borough weights. A simulation supplies the priorities, renter rankings, visible lists and assignments. All prices in the matching comparisons are July 2026 dollars. Each NYC market has 64 renters and 64 listings. Development and replication share calibration households and ranking assumptions. The January–March replication uses different listing cohorts and random draws. The written protocol predates that local replication run. The simulation holds prices fixed and cannot establish discrimination, realized welfare, a firm's commission motive or the effect of any current housing platform.

## Evidence and replay

Run the commands below **from the archive root**. The paper's experiments used Python 3.13.6, NumPy 2.2.6, SciPy 1.17.1 and a single HiGHS thread for the exact optimizer. Install NumPy and SciPy from synthetic/requirements.txt in a separate environment. Input preparation otherwise uses the standard library. The optional independent raw data check additionally needs pandas. Use the original versions for matching input checksums and reproducing the recorded simulation outputs. Runtime estimates depend on hardware and solver build.

### Structural results

~~~sh
mkdir -p replay
python -B artifact/degree-boundary/verify.py > replay/degree.json
python -B artifact/one-blocker/verify.py > replay/one-blocker.json
python -B sparse-ring/verify.py --artifact synthetic > replay/ring.json
~~~

The exhaustive one-blocker run checks every strict profile on every labeled three-by-three graph of maximum degree two. Its results.json records a nonuniform audit-cost vector, counts and an imperfect-baseline example. The ring verifier also checks all audit masks at the reported small sizes. These computations supplement the general proofs.

### Exact optimization

~~~sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B artifact/protection/run_protected.py --archive synthetic --out replay/protected.json
~~~

The 160 input instances, ordinary enumeration results, generating code, design grid and reference protected-solver logs are included. The protected solver checks its optima against independent DA and enumeration. It uses three paired timing repeats, so new wall times will differ. The design and independent checks are documented in artifact/protection/README.md.

### NYC catalogue preparation and development study

~~~sh
python -B nyc-inputs/fetch_inputs.py --data nyc-inputs/data
python -B nyc-inputs/prepare_acs.py --directory nyc-inputs/data/acs
python -B nyc-inputs/scarcity.py --data nyc-inputs/data --out replay/scarcity
python -B artifact/nyc-matching-dev/run.py --data nyc-inputs/data --catalogue-module nyc-inputs/scarcity.py --out replay/matching.json
python -B artifact/spillovers/replay.py --data nyc-inputs/data --catalogue-module nyc-inputs/scarcity.py --reference artifact/nyc-matching-dev/results.json --out replay/spillovers.json
python -B artifact/nyc-costs/run.py --data nyc-inputs/data --catalogue-module nyc-inputs/scarcity.py --out replay/nyc-costs.json
python -B artifact/anchored/run.py --data nyc-inputs/data --catalogue-module nyc-inputs/scarcity.py --out replay/shortlists.json
~~~

Each runner checks input fingerprints against the recorded public snapshots. The included JSON and CSV files give aggregate results without requiring raw listing downloads. In particular, artifact/nyc-costs/results.json includes local-index witnesses, audit sets and optimality evidence for all ten markets that failed the two-run screen. An optional independent analysis of the raw Census and listing data can be run with python nyc-inputs/verify_scipy.py after the preparation steps. This checks the included archived cell estimates.

### January–March replication and figure

The three earlier FirstMover monthly files come from the same public project. Their URLs and SHA-256 values are in artifact/winter-replication/sources.json. Download them into replay/winter with the command below, then rerun the frozen comparison. The program verifies data and protocol checksums. The official BLS snapshot used by this run is included.

~~~sh
mkdir -p replay/winter
python -c "import json,pathlib,urllib.request; target=pathlib.Path('replay/winter'); sources=json.loads(pathlib.Path('artifact/winter-replication/sources.json').read_text()); [urllib.request.urlretrieve(s['url'],target/k) for k,s in sources.items() if k.endswith('.csv')]"
OPENBLAS_NUM_THREADS=1 python -B artifact/winter-replication/run.py --data nyc-inputs/data --winter replay/winter --catalogue-module nyc-inputs/scarcity.py --out replay/winter.json
python -B figures/spillover.py
~~~

Figure 1 reads the two recorded spillover JSON files directly and requires Matplotlib to rebuild. The final paper and figure are included already. The manuscript averages displayed rents among positive-degree renters within each cell and then uses common cell weights in both periods. The aggregate files also retain the earlier pooled conditional estimate. The source-data links, data exclusions, full component counts and interval definitions are in the individual experiment directories. The public September and October listing files and the reserved donor split were not used as evaluation outcomes in this paper.

## Research and writing record

The research question and housing motivation came from the project team. Generative AI contributed substantially to the manuscript, proofs, code, simulations and analysis in this package. No human-participant outcomes were collected in these experiments. The archive is a team-review copy. It contains no participant contact details or renter names. All NYC matching outputs use simulation-local indices or aggregate counts. The manuscript uses the anonymous main-track format.
