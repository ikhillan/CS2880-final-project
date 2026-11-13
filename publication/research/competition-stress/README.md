# Competition stress test on 100 fresh seed blocks

The contrast changes only the higher-income renters' lists from uniformly drawn feasible apartments to the team's transparent score. Every lower-income shortlist stays fixed. The outcome is the percentage-point change in the provisional renter-proposing DA match rate among all lower-income renter draws, including those with zero options. The comparison uses the same three summer 2026 catalogues and calibration household pool as the development pilot, with 100 new independently seeded blocks. No new temporal cohort, evaluated landlord behavior, AI replies or leases enter this experiment.

At the selected price/noise setting (4, 0.5), the ordinary five-listing contrast was −0.31 points and the enriched five-listing contrast was +9.03 points. For lists containing 40% of feasible options rounded up, these contrasts were −0.26 and −1.33 points. The simultaneous 95% simulation bands over all 36 contrasts excluded zero at the selected setting in all four panels. Across all nine settings, the enriched five-listing contrast ranged from +7.63 to +9.35 points, always positive, whereas the ordinary five-listing contrast changed sign. The same price penalty is used in both the simulated renter rankings and the transparent score. CROSSED-DESIGN.md documents an additional exploratory decomposition that crosses those two choices at fixed private noise 0.5.

## Inputs and denominator

prepare.py uses the feature/protocol implementation, base commit 981ed97 in the shared team repository. It requires the three cleaned June–August listing catalogues, the project's prepared renter cache, coverage table, and CPI series. The corresponding raw and derived inputs are fingerprinted in source-inputs.json and provenance.json. The patched branch available in the project includes these public-data preparation instructions. Geographic weights are the same prespecified aggregate ACS recent-mover shares as the pilot. Only calibration donors enter the market outcomes. Individual records or addresses are not included in these results.

Each draw comprises 64 renters and 64 listings. The four supported boroughs and three months give 12 cells. Two market designs give 24 draws per seed. The 2,400 distinct eligible graphs are reused across nine settings and six arm/quota combinations for 129,600 DA executions. The ordinary and enriched draws are distinct. The same available market, outside option, uniform order, landlord priority and shocks are held fixed for all arms within a design, cell and seed. A uniform order is pinned to the base ID (4, 0.5) across the preference grid; the original fallback order changes with a change in the market ID. In both arms each renter sees exactly the same number of listings.

## Files and replay

DESIGN.md records the original analysis choices. The first seed is 2026111300. prepare.py produces compressed anonymous integer ranks and metadata; the archive is reproducible from the public data and is omitted here to keep the paper bundle small. The SHA-256 in ranks-metadata.json is the exact fingerprint used by the H200 run. The emitted columns and axis ordering are determined by prepare.py and run_gpu.py. The branch's selected point coincides with its original uniform-exposure convention.

Run these commands after preparing the team's development checkout. Replace the repo path with its root. To rerun under Slurm, run.sbatch expects the rank arrays, metadata and GPU script in the submission directory and an installed PyTorch environment; adjust the environment path and account for your cluster. Standalone commands work without Slurm.

~~~sh
python prepare.py --repo /path/to/CS2880-final-project --out ranks.npz --workers 4
python run_gpu.py --ranks ranks.npz --out primary-output --device cuda
OPENBLAS_NUM_THREADS=1 python summarize.py --ranks-meta ranks.json --counts primary-output/counts.csv.gz --out results.json
python run_gpu.py --ranks ranks.npz --out crossed-output --device cuda --cross-prices
OPENBLAS_NUM_THREADS=1 python summarize.py --ranks-meta ranks.json --counts crossed-output/counts.csv.gz --out crossed-results.json
~~~

The primary GPU job on the kkl NVIDIA H200 used torch 2.9.0 with CUDA 12.8. It checked the absence of visible blocking pairs and feasibility on all 129,600 allocations. prepare.py separately generated reference renter assignments with the repository's serial DA for every arm/grid/cell/design in the first block. All 1,296 of those complete renter assignment vectors matched exactly. A separately implemented serial C++ DA replay agreed with every aggregate count on all 2,400 graphs. compute.json records compute details. counts.csv.gz provides the primary seed-by-cell integer counts.

summarize.py pools renter numerators and denominators across seeds within borough-month, then standardizes the cell ratios with the same weights for both groups, arms and preference settings. Its 10,000 bootstraps resample entire seed blocks. The 36-effect simultaneous band uses the 95th percentile of the maximum absolute centered bootstrap effect divided by its simulation standard deviation. Values in results.json are in percentage points for contrasts and percentages for levels. The q40 budget reflects 40% of each person's options. Optimizer weights and effect uncertainty remain conditional on the behavioral model and original input preparation.

## Exploratory mechanism check

In the ordinary five-listing market, hold renter preferences at the selected price penalty of 4 and noise 0.5. Moving the transparent score's price penalty from 1 to 2 to 4 changes the spillover from +1.09 to +0.49 to −0.31 percentage points. Every fixed renter price weight in the crossed grid gives a positive spillover at score penalty 1 and a negative spillover at score penalty 4. All six corresponding simultaneous intervals exclude zero. The enriched top-five effect stays positive at all nine crosses, from +3.79 to +13.72 points. In enriched proportional markets with renter price 4, the effect moves from +5.00 to +2.72 to −1.33 points. This check leaves renters' rankings and all listings' priorities fixed when changing the score. The two sets of 129,600 GPU executions reuse 2,400 draws and include exactly repeated diagonal cases at noise 0.5. Treat the crossed price analysis as a post-primary mechanism check on those draws.

The crossed output and its H200 validation receipt reside in crossed/counts.csv.gz, crossed/results.json and crossed/compute.json. An independent serial check with a C++17 compiler can reproduce every cell's matched, blocked and transition counts for the entire archive:

~~~sh
python audit_counts.py --ranks ranks.npz --counts counts.csv.gz
python audit_counts.py --ranks ranks.npz --counts crossed/counts.csv.gz --crossed
~~~
