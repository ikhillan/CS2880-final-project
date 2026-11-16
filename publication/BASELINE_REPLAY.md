# What Can a Shortlist Repair?

This is the editable source and team reproducibility archive for the revised IJCAI-format paper. The included PDF is anonymous and line-numbered. The source archive contains detailed provenance and is intended for the research team. Public data releases and study code underlie the housing experiments. Utility scores, listing priorities, referrals, assignments and audit costs are modeled. Generative AI contributed substantially to writing, proofs, code and analysis.

## Build and inspect

Extract the whole ZIP. From its root, run `latexmk -pdf -halt-on-error main.tex`. The vector figure is already included. No network or GPU is needed to compile. The original IJCAI style and bibliography format are included, with checksums in template-provenance.json. MANIFEST.sha256 covers the published archive files.

## Choose the evidence

| Claim | Recorded evidence | Code |
|---|---|---|
| One-blocker degree boundary, exact sparse examples | artifact/one-blocker, sparse-ring | Independent graph and audit enumeration; proofs in main.tex |
| All-proposer protection and runtimes on synthetic markets | artifact/protection/protected_results.json | artifact/protection/run_protected.py with the 160 synthetic/ instances |
| Calibrated price/noise settings, sharp 27.78% bound | research/pilot/calibration.json and burden_support.json | calibrate_preferences.py and check_burden_support.py |
| Exact floors at larger inventories and fixed recipient constraints | research/rent-support/results.json, witnesses.json, flow_check.json | Lower bounds, explicit assignments, independent fixed-volume flow replay |
| Original 480-market competition contrasts | research/pilot/transparent_spillover.json and stage1_results.json | transparently scored and seeded uniform recommendations |
| Net reassignment and vacancy accounting | research/occupancy/results.json | 2,880 paired allocations |
| Fresh 100-block, nine-setting sensitivity and Figure 1 | research/competition-stress/results.json and counts.csv.gz | Batched H200 DA, independent CPU checks, simultaneous block-bootstrap bands |
| Separate catalogue and renter selection effects | research/factorial/results.json and block_counts.csv.gz | 40 additional blocks, all 480 intended keys, 408 supported by every design |
| Fixed-quota minimum-edit diagnostic | research/pilot/capacity_results.json and capacity_markets.csv.gz | research/implementation/experiments/shortlist_capacity.py |
| Earlier independent-priority audit and winter study | artifact/nyc-costs and artifact/winter-replication | Smaller-price-weight, separately cleaned specification |

For the status of the exploratory price-coefficient decomposition, see research/competition-stress/crossed. It reuses the primary H200 inputs. Each block of simulations conditions on the catalogues and input model. Running more random seeds does not add distinct source months or observed leases.

## Rebuild the calibrated development inputs

Python 3.13.6, NumPy 2.2.6, pandas 2.3.3 and SciPy 1.17.1 were used locally. Use a dedicated environment with the packages in research/implementation/requirements.txt. Use Matplotlib 3.10.8 to redraw the figure. The calibrated generator snapshot and required borough geometries are bundled under research/implementation. The baseline implementation is based on commit 981ed97f6b1c517b8aa9a785531c2592a700133c, with the team's subsequent local analysis fixes. Acquire the pinned public data with the script below, which verifies archive hashes and downloads only the three development listing months, the Census housing file and geography crosswalk. The ACS cache contains all renter groups for survey benchmarks; only the designated calibration split enters these outcome simulations.

~~~sh
mkdir -p replay
python -B nyc-inputs/fetch_inputs.py --data nyc-inputs/data
python -B research/pilot/prepare_development.py --repo research/implementation --housing-zip nyc-inputs/data/acs/csv_hny.zip --raw-listings nyc-inputs/data/firstmover --report replay/development_inputs.json
~~~

## Check the primary claims

Paths below assume the archive root, except the two clearly grouped subshells. These commands will run the deterministic/seeded arms; the distributed configuration references model caches that are unused for these results. No external model calls or API keys are required.

~~~sh
(cd research/implementation && python -B -m experiments.calibrate_preferences --workers 4)
(cd research/implementation && python -B -m experiments.run configs/stage1_pilot.json --workers 4)
python -B research/pilot/aggregate_pilot.py --repo research/implementation --out replay/stage1_results.json
python -B research/pilot/check_burden_support.py --repo research/implementation --calibration research/pilot/calibration.json --out replay/burden.json
python -B research/rent-support/nested_support.py --repo research/implementation --calibration research/pilot/calibration.json --out replay/nested
python -B research/rent-support/verify_witnesses.py --repo research/implementation --calibration research/pilot/calibration.json --evidence replay/nested
python -B research/rent-support/verify_flow.py --repo research/implementation --calibration research/pilot/calibration.json --evidence replay/nested
python -B research/pilot/transparent_spillover.py --repo research/implementation --stage1-results replay/stage1_results.json --out replay/spillover.json
python -B research/occupancy/run.py --repo research/implementation --reference replay/spillover.json --out replay/occupancy.json
python -B research/factorial/run.py --repo research/implementation --out replay/factorial
(cd research/implementation && python -B -m experiments.shortlist_capacity --output ../../replay/edits)
python -B research/competition-stress/summarize.py --ranks-meta research/competition-stress/ranks-metadata.json --counts research/competition-stress/counts.csv.gz --out replay/grid.json
python -B figures/preference_grid.py replay/grid.json replay/grid.pdf
~~~

The two H200 jobs each executed 129,600 allocations on the same 2,400 market draws. The second crossed scoring and latent price coefficients at fixed noise. All 43,200 market-level rows per experiment were independently reproduced on CPU. The cluster README under research/competition-stress describes rebuilding ranks, replaying all assignments and verifying the GPU results. It includes the frozen designs, run scripts, hardware and simulator versions, compressed cell/seed counts and reference outcomes. Source releases are fingerprinted; archives compressed with local timestamps can have different byte digests despite equal contents. The KKL Slurm identifiers in the receipts correspond to completed jobs. The package does not include input household identifiers, raw addresses or the full rank tensor; the tensor is regenerated from the public inputs.

## Earlier audit benchmark

The last empirical section uses a different donor hash and catalogue cleaning from the calibrated experiment. The same public listings supplied source data, but these pipelines produce different markets. The earlier preparation and replay notes are included in EARLIER_REPLAY.md; use the graph, solver, input and NYC commands there. Its earlier-paper build instructions and its old figure refer to the first manuscript version. The current root main.tex and figures/preference_grid.py build this revision. Charging a full audit for a portfolio is an assumption in the audit experiment; changing one displayed slot is a different action used only for the capacity diagnostic.
