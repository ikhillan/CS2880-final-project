# Common benchmark and additional robustness checks

The public branch recorded both later application years and the two directional criteria in [the validation plan at commit 261e1e7](https://github.com/ikhillan/CS2880-final-project/blob/261e1e7d2a2863589c5092b5b5df944e16673e6e/publication/two-studies/robustness/VALIDATION_PLAN.md) before either download. The rental pilot and the 2022–23 school outcomes were known when we specified the other robustness analyses.

| Check | Evidence |
| --- | --- |
| First versus uniform five, both sites | [Four-panel table with Monte Carlo errors](../evidence/common-benchmark.csv) |
| Later years and realized-order agreement | [2024–25 school outcomes](../evidence/later-school-results.csv) · [Cohort provenance and frozen fit digest](../evidence/later-cohorts.json) |
| Partial adoption and preference/screening perturbations | [NYC methods and rerun](nyc/rental-rollout-methods.md) · [2022–23 school quota, rollout and lottery methods](chile/README.md) |
| Assignment-preserving switches and turnover proof checks | [Method, reproduction and comparison with related theory](theory/README.md) · [Policy counts](theory/rollout-certificates.csv) |

These checks need a recent copy of the project, its Python dependencies, and the prepared inputs described in the paper's main README. From this directory set REPO to the absolute path to the checkout and RECORDS to a private scratch directory outside the repository. Run the following for the common benchmark and the independent aggregate consistency tests.

```sh
mkdir -p "$RECORDS"
python common_benchmark.py --repo "$REPO" --output "$RECORDS/common-benchmark.csv"
python verify_robustness.py
```

To reproduce the later cohorts, use the original public Ministry archives listed in [sources.json](sources.json). Set NEW to the directory holding those two files, OLD to the directory containing SAE_2019.zip, and NUM to private scratch for the new binaries. The scoring script checks the source digests before producing the private replay inputs. The original fit uses only 2019 program history. For an independent port-equivalence check, the archived scoring pipeline was run with just the year loop, source URLs and filenames changed. The two numeric markets and all twelve mask/partner files matched byte for byte. [Pipeline receipt](validation/legacy-pipeline-check.json) The four priority flags have matching definitions across both releases. For application year 2025 the codebook still calls the course tie position LOTERIA_ORIGINAL and gives smaller values higher precedence. The [ministry describes the switch to a deterministic ordering](https://www.mineduc.cl/sadmision-escolar-2026-sae-inicio-el-registro-anticipado-de-apoderados/). The model consumes the released within-course order as supplied. Nothing in the validation requires random tie-breaking.

```sh
PUB="$REPO/publication"
python frozen_school.py --school "$PUB/research/observed/school" --old "$OLD" --new "$NEW" --numeric "$NUM" --sources sources.json --record "$RECORDS/later-cohorts.json"
c++ -O3 -std=c++17 -I"$PUB/research/observed/school" "$PUB/research/policy/equity/export_menus.cpp" -o "$RECORDS/export_menus"
for year in 2024 2025; do
    "$RECORDS/export_menus" "$NUM/$year.bin" "$year" "$NUM"
done
python later_school_results.py --publication "$PUB" --numeric "$NUM" --out "$RECORDS/later-school-results.csv"
```

Each later-cohort policy was replayed twice, with the original serial DA and the batched implementation. Every partner vector matched. An independent reader recomputed all eighteen year-by-group-by-policy aggregate rows directly against the archived A1, B1, C1 and D1 tables. See [the validation receipt](validation/direct-release-check.json). The school tables released here contain only aggregate counts. The model inputs and applicant vectors stay in private scratch.

The NYC common benchmark uses exact expected full-partner coverage under a uniformly sampled five-option menu plus 48 draws for expected placement. The original predictor contrast uses its one frozen uniform display. Their uniform estimates can differ. In the two school panels the comparison uses the same 48 draws as the previous paper. Counts and percentages in each table state the target population and denominator. Simulation-block intervals in NYC measure variation across blocks. Monte Carlo errors for rollout measure uncertainty from simulating Bernoulli adoption on the fixed inputs. Held-out school years supply finite cohort comparisons. We do not infer a citywide housing effect or a causal placement effect in either location.

The [prefix check](theory/prefix_obstruction.py) tests the first-five guarantee and the source of the school continuity exception. Run it with the same `--repo` and `--numeric` paths as the policy-screen checker and an `--out` filename. [Counts by setting](theory/prefix-obstruction.csv)

To rerun the independent checks of the later school releases, use the same private input directories as above.

```sh
python validation/check_legacy_pipeline.py --repo "$REPO" --old "$OLD" --archives "$NEW" --numeric "$NUM" --out "$RECORDS/legacy-pipeline-check.json"
python validation/check_releases.py --paper "$PUB/two-studies" --archives "$NEW" --numeric "$NUM" --out "$RECORDS/direct-release-check.json"
```
