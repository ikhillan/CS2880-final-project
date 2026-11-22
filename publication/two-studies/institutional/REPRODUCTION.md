# Reproducing the Chile institutional check

Run the driver from the `institutional` directory in the full project checkout, with a C++17 compiler, libarchive’s `bsdtar` with RAR support, and Python with NumPy, pandas and SciPy. The original project supplies the frozen predictor coefficients, source manifests, eligibility rules, and the published DA implementation. The driver reads those files from `--publication`, the checkout’s `publication` directory. `--results` contains aggregate tables. `--private` contains temporary applicant-level files and executables. Use separate, non-overlapping directories.

```sh
python3 reproduce.py --publication /path/to/CS2880-final-project/publication \
  --archives /path/to/sae-archives --registers /path/to/register-archives \
  --register-template 'matricula-{year}.rar' \
  --private /path/to/local-scratch --results /path/to/replay-results \
  --download
```

`--download` fetches missing public source files from the URLs in the original study’s manifests. Omit it to run from local archives. The template refers to the April **enrollment year**, one greater than the application year. The driver checks each archive’s SHA-256 before it uses the file. For the 2025 application cohort it uses the school enrollment register `Matricula-Oficial-2026.rar` from the Ministry. The enrollment reader checks the entry year and reads `MRUN`, `RBD`, `COD_ENSE` and `COD_GRADO`. The original frozen 2019 predictor pipeline also reads the 2020 and 2021 archives to export its historical reserve files. [Ministry SAE releases](https://datosabiertos.mineduc.cl/sistema-de-admision-escolar-sae/) · [Ministry pupil-level enrollment releases](https://datosabiertos.mineduc.cl/matricula-por-estudiante-2/)

If the original policy files already exist, supply `--early-numeric /path/to/2022-23` and `--late-numeric /path/to/2024-25`. Each directory must contain `<year>.bin`, `<year>-blind.mask` and `<year>-rank.mask` for its two years. This skips the predictor regeneration and the early historical archives. Both paths can designate the same directory if it contains all four years.

Set `--stage historical` to reproduce D1 agreement and the menu and register comparisons. This also runs the small-market mechanism tests and the lottery-order diagnostics. `all-YYYY.csv` evaluates every application on the complete C1 reports. `cutoff-YYYY.csv` repeats that comparison after imposing the frozen predictors’ original eligibility cutoff at continuity. `official-YYYY.csv` records full, blind and informed outcomes on the latter pool. `ablation-YYYY.csv` adds the routing and family rules successively. `linked-YYYY.csv` and `first-YYYY.csv` compare the rank-informed policy with the rank-blind policy and with first-five, respectively. The three cohorts in the paper’s lottery experiment are application years 2022, 2023 and 2024. The 2025 historical replay uses the released order under that year’s revised tie procedure.

Set `--stage ceilings` to regenerate the original pooled control, the matched institutional run, the school- and course-specific family lottery sensitivities, and the exhaustive design-draw check. The first three cohorts each use 1,024 deterministic design draws. A separate set of 64 draws evaluates the matched institutional menu, and 16 draws per family specification evaluate the two family sensitivities. `ceiling-summary.csv` includes the historical single-order result and nested support sizes. The control must reproduce 81, 95 and 102 additional April-school hits. `exact-design-checks.csv` reports re-clearing checks. `ceiling-overlap.csv` records how many of the same pupils contribute to each comparison. The default `--stage all` runs both parts.
