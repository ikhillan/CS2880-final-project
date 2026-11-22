# Chile’s 2021 entry-grade applicants

This benchmark compares two five-course recommendation rules in the same application year and grades as the School Explorer experiment. The 207,578 entry-grade applicants are the national regular-round population. Agte et al. report the same population size in Appendix Table A.IV. The public files do not identify which families received experimental advice. We re-clear the entire market for each rule, then count the entry-grade outcomes. All families retain their submitted ordering and the Ministry’s realized priorities throughout the comparison. [Experimental sample and population](https://www.nber.org/system/files/working_papers/w32670/w32670.pdf) · [Ministry SAE catalog](https://datosabiertos.mineduc.cl/sistema-de-admision-escolar-sae/)

## Reproduce both tables

Use the full paper repository for the published school-market and institutional code. The driver accepts the location of its `publication` folder. It requires Python 3 with NumPy, pandas and SciPy, a C++17 compiler, and libarchive’s `bsdtar` with RAR support. The following sequence uses a new working directory. Set `COHORT` to the folder containing this README. The pinned public revision supplies all reused modules. [Project code](https://github.com/ikhillan/CS2880-final-project)

~~~sh
COHORT=/absolute/path/to/this/cohort
git clone https://github.com/ikhillan/CS2880-final-project.git project
git -C project checkout 28f0376ff3cc0069f05faf4b17797ed250ec1dc0
mkdir -p archives scratch results
curl -fL https://datosabiertos.mineduc.cl/wp-content/uploads/2021/12/SAE_2019.zip -o archives/SAE_2019.zip
curl -fL https://datosabiertos.mineduc.cl/wp-content/uploads/2022/02/SAE_2021.rar -o archives/SAE_2021.rar
curl -fL https://datosabiertos.mineduc.cl/wp-content/uploads/2022/09/Matricula-por-estudiante-2022.rar -o archives/matricula-2022.rar
python3 "$COHORT/reproduce.py" --publication project/publication --archives archives \
  --register archives/matricula-2022.rar --scratch scratch --results results
~~~

If the original 2021 market and frozen masks have already been prepared, append `--numeric /path/to/market-and-masks` to the same command. That run requires only the 2021 admissions and 2022 enrollment archives. The runner prints each subcommand. Both versions build fresh side information and a fresh April register link, independently validate the initial assignment, and compare both final tables byte for byte with the versions distributed here. Temporary applicant-level files live in the private scratch directory and are removed on completion.

**What was fixed ahead of this register comparison.** The two 2019 offer predictors have the same candidate courses, demand, capacity and priority features. The second also uses submitted rank. We apply the archived coefficients and 2019 school-course offer-rate priors to 2021. For each applicant the two rules retain up to five discretionary courses by score and also keep any protected continuity course. Both use the same seeded tie priorities and preserve the original preference order on the courses retained. The original cutoff at the first continuity course remains part of the policy definition. The raw official-assignment check instead retains the complete submitted C1 file, including entries below that cutoff.

**How the joins work.** A1 declares the school/course identities and typed capacities, B1 supplies pupils and grades, C1 supplies preferences and realized lottery positions, and F1 supplies family links. The unabridged historical solver receives dummy offer fields and writes one predicted school/course or outside status for each applicant. A separate comparator then opens D1 and matches each pupil’s initial offered course, or outside status, by ID and school/course key. All 461,223 are equal. The attendance check joins each regular-round pupil’s pseudonymous B1 ID to the following April’s `MRUN` and reads the enrolled school `RBD`; the join checks unique pupil identifiers, the 2022 register year and the course-to-school mapping. It uses school identity because the enrollment register does not identify the offered SAE course. [Ministry enrollment catalog](https://datosabiertos.mineduc.cl/matricula-por-estudiante-2/)

## Reading the counts

- `all` covers the entire 2021 pool. `entry` pools `cod_nivel` −1 (prekindergarten), 0 (kindergarten), and 1 (first grade). `other` comprises grades above 1. Groups are recorded after both national clearings.
- `applicants` is the group size and `linked` counts pupils found in the April register. The entry-grade counts are 207,578 and 200,591, leaving 6,987 pupils without a register link.
- `official_placed` counts original D1 initial offers. `exact_official` counts all applicants whose complete-report historical reconstruction equals D1, including the outside option. In the independent D1 table the disagreement categories partition all errors. `younger_block` records younger pupils in a different-grade family-block request.
- `blind_placed` and `rank_placed` count simulated assignments among all group members; the `*_linked_placed` fields count simulated assignments among April-linked pupils. `*_april_hits` counts these pupils whose historical April school appears somewhere in the respective menu. A school can have several courses. A hit counts once per pupil. April enrollments stay fixed under both rules.
- `linked_gains` counts linked pupils with a simulated seat only under the rank-informed rule, and `linked_losses` counts pupils with a seat only under the rank-blind rule. Among linked entry-grade pupils, the rules cover 160,117 and 160,820 April schools (+703); simulated placements are 183,634 and 183,325 (133 gains, 442 losses, net −309). Across *all* entry-grade applicants, simulated placements change from 188,933 to 188,589 (−344).

## Checksums

The original archives and archived model are pinned before use. The driver checks both regenerated menu hashes. The staged-input route also verifies the market binary; floating-point libraries can produce byte differences in regenerated continuous scores even when the final menus are identical.

| File | SHA-256 |
| --- | --- |
| SAE_2019.zip | `79df389c05e37b040c728dfc80bdc0dfd7ca1cbdde708710771e3e048e582f6e` |
| SAE_2021.rar | `8bd459562803c6a57e60e51ecf8198beb212100400e8cdd058cbcfbf1aa29261` |
| Matricula-por-estudiante-2022.rar | `9d5ecc67b3bf0e70256c001b70d4246d89b7618f857a395a58c2b4b58840bd9a` |
| fit-and-inputs.json | `122413f014977a69ddd213e41b87a3979b4b979dd9414c97b7c4a1fe91ae4f75` |
| Original 2021.bin | `b376961275925006c11ef1800d14f017e71e05a8a8fb75612d6ec9547478a2e7` |
| 2021-blind.mask | `6026f7fe02a6921381dd805c725b91b8eace01d9145887e20d132b3108e3c478` |
| 2021-rank.mask | `3d9085e7f6a797408d435c44ad828a8a65ad7f33f6781003be967ad28c94c114` |
| entry-2021.csv | `46f47d15c1de5f1dbd39e1a5298d99312dbe94f095de8ee971e4e7c2f10a2004` |
| official-2021.csv | `967903be54cbfbb5bbd1f6f461f589d0386dc26d189e9a6edb6975e92c476488` |
