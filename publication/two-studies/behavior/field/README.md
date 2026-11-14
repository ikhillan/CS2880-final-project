# Where did applicants actually enroll?

The Ministry's applications record which schools families listed in two successive rounds. A separate student register records where each pupil enrolled by April 30 of the following school year. We use four application years, 2022–25, corresponding to the 2023–26 school years. The unit in pooled counts is one applicant in one application year. The same pupil can appear in more than one year. [SAE releases and codebooks](https://datosabiertos.mineduc.cl/sistema-de-admision-escolar-sae/) · [Annual official enrollment registers](https://datosabiertos.mineduc.cl/matricula-por-estudiante-2/)

## Predicting where pupils enrolled

Of 2,044,649 regular applicant-years, 1,969,120 (96.31%) link to a next-April enrollment record. Both menus are scored on this same population. A menu gets one hit if any of its displayed courses belongs to the school recorded in the register. Every applicant can contribute at most one hit. The historical-offer comparison uses the school named in the initial D1 offer. An applicant without such an offer contributes zero hits and stays in the common denominator. The final column reports the change in pooled-DA placement for the same register-linked pupils.

| Application year | Linked pupils | Enrollment-school hits gained | Hits lost | Net enrollment-school hits | Net initial-offer-school hits | Net simulated placements |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 549,997 | 1,425 | 418 | +1,007 | +993 | −405 |
| 2023 | 516,921 | 1,633 | 435 | +1,198 | +1,182 | −482 |
| 2024 | 457,231 | 1,765 | 380 | +1,385 | +1,474 | −434 |
| 2025 | 444,971 | 1,361 | 313 | +1,048 | +1,170 | −288 |
| Total | 1,969,120 | 6,184 | 1,546 | +4,638 | +4,819 | −1,609 |

From rank-blind to rank-informed, coverage rises from 1,605,941 to 1,610,579 (81.56% to 81.79%). Requiring the same school, education type and grade yields 4,487 net additional hits, positive in all four years. The predictions use the realized historical destination. Changes in school attendance had either menu been implemented are unobserved.

The original **eligible graph** contains the submitted regular-round courses through each pupil's continuity entitlement, including that entitlement. A menu drawn from it can only cover enrollments within that graph. It contains the April school for 1,614,681 linked pupils (82.00%). Among the 126,708 registered pupils who had no regular or wait-list offer, the ceiling is 25,499 (20.12%). Here the two menus cover 25,158 and 25,122 April schools, respectively. The yearly net changes are +4, −14, −8 and −18. Among the remaining 1,842,412 linked pupils, the ceiling is 86.26% and rank information adds 4,674 net hits. Selection on past outcomes, the historical allocation and outside options all matter to this comparison. [All cohorts, denominators and resolutions](enrollment-target.csv)

## Bounding the effect of missing enrollment records

An unlinked pupil can favor the blind predictor in school-level coverage only if the blind menu contains at least one school absent from the informed menu. An exclusive course at a school on both menus cannot make a difference. The reverse condition limits gains for the informed predictor. We counted both conditions after deduplicating schools within applicant and predictor. Subtracting the blind-exclusive count from the observed change gives a lower bound for the complete cohort. Adding the informed-exclusive count gives an upper bound. These limits allow each missing pupil a different school or no enrollment. They impose no random-missingness assumption. They concern coverage of the historical school only.

| Year | Documented net increase | Unlinked with blind-exclusive school | Unlinked with informed-exclusive school | Range including all missing outcomes |
|---|---:|---:|---:|---:|
| 2022 | 1,007 | 953 | 954 | [54, 1,961] |
| 2023 | 1,198 | 885 | 881 | [313, 2,079] |
| 2024 | 1,385 | 943 | 950 | [442, 2,335] |
| 2025 | 1,048 | 979 | 972 | [69, 2,020] |
| Total | 4,638 | 3,760 | 3,757 | [878, 8,395] |

The separate [bound calculation](enrollment_bounds.py) reconstructs the entire cohort's eligible course sequence from raw files, checks it against the frozen binary graph and deduplicates each menu by school. [Yearly counts](enrollment-bounds.csv) · [Independent check using per-pupil school sets](missing-bounds-check.csv)

## What pupils did in the complementary round

After the regular round and its wait list, families with no assignment could apply for residual vacancies. Families who had rejected an offer and new applicants could also enter. We count people with no school at either regular decision as a defined stratum eligible to apply again. Other regular participants and those first seen in the complementary round are tabulated separately. [2022 Ministry guidance](https://www.mineduc.cl/periodo-complementario-de-postulacion-para-la-admision-escolar-2023/) · [2023 Ministry guidance](https://www.mineduc.cl/familias-tienen-hasta-el-24-de-noviembre-para-postular-al-sae/) · [2024 Ministry guidance](https://subeduc.mineduc.cl/sistema-de-admision-escolar-sae-comienza-el-periodo-complementario-de-postulacion-a-colegios/)

| Application year | No initial or wait-list offer | Reapplied at same grade | Explicitly added a school | Offered a course at a new school they listed | Enrolled at that school in April |
|---|---:|---:|---:|---:|---:|
| 2022 | 43,077 | 21,371 | 20,858 | 15,095 | 8,192 |
| 2023 | 37,757 | 18,537 | 18,116 | 13,895 | 7,923 |
| 2024 | 30,486 | 14,599 | 14,532 | 12,340 | 7,304 |
| 2025 | 30,025 | 16,041 | 15,730 | 13,071 | 7,280 |
| Total | 141,345 | 70,548 | 69,236 | 54,401 | 30,699 |

All 141,345 were eligible on the stated no-offer criterion. Of these, 71,143 appeared in the complementary round, including 595 at a different grade. Among the same-grade returners, 60,152 ranked a new school first. Of the 46,940 with a previously listed manual course that still had at least one published vacancy, 45,848 also listed a new school. A positive vacancy is a course-level measure. The applicant's priorities and other applications determine whether they could obtain it.

D2 contains an assignment and no response field. Among the 54,401 who received an offer for a new school they explicitly listed, 30,699 were registered at that school in April (56.43%), 21,257 were registered at another school and 2,445 did not link to the register. Restricting the match to the offered school, education type and grade gives 26,742. Of the 141,345 in the full group, 21.72% followed this observed path from no offer to enrollment at a previously unlisted school. These 30,699 also account for 30.33% of the 101,209 registered, previously unoffered pupils whose eventual school was outside the original eligible graph. Timing, information about vacancies and the student's outside options all changed between rounds. [Application and offer counts](round-link-counts.csv) · [Enrollment counts](round-link-register.csv)

Of the 2,527 gross simulated placement losses, 509 historically had no regular or wait-list offer. Among these 509, 293 reapplied at the same grade, 198 obtained a later offer at a newly listed school and 103 enrolled there. Among the 740 simulated gains, 103 had no historical regular or wait-list offer, 55 reapplied at the same grade, 33 received a new-school offer and 14 enrolled there. The menu comparison is counterfactual. Each subsequent action comes from the historical system. These counts reveal how many of the affected pupils are represented in the historical follow-up. [All joint strata](round-link-policy.csv)

## Definitions and verification

- The B1 roster, D1 outcomes and C1 applicant IDs coincide in each year. The same check holds for B2, D2 and C2. Link on numeric MRUN within year and require equal `cod_nivel` between application rounds for choice and offer comparisons. The initial D1 response code 6 means no offer and compulsory wait-list participation. Post-wait-list code 6 means unassigned. Both indicators together agree for every record with both school fields being empty.
- A school is the institution identifier RBD. A **new school** has no occurrence anywhere in that pupil's raw C1 list, even below the continuity entitlement or in an automatically appended row. The new school must occur in an explicitly entered C2 row (`agregada_por_continuidad == 0`). An offer must match the exact RBD **and course** on such a C2 row, so assignments outside the family's explicit list cannot create a hit. In the same-grade focal group 66,082 received any D2 offer, 55,948 received an explicitly listed course, and 54,401 received an explicitly listed course at a new school. Of the latter, 54,301 courses reported a positive vacancy in both A1 and A2.
- Link each SAE year to the following school year's SIGE register by the masked student identifier MRUN. Every matched regular applicant has one unambiguous April school. `RBD`, `COD_ENSE`, and `COD_GRADO` identify school, teaching type and grade. For the strict course-type check, obtain the offered course's teaching and grade codes from the A2 supply table. April measures enrollment on the census date. Earlier enrollment followed by a school change can therefore generate a different April school. A missing link is reported separately.
- As a linkage check, 1,025,818 of 1,048,526 applicants who explicitly accepted their initial regular-round offer without waiting have a register record. Of those originally accepting, 914,634 appear at that school in April. [Enrollment file digests and checks](register-inputs.csv) · [Admissions file digests and coverage](round-link-inputs.csv)
- Two independent calculations verify the focal historical counts, including enrollment. A separate school-hit checker rebuilds the entire eligible C1 edge order from raw MRUN, rank and continuing-seat priorities. It compares distinct ID sets with results from the binary replay file for both labels and both resolutions. Both use the same frozen menu masks. [Cross-round check](round-link-checks.json) · [Menu-label check](enrollment-target-checks.json)

## Reproduce

To check the distributed aggregates using Python alone, run `python verify_counts.py`. Use the commands below to reproduce the estimates from the original files.

The scripts use Python, NumPy, pandas and `bsdtar` with RAR support. The main runs used Python 3.13.6, NumPy 2.2.6 and pandas 2.3.3. Run from this directory. Put the four SAE archives in `$SAE` under the names given in `round_link.py`. Download the four register archives using the URLs in `register-inputs.csv`, naming them `matricula-2023.rar` through `matricula-2026.rar` in `$REG`. Write outputs to a separate `$OUT` directory. The original cohort files and each year's register digest permit direct input checks. All released results are aggregate counts.

```sh
mkdir -p "$OUT"
python round_link.py --early-archives "$SAE" --late-archives "$SAE" --out "$OUT"
python link_register.py --early-archives "$SAE" --late-archives "$SAE" --enrollments "$REG" --out "$OUT"
python check_round_link.py --early-archives "$SAE" --late-archives "$SAE" --enrollments "$REG" --out "$OUT"
```

For the school-label validation and joint counterfactual groups, set `$NUM_EARLY` and `$NUM_LATE` to the generated frozen-policy numeric files for the 2022–23 and 2024–25 application years. The `research/policy/equity` and `two-studies/robustness` portions of the paper archive give the preparation commands. Each year requires its `.bin`, `full/blind/rank.mask` and `blind/rank.partners` files. Ascending numeric MRUN is the applicant order in the original preparation script. The binary grade and exact historical offer index agree with the raw files for every original applicant.

```sh
python link_register.py --early-archives "$SAE" --late-archives "$SAE" --enrollments "$REG" --early-numeric "$NUM_EARLY" --late-numeric "$NUM_LATE" --out "$OUT"
python enrollment_target.py --early-archives "$SAE" --late-archives "$SAE" --enrollments "$REG" --early-numeric "$NUM_EARLY" --late-numeric "$NUM_LATE" --out "$OUT"
python check_enrollment_target.py --early-archives "$SAE" --late-archives "$SAE" --enrollments "$REG" --early-numeric "$NUM_EARLY" --late-numeric "$NUM_LATE" --out "$OUT"
```

To reproduce the missing-register bounds using the same inputs, run

```sh
python enrollment_bounds.py --early-archives "$SAE" --late-archives "$SAE" --registers "$REG" --early-numeric "$NUM_EARLY" --late-numeric "$NUM_LATE" --out "$OUT/enrollment-bounds.csv"
python check_missing_bounds.py --early-archives "$SAE" --late-archives "$SAE" --early-numeric "$NUM_EARLY" --late-numeric "$NUM_LATE" --enrollments "$REG" --target "$OUT/enrollment-bounds.csv" --out "$OUT/missing-bounds-check.csv"
```

## Evidence from actual advice

Arteaga, Kapor, Neilson and Zimmerman studied actual platform warnings in earlier Chilean cohorts. Near a 30% predicted nonplacement threshold in 2018–20, receiving the pop-up increased the probability of adding a school by 21.6 percentage points (SE 1.0), a placement at a listed preference by 3.8 points (SE 0.9), and enrollment at the placed school by 2.4 points (SE 1.0). These are the authors' pooled reduced-form local linear discontinuity estimates using a 0.10 bandwidth and triangular kernel, with 20,359 and 21,145 observations on either side. In a separate 2020 high-risk sample of 17,970, random assignment to supplementary WhatsApp messaging increased adding any school by 3.3 points (SE 0.2) before a subsequent SMS and 4.4 (SE 0.3) by endline. These effects pertain to the studied warnings and populations. They give no estimate of adherence to either frozen shortlist. [Original paper](https://doi.org/10.1093/qje/qjac013) · [Authors' replication archive](https://doi.org/10.7910/DVN/VCXL6A)

The authors explain that public SAE data lack application drafts, risk scores and treatment indicators. `read_external_tables.py` verifies checksums for two published aggregate output files, then extracts the named column and its reported standard error. This reproduces the published numbers from tables. The historical linkage and menu analyses above use the independent public records for 2022–26. [Authors' data access statement](https://dataverse.harvard.edu/api/access/datafile/5863241) · [Selected estimates](published-estimates.csv)

```sh
python read_external_tables.py --tables "$OUT/source-tables" --download --out "$OUT"
```
