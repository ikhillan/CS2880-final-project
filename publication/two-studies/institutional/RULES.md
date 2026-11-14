# Chile’s regular-round assignment, 2022–2024 application years

A historical replay has an observable target. For each person who applied in the regular round, predict the initial school and course, or no offer. The later waiting-list assignment, the family’s response, and the April enrollment register each answer a different question. A simulation that generates new lotteries faces an additional empirical question. Which random orderings could the system have generated? A perfect historical replay at the released order does not answer it.

## What goes into the historical replay

| Release | Information available in all three cycles | Use |
| --- | --- | --- |
| A1 | `rbd`, `cod_curso`, grade, `vacantes_regular`, `vacantes_prioritarios`, `vacantes_pie`, `vacantes_alta_exigencia_t`, `vacantes_alta_exigencia_r`, `software_aleatorio` | Identify the program by school and course and retain its physical seat counts by category. The last variable records Alpha or Beta, the software the school chose. |
| B1 | Pseudonymous applicant ID `mrun`, `prioritario`, `alto_rendimiento` | Identify disadvantaged applicants and eligibility under the permanent academic reserve. |
| C1 | Applicant-by-program report, `preferencia_postulante`, `agregada_por_continuidad`, priority flags for continuing pupils, siblings, staff children and former pupils, `loteria_original`, `orden_pie`, `es_pie`, `orden_alta_exigencia_transicion` | Retain submitted order and system-added rows. `loteria_original` gives the realized order before substantive priority tiers, with lower numbers better. Application-specific PIE rank establishes candidacy. Prior PIE status at the home course has a separate use in routing. |
| F1 | `mrun_1`, `mrun_2`, `mismo_nivel`, `es_hermano`, `postula_en_bloque` | Distinguish higher-grade links, sibling status, and the guardian’s request for the family block. For different grades the first ID is the higher-grade applicant. One row is a relationship. The numbers of rows are 101,014, 91,717 and 72,880. |
| D1 | `mrun`, `rbd_admitido`, `cod_curso_admitido` | Use only after assignment to score exact agreement, including no offer. D1 provides no seat-category outcome against which to validate inferred routing. |

The data and codebooks come from Chile’s Ministry of Education. The 2022, 2023 and 2024 labels refer to the application year. In particular, an archive’s complementary round or post-waiting-list capacity file cannot supply the initial capacities for this comparison. [Ministry archive](https://datosabiertos.mineduc.cl/sistema-de-admision-escolar-sae/)

## Admissions decisions that can change an offer

1. **Construct the full report before processing families.** An incumbent has a protected option at the current program. A family block can promote a different school above it, including one originally listed below the protected option. Keeping all C1 rows through this step matters. The personal fallback belongs to the incumbent if no physical seat remains at the protected program. A returning student uses a physical seat when available.
2. **Run grades from highest to lowest.** A sibling just assigned at a school in a higher grade can confer priority on a younger applicant there. A declared block can move listed programs at higher-grade family members’ schools to the front of the lower-grade applicant’s report. Preserve internal order among promoted options and among remaining options. Recompute both effects after every change in the lottery or displayed menu. The published reconstruction applies these updates from previously cleared grades. For same-grade siblings, article 41 restricts the statutory priority to families requesting a block. Same-course lottery dependence is a separate feature assessed below.
3. **Route proposals through seat categories.** Keep regular, disadvantaged, PIE, and the applicable transitional or permanent academic subschool. For a non-incumbent, active PIE and academic subschools are tried first. Between disadvantaged and regular seats, the disadvantaged subschool comes first except for a non-disadvantaged sibling. At home, an incumbent tries the disadvantaged and regular subschools in the order determined by disadvantaged status, before the academic subschool. Prior PIE places the PIE subschool first, otherwise last. An otherwise ineligible pupil can consume unused reserved capacity. The sum of the category capacities can differ from the number of entrants when personal fallback rights are used.
4. **Give each subschool its own priority.** At an ordinary seat the order is incumbent, enrolled sibling, newly assigned higher-grade sibling, staff child, former pupil, remaining applicants. At a disadvantaged seat the two sibling groups precede disadvantaged status. PIE and transitional academic candidacy use school-specific order fields as appropriate. The permanent academic route uses performance eligibility. Recorded lottery order and ultimately applicant ID resolve remaining ties. The 15% disadvantage target depends on enrollment already present. A1 publishes the resulting number of seats, so the reconstruction need not infer it by multiplying open vacancies by 0.15.

Articles 24, 27, 34–37, 41 and 42 of Decreto 152 set the legal framework. The seat-order implementation and the distinction between a reported option and the list actually processed come from the designers’ description and the more recent administrative reconstruction. ConsiliumBots reports operating Chile’s assignment in all three cycles. [Admissions regulation](https://bcn.cl/3lorh) · [Correa et al., §§4.1–4.2](https://www.dii.uchile.cl/~jcorrea/papers/Journals/2022CEER.pdf) · [Doğan, Imamura and Yenmez, Appendix S7.1](https://arxiv.org/html/2610.28471v1) · [Operator’s project description](https://www.consiliumbots.com/projects/sae-algorithm-feedback-chile/)

Doğan, Imamura and Yenmez report all 570,891, 536,353 and 473,482 official program-or-no-offer outcomes reproduced in these cycles. The corresponding numbers with an official offer were 520,420, 492,888 and 439,038. These are an external benchmark. See this study’s own agreement table for its independent replay. Agreement on initial destinations cannot validate latent seat labels, identify all proposal orders, or establish what assignment would follow a different lottery. [Doğan, Imamura and Yenmez, Table S1 and §S7.1.1](https://arxiv.org/html/2610.28471v1)

## What the lottery records let us test

The designers describe drawing one family order per school, then drawing an order among children in each family. Article 36 addresses an applicant’s order for a particular course at a school. Those descriptions motivate a direct test in the three study cohorts. If every course at a school used the same strict individual order, two applicants to the same two courses would appear in the same relative order at both. This implication holds even if ranking scales or applicant pools differ across courses. [Correa et al., §4.1.2](https://www.dii.uchile.cl/~jcorrea/papers/Journals/2022CEER.pdf) · [Decreto 152, article 36](https://bcn.cl/3lorh)

| Application year | Applicant-pair/course-pair comparisons | Reversed | Same-school course pairs with a reversal / testable pairs | F1 family/course groups consecutive / eligible groups |
| --- | ---: | ---: | ---: | ---: |
| 2022 | 664,372 | 323,303 | 2,537 / 2,979 | 18,285 / 19,033 |
| 2023 | 609,370 | 299,697 | 2,375 / 2,793 | 17,234 / 17,908 |
| 2024 | 549,253 | 268,182 | 2,091 / 2,546 | 15,598 / 16,166 |

The 891,182 reversals in 1,822,995 comparisons rule out a common strict individual order across every course at each establishment. The statistic compares raw ranks before priority tiers. Each unordered applicant pair in each unordered course pair counts once. People, families and schools can recur across comparisons. The reversal share is a description of the released orderings and does not identify independence between courses.

We also ask whether applicants from the same F1 connected component occupy consecutive positions in a course lottery. We include courses with at least two applicants from the component and at least one outsider, then compare rank positions against all applicants at that course. In total, 51,117 of 53,107 groups (96.25%) were contiguous. With independent individual tickets, the expected number of contiguous groups conditional on the observed group and course sizes is 2,833.49 (5.34%). A group of size $k$ in a course of size $N$ occupies one of $\binom{N}{k}$ equally likely position sets. Exactly $N-k+1$ of these are contiguous. The same calculation restricted to components connected by opt-in block links gives 43,068 of 44,310 (97.20%). These counts give empirical grounds to examine family dependence and to label a generator that forces every observed family to be contiguous as an approximation. The administrative archives disclose one order per course per year. The joint distribution of hypothetical future orders requires assumptions.

## Reproduction

The commands run from this directory. Supply the three Ministry SAE archives in chronological order. Python with pandas and an archive reader that supports RAR (`bsdtar`) are required. Both scripts write only aggregate diagnostics.

```sh
python3 lottery_architecture.py --archives "$SAE_2022" "$SAE_2023" "$SAE_2024" --out lottery-architecture.csv
python3 family_adjacency.py --archives "$SAE_2022" "$SAE_2023" "$SAE_2024" --out family-adjacency.csv
```
