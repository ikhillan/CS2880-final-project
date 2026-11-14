# Reapplication and shortlists

This linkage asks which courses later named by an applicant could have been restored to their hypothetical five-choice menu. It retains every regular-round applicant and searches their complementary-round record for an explicit choice at the same grade, school and course. System-added continuity entries in C2 are excluded. The re-listed edge gets its complementary rank for the ordered replay. D2 offers and April enrollment enter after the edge vector is written. The reconstruction holds participants’ historical later search fixed across the two modeled menus.

Use the project's frozen numeric markets, next-April register vectors and the ministry regular-plus-complementary SAE archives for 2022–25. The institutional reproduction instructions show how to prepare the numeric markets and registers. B1 masked identifiers are sorted and A1/A2 course indices follow the original source row order. The script verifies every C1 applicant and course in the retained graph against the numerical market. For each year run

```sh
python prepare_relisted.py --publication "$PUBLICATION" --year "$YEAR" \
  --archive "$SAE_ARCHIVE" --numeric "$NUMERIC/$YEAR" --register "$REGISTER" \
  --private "$PRIVATE" --out "$RESULTS/relisted-$YEAR.csv"
```

The solver reads `YEAR-relisted.i4` as one little-endian integer per retained C1 edge, zero for unavailable and the explicit C2 rank otherwise. `YEAR-unoffered.u1` describes the historical group with neither an initial nor a post-wait-list regular offer. A separate two-stage bound reads two private arrays, each little-endian int32 in row-major order. `YEAR-second-edges.i4` has four columns (regular applicant index, complementary course index, explicit C2 rank, indicator for a school absent from all original C1 applications). `YEAR-second-courses.i4` has three (A2 published total vacancies, A1 course index or −1, establishment identifier). The edge stream includes actual same-grade returners originally in B1. The course stream includes all A2 courses.

`relisted-YEAR.csv` counts pupils and individual course mentions separately. A pupil may name both a new school and a course they pursued previously. On the course scale, eligible repeats, repeats below the continuity cutoff, and courses absent from the entire C1 record form a partition. On the pupil scale these groups can overlap. An opportunity to repeat an omitted choice means that at least one eligible regular option is absent from either of the two frozen menus. An open opportunity also requires positive published total vacancies at that same complementary course. These conditions describe availability. They do not establish that a particular pupil qualifies for a reserved seat. An observed enrollment hit requires D2's exact initial offer to match the explicitly named course and the following April register to name its school.

After the behavioral solver writes its privately held flags for pupils placed by rank-blind only, run

```sh
python loss_history.py --inputs "$PRIVATE" --solver "$SOLVER_PRIVATE" --out "$RESULTS/loss-history.csv"
python verify_relisted.py --publication "$PUBLICATION" --results "$RESULTS"
```

The verification compares annual roster, explicit returners, new schools, new courses and new-school offers followed by enrollment with previous independent analyses. It also checks the course partition and opportunity denominators. Only aggregate outputs are distributed.
