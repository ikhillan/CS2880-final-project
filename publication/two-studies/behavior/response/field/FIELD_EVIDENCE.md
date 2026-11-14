# Chilean evidence on list expansion

## What the external contrasts measure

Arteaga, Kapor, Neilson and Zimmerman study a risk warning shown after a family attempts to submit its application. The window covers Chilean application years 2018–20. The warning is triggered above 30% predicted nonplacement risk. Table II uses local linear fits with a triangular kernel of bandwidth .1, with 20,359 people below and 21,145 above the threshold. Some years also contained reminder interventions. The warning encourages adding schools and does not name a school. It changes a family's decision while applicants can still edit the same application. [Published article, Sections III.B and V](https://www.christopher-neilson.com/work/documents/Warnings/qjac013.pdf)

Table III gives .199 as the fitted probability of adding a school just below the cutoff and +.216 as the increment at the cutoff. Their sum, .415, describes the fitted level just above the cutoff using the published rounded inputs. A scenario can vary the assumed fraction responding from approximately one-fifth to two-fifths to examine sensitivity at this scale. Assigning the same probability to all pupils with a recoverable course, or coupling their simulated decisions across menu policies, are additional modeling choices. A Bernoulli probability applies to a specific denominator. Families whose original list fits completely on a hypothetical menu have no previously reported course to recover. The RD could induce genuinely new schools even for a pupil with one initial choice. [Table III](https://dataverse.harvard.edu/api/access/datafile/5862809)

Table II reports increases of .205 in adding at the end, .017 in adding in the middle, .340 in the number of schools added, and .073 in adding an undersubscribed school. Respondents could make multiple additions. The IV estimate of schools added per induced school-adder is 1.573 (SE .090). These quantities support testing tails, multiple choices and safer options separately. They cannot identify the national share choosing a particular rank, the relative response to two five-course displays, nor the identities of future additions. The .038 placement increase concerns warned families at the threshold. It is not a coefficient for the net effect of switching the recommendation policy for a national cohort. [Table II](https://dataverse.harvard.edu/api/access/datafile/5862803)

A closer information intervention in Agte et al. showed parents a report on known schools and suggested further ones. In its feedback experiment, instrumenting report opening with randomized access gives +.061 (SE .018) adding a school, +.020 (.012) deleting one, and −.002 (.005) changing the top school. This is a different instrument, population and message. It is evidence that what the platform shows matters for how families revise their applications. [Study, Table II](https://www.nber.org/system/files/working_papers/w32670/w32670.pdf)

## Interpretation of the national stress tests

An equal response rate under both menus isolates the allocation effect of a given mode of recovery. Differential response is plausible because the menu could alter a family's own perceived safety. A cross-policy comparison at every pair of tested uptake levels asks whether any sign change occurs when one rule induces more search. It can be computed from the absolute placement and April-hit counts already generated, by crossing the two realized clearing outcomes sharing a common seed. No assumption about independence of pupils' placements is needed. Each realization clears the full interacting market. A reversal proves sensitivity to the stated behavioral difference; a grid of preserved signs proves neither monotonicity between grid points nor robustness to unknown targeted behavior.

Restricting a retrospective replay to previously submitted courses explicitly named again in the complementary round chooses the identity of the additional course using a real applicant's own behavior. Reapplicants self-select after the actual regular outcome and confront a changed vacancy set. Some newly named schools lacked a first-round application edge and have no observed first-round lottery ticket for that pupil. A replay on the original graph therefore measures recourse from revisiting known options only. Testing full restoration of the same originally ranked choices under both policies is an exact zero-gap control; it also clarifies how far positive findings can extend.

The first-round course orders in the administrative files already incorporate any advice delivered before submission. Both shortlist policies were created retrospectively from these submitted choices. The data contain no untreated baseline for the new hypothetical advice.

## Checks

The two input tables are public aggregate outputs. Their two Dataverse file identifiers and checksums are embedded in the verifier. Running the following command reproduces the field comparison file after downloading the two linked tables:

```sh
python3 check_field_moments.py --table2 table-II.tex --table3 table-III.tex --out field-moments.csv
```

The suffix example gives a boundary to fixed-priority occupancy monotonicity under dynamic sibling priorities. The check invokes the same C++ generator as the national offer reconstruction and verifies every assignment for five family sizes and two menus, with and without the senior-sibling update.

```sh
python3 check_suffix_example.py --publication /path/to/CS2880-final-project/publication
```

## Who could turn an April miss into a hit?

Holding a single candidate and insertion-order assumption fixed, the April hit has an exact formula. For the first originally ranked course at a school the menu misses, blind can add 6,024 April hits and informed can add 1,981. With homogeneous marginal response probabilities among these respective applicants, the expected difference is **4,638 + 1,981 q_informed − 6,024 q_blind**. Independent response decisions across people are unnecessary. The formula also allows heterogeneous response if each probability is interpreted as the average within its own group of potential additional hits. The corresponding numbers for the least-congested candidate are 4,713 and 2,251. Placement outcomes require full market clearing.

A stronger conclusion uses the actual identities. If each pupil would respond under either menu or under neither, the possible difference in hits depends on how many *of the same* responding pupils would add a previously missed April school under each alternative. The C++ program independently rebuilds the two candidate courses per applicant, checks the frozen masks and counts the intersection and two exclusive groups. To attain each endpoint, the replay selects the pupils helping the corresponding policy only and assigns other families no response. A second construction gives both endpoints the same number of willing applicants per cohort by adding pupils outside the two exclusive groups in a deterministic keyed order. An applicant can be willing to respond but lack an eligible option under one menu, so the effective addition counts need not match.

For an arbitrary fixed budget of m willing applicants among U eligible under either policy, write b for the blind-only count, r for the informed-only count, and g for the initial hit difference. The sharp lower limit is g − min(m,b) + max(0,m−(U−r)). The sharp upper limit is g + min(m,r) − max(0,m−(U−b)). This follows by taking negative, zero, and positive contributions in that order for the lower limit and the reverse order for the upper. The 19.9% budget lies in the range where both unconstrained extremes are still feasible in each cohort. The pupil identities that construct the extremes deliberately use the historical enrollment labels. Their placement differences represent two attainable outcomes, not placement bounds.

For each of the two rules and four cohorts, holding the common willing set fixed, we ran the institutional solver on the lower and upper constructions, both insertion orders, and both exclusive-only and 19.9% participation, for both policies (128 national clears). For the equal-budget tail scenario with original-rank candidate selection, the lower endpoint gives 179 fewer hits and 1,669 fewer linked placements. The upper endpoint gives 5,412 more hits and 1,643 fewer linked placements. With candidate selection by lowest congestion, the lower endpoint stays positive in each cohort and reaches +920 overall even at arbitrary selection. Its corresponding four-cohort placement difference is −1,612.

To reproduce with the full repository, set PUB to the publication directory, ENGINE to the directory holding response_replay.cpp, FIELD to these source files, PRIVATE to the private work directory, RESULTS to the destination for aggregates and INPUTS to the directory with field-YYYY.csv and endpoints-YYYY.csv. PREFIX is the annual numeric market prefix (without extension), SIDE is the aligned institution sidecar and REGISTER is the matching prepared enrollment stream. All programs output aggregates.

```sh
c++ -O3 -std=c++17 -I"$PUB/research/observed/school" -I"$PUB/two-studies/institutional" "$FIELD/coverage_selection.cpp" -o "$PRIVATE/coverage_selection"
c++ -O3 -std=c++17 -I"$PUB/research/observed/school" -I"$PUB/two-studies/institutional" -I"$ENGINE" "$FIELD/selection_replay.cpp" -o "$PRIVATE/selection_replay"
"$PRIVATE/coverage_selection" "$PREFIX" "$SIDE" "$REGISTER" "$YEAR" "$RESULTS/selection-$YEAR.csv"
"$PRIVATE/selection_replay" "$PREFIX" "$SIDE" "$REGISTER" "$YEAR" "$RESULTS/selection-replay-$YEAR.csv"
```

Run the two annual executables for 2022 through 2025, then reconcile the per-applicant set calculations against the unselected full-uptake endpoint totals.

```sh
python3 "$FIELD/verify_selection.py" --inputs "$INPUTS" --results "$RESULTS" --out "$RESULTS/selection-summary.csv"
python3 "$FIELD/coverage_thresholds.py" --inputs "$INPUTS" --out "$RESULTS/coverage-thresholds.csv"
python3 "$FIELD/differential_response.py" --inputs "$INPUTS" --out "$RESULTS/differential-response.csv"
```

## Fixing the identities of the searchers to observed reapplicants

The returner comparison uses the same-grade applicant intersection from B1/B2 as a common willingness indicator. It can be restricted to the historically unoffered group using the observed initial and post-wait-list D1 outcomes. The previous section selected a group using enrollment to test a bound. Here neither the returner flag nor the additional course uses enrollment, although the later decision to return remains conditional on what actually happened. The recovered course at a previously undisplayed school follows the specified original-rank or congestion rule. It can differ from every course the family actually named in its complementary application. The companion re-listed course analysis instead follows explicit C2 entries.

First produce the private arrays and aggregate returner counts using prepare_relisted.py. Set REAPPLICATION_PRIVATE to their private directory and RELISTED_RESULTS to the aggregate directory containing relisted-YYYY.csv. For each of 2022–25 run

```sh
c++ -O3 -std=c++17 -I"$PUB/research/observed/school" -I"$PUB/two-studies/institutional" -I"$ENGINE" "$FIELD/participation_replay.cpp" -o "$PRIVATE/participation_replay"
python3 "$FIELD/participation_mask.py" --counts "$REAPPLICATION_PRIVATE/$YEAR-reapplication-counts.npz" --out "$PRIVATE/returned-$YEAR.u1"
"$PRIVATE/participation_replay" "$PREFIX" "$SIDE" "$REGISTER" "$YEAR" "$PRIVATE/returned-$YEAR.u1" "$REAPPLICATION_PRIVATE/$YEAR-unoffered.u1" "$RESULTS/participation-$YEAR.csv"
```

After the four annual files exist, verify their counts against the C2 aggregate tables and the sharp coverage bounds, then sum by scenario:

```sh
python3 "$FIELD/verify_participation.py" --relisted "$RELISTED_RESULTS" --results "$RESULTS" --out "$RESULTS/participation-summary.csv"
```

Of 154,988 observed same-grade reapplicants, 20,501 could add the candidate under at least one menu. The realized willingness set yields 4,572 additional April-school hits and 1,665 fewer linked placements when appending the highest-ranked candidate, and 4,671 more hits and 1,625 fewer linked placements with the least-congested candidate. Each comparison uses the complete national clearing and the same historical register-linked denominator. A person whose regular choices all fit on the menu can still name new schools in the complementary round.
