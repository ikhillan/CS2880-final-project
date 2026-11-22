# Testing a two-policy rollout against the baseline assignment

The proposed display replaces complete applicant rows. A row is flagged if it removes its baseline contract or exposes a blocking pair against the baseline allocation. For a stable baseline, the exact set of row subsets that retain it as a stable witness is the power set of the unflagged rows. The probability part uses the one-sided factorization from Aziz et al., *Stable Matching with Uncertain Linear Preferences*, Theorem 3. The count of people placed under the subsequent allocation and the restricted-coverage ceiling are separate questions. The paper extends the residual-capacity proof of the omission theorem to arbitrary, nonnested displays. [Primary prior work](https://doi.org/10.1007/s00453-019-00650-0)

The empirical script uses the exact stored NYC menus and full-list DA reference, the Chilean frozen predictor masks and reported priorities, and the specified pooled-priority replay. It verifies the stability condition on all unflagged changes simultaneously, the gross-count inequalities under the full change and protection of every incumbent’s rank when only those unflagged changes are enacted. A flagged change may cause no harm. The screen certifies retention of the original matching itself as a stable possibility. Forecasting the oracle coverage gains or deploying the screen with unobserved rankings would require additional information.

All counts are raw sums, including the NYC aggregates over 240 markets per design. 'full_safe_improving_rows' and 'recorded_safe_improving_rows' are the exact maximum gains in the two reference coverage measures within the class of baseline-preserving A/B row switches. Maximizing one of these objectives is retrospective. A historical offer outside the retained graph counts as missing under both policies. 'bound' adds the market-level, realized bounds from flagged applicants’ recipient-set union and program occupancy overlap.

Two counts in the 2022/23 results happen to coincide. Of 3,750 people newly shown their full-list DA partner at universal adoption, 3,728 received that actual program in the replay and 3,728 had a menu that fails the original-stability screen. The intersection of those two groups has 3,711 applicants.

From the repository root, with the NYC development data, school numeric market files and Python environment prepared as in the two-study paper:

```sh
python rollout_certificate.py --repo "$REPO" --numeric "$NUMERIC" --out rollout-certificates.csv
c++ -std=c++17 -O3 verify_rollout_certificate.cpp -o verify_certificate
./verify_certificate > rollout-exhaustive.json
```

The C++ program generates all 34 partial matchings on three applicants and three one-seat programs. Feasibility and the absence of blocking pairs directly identify all stable selections. Every ordered display pair and complete strict profile is represented up to relabeling. When the same baseline matching is stable on several displays, the bound depends only on that matching and the target graph, so the calculation is reused and weighted by the exact number of baseline displays. All 1,073,293,568 stable-outcome-pair checks passed. For eligible graphs, applicant-optimality is tested independently by choosing a stable outcome minimizing total applicant rank, which is attained uniquely at the applicant-optimal stable allocation under strict preferences. The finite checks supplement the residual-capacity proof.
