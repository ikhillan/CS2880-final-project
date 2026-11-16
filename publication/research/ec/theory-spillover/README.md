# A shortlist's effect on who gets matched

In a strict one-to-one market, two outsider policies that end up assigning the same set of listings to outsiders leave exactly the same incumbents matched. Recompute stable matching for incumbents alone after removing the listings occupied by the outsiders. The identities of the matched incumbents in this residual calculation equal those in the original outcome. This holds for any matching stable on the visible graph. To predict which listings outsiders will take under a new policy, their preferences and the mechanism can still matter.

For an individual outsider choosing among feasible listings, the supplement combines this identity with the DA menu principle. Comparing no entry with a singleton entrant for each listing identifies both which listings the outsider can attain and which incumbent, if any, would lose a match. Each feasible display of any given fixed length can then be evaluated by inspecting the highest-ranked attainable listing it contains. The uniform-display probabilities and feasibility of protecting an arbitrary incumbent group follow by counting. The full argument and sharp examples are in menu.tex. Changes to incumbents' ranks, satisfaction or exact partners are different outcomes from gaining or losing a match.

## Relation to prior work

The argument uses the stable correspondence's consistency under deleting matched pairs, matched-set invariance, entry monotonicity, proposer strategyproofness and DA menu descriptions. Pápai's Proposition 5 explicitly proves that a participant receiving a listing can obtain it by reporting only that listing. The supplemental section turns those existing components into a prediction of the identities of incumbents who lose a match under shortlist changes. The prior literature does not justify a priority claim for matched-set invariance, consistency or the ordinary strategyproof menu. The corresponding primary publications and preprint are identified in references.bib. [Sasaki and Toda](https://doi.org/10.1016/0022-0531(92)90078-V) · [Gale and Sotomayor](https://doi.org/10.1016/0166-218X(85)90074-5) · [Roth](https://doi.org/10.2307/1913160) · [Gonczarowski, Heffetz and Thomas](https://www.nber.org/papers/w31506) · [Pápai](https://arxiv.org/abs/2609.27293)

## Verification

Run the verifier with a current standard-library Python 3 installation. No source data or matching libraries are required.

~~~sh
python -B verify.py --out replay.json
~~~

The verifier ranges over every mutually acceptable bipartite graph and every pair of strict priority/preference profiles restricted to that graph on the four labeled sizes 2×2, 2×3, 3×2 and 3×3. It independently enumerates all partial matchings, tests stability on the given edges and checks that the round-based DA result is the coordinatewise proposer optimum in the resulting stable set. All tests use immutable integer outcomes. It restricts each focal applicant to every subset in true order and groups them by quota. Disjoint incidence sets and stock variation are tested on both sides.

The 136,727 profiles yielded 1,090,240 coalition residual checks, 2,700,974 focal-list predictions, 1,479,994 exact quota probability and support tests, and 8,707,616 directed stock-pair bound checks. All passed. In 19,740 focal-list cases at least one incumbent received a different partner in the singleton replay, even though every incumbent's matched status was preserved. These exhaustive finite results supplement the general proof.

Raw counts by tested dimension and the labeled counterexample in which seat marginal contributions violate both submodularity and supermodularity are in evidence.json. The counterexample encodes a listing subset by a bitmask, assigning bits 1, 2 and 4 to x, y and z.

The saturation theorem bounds the number of initially matched protected applicants who can lose a match in a full rollout after a lossless intermediate rollout. The upper bound is the number of remaining edits. The construction realizes the bound while making every intermediate rollout involving at most r editors lossless, with common priority and a universal list quota k at least r+1. The verifier checks all 2,800 rollout assignments in 60 versions of this family through seven editors.

The representation theorem counts the bits needed to encode the map from a focal report to all incumbents' binary match statuses after preprocessing the background market. The tight bound is Θ(n log n) bits for labeled identities and Θ(n) for the matched count of any predeclared nonempty group, even one named incumbent. Gonczarowski and Thomas give an Ω(n²) worst-case bound for the full DA allocation function. [Structural Complexities of Matching Mechanisms, Corollary 3.10](https://arxiv.org/html/2212.08709v3)
