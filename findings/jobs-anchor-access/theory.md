# Coordinated menus: guarantees and counterexamples

This records algorithmic properties and constructed examples for the planned
jobs-anchor follow-up. **It reports no new empirical experiment.** The archived
jobs markets have modeled preferences. None of the policies below accesses those
realized preferences or priorities when choosing a menu.

Implementation: [`experiments_jobs/coordinated_policies.py`](../../experiments_jobs/coordinated_policies.py).
Checks: [`tests/test_jobs_coordinated_policies.py`](../../tests/test_jobs_coordinated_policies.py).

## Inputs and intervention

Let `E[i,s]` be feasibility for seeker `i` and physical slot `s`. Slots map to
posting IDs through `s2p`; exposing a posting exposes every feasible slot of that
posting. Let `D` be the original displayed graph. Its per-person **posting**
quota is fixed. This differs from preserving the number of slot edges when
postings have several openings.

All methods use `E`, posting membership and the original menus. A supplied
drop order may use the cached recommender ordering. No method takes an
`Instance`, true rank matrix, employer priority, assignment, or evaluation label.
This API restriction does not make arbitrary supplied scores legitimate:
score provenance must still be recorded.

## Maximum-matching anchor

The archived anchor finds a maximum matching of `E`, breaking ties by maximizing
the number of its already displayed edges. For each assigned seeker whose anchor
posting is absent, it exchanges one displayed posting for that anchor. Empty
menus stay empty.

**Displayed-capacity certificate.** If every seeker assigned an anchor has a
nonempty original menu, the edited graph contains the entire anchor matching.
It is a subgraph of `E`, so its maximum matching cardinality equals that of `E`.
This statement allows multiple slots per posting. Each target anchor slot is
exposed when its posting is inserted.

This is a feasibility certificate, **not a statement about deferred acceptance
(DA)**. It does not guarantee aggregate DA improvement, individual retention,
rank improvements, coverage, salary, or protection of either wage group.

The archived assignment cost uses a fixed infeasible-edge penalty of 10,000.
It is safely lexicographic for the archived 24-person markets, but should not
be extrapolated to arbitrary market sizes. `original_anchor` calls the archived
implementation to reproduce its outputs. `permuted_anchor` applies seeded
row/slot permutations before that same solver and maps indices back; it tests
sensitivity to arbitrary ties, not uncertainty about underlying preferences.
Different tie seeds need not produce different menus, and are not independent
empirical markets.
For an isolated matching-tie sensitivity, freeze each seeker's drop order across
replicates as well. Resetting one shared random stream is insufficient when it
is consumed only on changed rows: a different set of early edits can change a
later seeker's random drop.

**Empty-menu exception.** With two seekers, two feasible postings each, and
original menus `[]` and `[0]`, a full matching has size two. The edited menus
still have cardinalities zero and one, so their matching capacity is only one.
A missing recommender response must not silently be counted as covered by the
full-capacity certificate.

## A one-edit counterexample: capacity rises and DA placement falls

The following four-person instance is exercised against the repository's actual
`apply_anchor` and `deferred_acceptance` functions. Lower ranks are better; `-`
means infeasible. Every posting has one seat.

| Seeker | Feasible preference order | Original menu | Anchored menu |
|---|---|---|---|
| a | 3, 2, 0, 1 | 0, 1 | 0, 1 |
| b | 2, 3, 0, 1 | 0, 1 | 0, 1 |
| c | 3, 0, 2 | 0, 3 | 0, 2 |
| d | 0, 1, 3 | 1, 3 | 1, 3 |

Posting priorities, highest first:

- Posting 0: c, a, b, d.
- Posting 1: d, a, b.
- Posting 2: a, b, c.
- Posting 3: c, d, b, a.

The archived solver returns anchor assignment `(0,1,2,3)`. Choose a drop order
that removes posting 3 from c's old menu; this can be a saved recommender's last
choice even though it is c's most preferred feasible option in the modeled
evaluation preferences. Only c's menu changes.

Before the edit DA gives `(0, unmatched, 3, 1)`: **three placements**, and the
displayed graph has maximum matching size three. After the edit DA gives
`(unmatched, unmatched, 0, 1)`: **two placements**, even though displayed matching
capacity is now four. Removing c's posting 3 sends c to 0; c displaces a, who
cannot beat d at posting 1. One incumbent exits and nobody enters.

Thus even an improvement in the graph's maximum matching size does not imply
an improvement in DA placement. This is a counterexample, not an empirical
estimate of how often the intervention harms participants.

## Fixed quotas can prevent any maximum-cardinality DA guarantee

Take seekers a and b with feasible sets `{x,y}` and `{x}`. Require quotas two and
one, respectively, so there is only one legal menu profile. Let a prefer x to y
and let x prioritize a above b. DA places only a, although the feasible matching
`a -> y, b -> x` seats both. No alternative menu algorithm respecting these
quotas can fix this instance.

Offering only each matched person's maximum-matching anchor can attain maximum
cardinality, because every anchor receives a proposal and any rejection keeps
that seat filled. This singleton-menu intervention relaxes the original quotas.
Forcing anchors to the top of application lists likewise changes the ordering
assumption; the existing replay follows the modeled preferences. Neither is an
equivalent five-option recommendation treatment.

## Score-weighted anchor: an explicit information baseline

`score_anchor` first maximizes matching cardinality, then the total supplied
within-person normalized score rank of assigned postings. Ties are randomized
by a recorded seed. Reuse of old displayed edges is not its secondary objective.
The insertion step still changes at most one posting per affected seeker.

`normalize_score_ranks` gives equal scores equal rank, scales distinct ranks to
integers from zero to 1,000, and gives a row with only one distinct feasible score
1,000 on each option. This convention is ordinal. It does not create comparable
cardinal utilities or calibrated admission probabilities. The assignment's
infeasible-edge cost is `1 + 1000 * min(number_of_seekers, number_of_slots)`;
it dominates every possible score difference, preserving cardinality priority.

Valid deployable scores require accessible features and a scoring rule frozen
before evaluation. The existing `market.score_deterministic` uses salary,
prior wage and seniority but also the simulator's utility coefficients. It is
therefore a **model-informed benchmark**, not independently validated learning
from observed employment choices. Evaluating matching on latent taste shocks,
true employer priorities or full-information DA targets would instead be an
oracle comparison.

Cached GPT top-five choices do not provide scores for all feasible pairs.
`cached_rank_relevance` gives the displayed choices scores `k,...,1` in their
saved order and ties unlisted options at zero. This is an explicit censoring
convention for a coordination baseline; it must not be described as observing
low relevance for all omitted choices.

## Balanced menus: exact objective, different intervention

For displayed posting indicators `x[i,j]`, fix each row sum `k[i]`, require
`x[i,j] <= E_post[i,j]`, and write `l[j] = sum_i x[i,j]`. `balanced_menu`
lexicographically optimizes

1. Minimize `sum_j l[j]^2`.
2. Among those optima, maximize `sum_(i,j) relevance[i,j] * x[i,j]`.

This can replace multiple options per person. It is not a one-slot anchor
ablation. Posting load measures how many menus display a posting, not its hiring
capacity. Equal-load balancing is most directly interpretable for the primary
one-seat-per-posting design; capacity-weighted balancing would be a different
specified objective in the expected-hires sensitivity.

**Exact flow implementation.** Send `Q = sum_i k[i]` units through
source -> seeker -> posting -> sink. Source-to-seeker capacity is `k[i]`;
each feasible seeker-to-posting arc has capacity one and relevance cost.
Parallel unit posting-to-sink arcs have marginal costs `1,3,5,...`, whose first
`l[j]` terms sum to `l[j]^2`. Multiply these costs by
`L = Q * (max(relevance) - min(relevance)) + 1`. Every one-unit improvement in
the primary integer objective outweighs the entire possible secondary-score
range. The implementation uses exact integer costs and successive shortest
augmenting paths, with no new solver dependency.

Balanced loads alone do not certify full matching capacity. A tested four-person
example uses feasible posting sets `{1,2,3}`, `{3}`, `{0,1,2}`, `{1}` and quotas
`(2,1,2,1)` (a common two-option cap). An optimal balanced solution can show
`{1,3}`, `{3}`, `{0,2}`, `{1}`. Its loads `(1,2,1,2)` minimize squared load, but
the first, second and fourth seekers can use only two postings. Its matching
capacity is three while the full feasible graph supports four. Relevance can
strictly select this menu among load-minimizing alternatives. DA outcomes and
individual harms therefore remain evaluation quantities for this baseline too.

## Required evaluation and tests

Pair random/GPT menus with their respective anchors on identical markets.
Keep the drop convention explicit. Report placement, displayed matching
capacity, target coverage, gross exits, entrants and rank changes, both overall
and by wage group. A post-hoc veto using realized losses would add privileged
information; label it as an oracle if explored. Repeated solver ties are
sensitivity draws, not additional independent markets or extra sample size.

The standard-library test runner exhausts every 2-by-3 feasibility/display state
for the archived anchor objective, and every 2-by-3 feasibility graph and row
quota for balancing under three relevance specifications. It independently
enumerates matchings/menus; it also tests rectangular graphs, seeded index
remapping, posting clones, empty menus, and the explicit negative examples.

```sh
python3 -m unittest tests.test_jobs_coordinated_policies
```

These checks establish the stated tiny-instance algorithmic properties. They
are not a claim of prior-literature novelty or a substitute for running the
registered empirical evaluation.
