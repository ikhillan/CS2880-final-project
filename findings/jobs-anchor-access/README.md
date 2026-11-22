# What the released jobs aggregates can establish

The original jobs input extract and cached GPT menus are not in this repository.
The released placement and menu-change totals nevertheless imply useful bounds
on the anchor's unreported individual outcomes. **These are calculations from
existing aggregates, not a new jobs replay.** They provide neither observed exit
counts nor an estimate of post-anchor coverage.

| GPT five-option menus | Frozen window | Replication |
|---|---:|---:|
| Seeker–market observations | 5,760 | 3,840 |
| Placed before / after the anchor | 3,595 / 4,341 | 2,437 / 3,049 |
| Net additional placements | 746 | 612 |
| Changed personal menus | 1,205 | 920 |
| Gross exits: valid outer interval | **0–459** | **0–308** |
| Gross exits per all observations | **0–7.969%** | **0–8.021%** |
| Gross entries: valid outer interval | 746–1,205 | 612–920 |
| Original pooled conditional coverage | 32.544% | 32.006% |
| Post-anchor pooled conditional coverage: outer interval | **6.365–58.723%** | **3.166–60.846%** |

The positive placement gains do not establish individual retention. The
coverage bounds include substantial increases and decreases. We do **not** claim
that every endpoint is attainable under the particular anchor and DA instance.
Neither interval is a confidence interval. Sampling uncertainty is not added
to these bounds on the released panel.

The [protocol](PROTOCOL.md) specifies the still-pending jobs replay. The
[algorithm notes](theory.md) give constructed counterexamples and the
information assumptions for the coordinated baselines.

## 1. Bounds from placement margins

On a common population of size $N$, let $P_0$ and $P_1$ be placement counts
before and after an intervention. Let $X$ count gross exits (placed before,
unplaced after), $Y$ count gross entries, and $\Delta=P_1-P_0$. Then

$$
Y-X=\Delta,\qquad
\max(0,P_0-P_1)\le X\le\min(P_0,N-P_1).
$$

For any integer $X$ in that interval, the four binary transition counts
are nonnegative:

| | Placed after | Unplaced after |
|---|---:|---:|
| Placed before | $P_0-X$ | $X$ |
| Unplaced before | $X+\Delta$ | $N-P_1-X$ |

Thus these elementary bounds are sharp for arbitrary binary outcomes with
only the two margins fixed. This is **not** sharpness under DA, menu quotas,
the specific anchor, or the unpublished individual markets.

Apply the same formulas within each wage group. The group counts sum to the
overall count, so summing their lower and upper bounds can tighten the overall
interval. The scripts use exactly 12 lower-wage and 12 higher-wage observations
per primary market. They do not combine the natural-sampling tables with this
design. Full-graph maximum matching capacity supplies no extra overlap
information about the identities retained by the anchor.

## 2. A stronger bound from the number of changed menus

For the fixed strict priorities in the jobs simulation, changing **one
applicant's menu** can cause at most one gross exit and at most one gross entry.
This statement concerns receipt of any seat. Many people can change jobs or
receive worse-ranked jobs.

**Proof.** Remove the changed applicant $i$ and run applicant-proposing DA on
the remaining market; let its recipient set be $R$. Inserting $i$ can be
implemented by a rejection chain, starting from that DA allocation. Every
occupied slot receiving a preferred proposer stays occupied; at most one
applicant is free along the chain. Insertion therefore either leaves $i$
unmatched, adds $i$ without dropping a recipient, or replaces one member of
$R$ by $i$. Apply this description to both the old and new menus of $i$.
Their recipient sets differ by at most one element in either direction.

For $C$ changed rows, change them sequentially. Each final exit must exit at
some intermediate step, and similarly for entries. Therefore

$$
X\le C,\qquad Y\le C,\qquad
X\le C-\max(\Delta,0).
$$

Intersect this restriction with the group-margin intervals above. In the frozen
GPT comparison, $C-\Delta=1,205-746=459$; in the replication it is
$920-612=308$. The usual binary-margin upper bounds alone would have been
1,419 and 791, respectively.

The argument applies to the jobs code's fixed strict slot priorities, including
cloned slots for responsive posting capacities. It is not asserted for Chile's
endogenous sibling priorities or other mechanisms in which changing one report
changes other participants' priorities or feasible choices. It does not imply
that the people who exit are those whose menus changed.

The overall restriction can also tighten an individual group's marginal upper
bound: subtract the other group's minimum possible exits from the overall
maximum. Group intervals in the output are **marginal**; their upper endpoints
need not hold simultaneously.

## 3. Coverage is local to the menu, even when placement is not

Let $F$ be the number of people matched under full-information DA, and let
$H_0$ of their reference postings appear on the original menus. The reference
allocation stays fixed when evaluating the edited menus. If a person's menu
does not change, their reference-option coverage indicator cannot change.
For $C$ changed menus in total,

$$
\max(0,H_0-C)\le H_1\le\min(F,H_0+C).
$$

Divide by $F$ for pooled conditional coverage and by $N$ for unconditional
coverage. Equivalently, conditional coverage changes lie between
$-\min(H_0,C)/F$ and $\min(F-H_0,C)/F$.

For a wage group, the released anchor table does not give its number of changed
menus. We use the valid marginal budget $\min(C,N_g)$; it can be loose, and the
same overall budget cannot be spent independently in both groups. We make no
claim that the bounds are sharp jointly with the DA restrictions or matching
structure. In particular, local coverage bounds do not imply that placement
changes are confined to changed-menu applicants.

## 4. Why multiplying these particular coverage levels is valid

The release has two different coverage summaries:

- `accuracy.csv` stores `part.accurate.mean()` over individual observations.
  People with no full-information match have a missing coverage value and are
  excluded from this mean. It is exactly the **pooled** ratio $H_0/F$.
  Its `full_match_share` is $F/N$. Their product reconstructs $H_0/N$.
- `accuracy_pairs.csv` instead averages **within-seed conditional differences**.
  Its point estimate need not equal the difference between pooled levels.
  We do not use that file for these bounds or multiply its estimate by a mean
  full-match share.

The anchor script reports means of per-market placement and menu-change
percentages. Every primary market has 24 people and the same group sizes, with
equal numbers of markets per seed, so these means recover the corresponding
pooled counts. The reconstruction checks all values are within $10^{-6}$ of
an integer and verifies counts add across groups and agree between the
independently released tables. Observations are seeker–market appearances;
they are not unique workers or independent sampled people.

## Reproduce

```sh
python3 findings/jobs-anchor-access/aggregate_bounds.py
```

The standard-library script rebuilds [bounds.csv](bounds.csv): both windows,
eight policies, and three population/group rows, for 48 rows total. It records
all source SHA-256 hashes in
[aggregate_bounds_sources.json](aggregate_bounds_sources.json).

The CSV retains the binary-only interval, the group-margin refinement and the
DA/menu-change refinement separately. Placement and entry/exit percentages use
the row's full population; the explicitly named conditional coverage columns
use `full_match_count`. `da_exit_recipient_upper_pct` instead uses baseline
recipients. Coverage changes are percentage points, not relative percentages.

Small internal checks exhaust binary outcome tables for populations up to five
and all 2,048 single-row edits in two-person strict-priority DA markets. These
invented checks verify arithmetic and the elementary lemma; they are not jobs
results. Accessing the original replay inputs is still necessary to identify
the actual transitions, post-anchor coverage or paired uncertainty.
