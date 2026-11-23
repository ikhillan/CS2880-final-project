# Latest GitHub review for the presentation

Reviewed 22 November 2026. Remote references were fetched; no checkout, merge, experiment or paper change was made.

## Ishan's latest contribution is already included

The latest Ishan-authored commit found across the remote branches is [`6c44e7ec941e8c56d2ccb74004a496aa9ca6e5f4`](https://github.com/ikhillan/CS2880-final-project/commit/6c44e7ec941e8c56d2ccb74004a496aa9ca6e5f4), **22 November 2026, 02:59:37 ET** (recorded as 21 November, 23:59:37 −08:00): “Add gpt-6-luna rental findings for all three windows.” It changes `experiments/rental_accuracy.py` and adds the three rental caches and findings tables. It is an ancestor of our reviewed baseline, `d456f71`; no remote branch contained a later commit than that baseline when checked.

The public caches contain **20,740 development, 13,580 held-out and 20,795 replication replies**, totaling **55,115**. Every row has `status=ok` and a candidate hash. The completed rental reconstruction verified all 55,115 hashes against the reconstructed ordered feasible listing identities, and reproduced the archived placement results. These are cached model recommendations, not observed renter preferences or rental outcomes. See [the original findings](../../findings/gpt-luna-rentals/README.md) and [the completed access audit](../../findings/rental-anchor-access/README.md).

## Calendar windows are not disjoint property samples

Ishan's older [`a29e2333b5dcdaf0a6323181c539372da6fcd993`](https://github.com/ikhillan/CS2880-final-project/commit/a29e2333b5dcdaf0a6323181c539372da6fcd993), **13 November 2026, 00:36:19 ET**, adds development/held-out unit deduplication on `feature/infra-cleanup`. It changes `hiddenoptions/data.py`, `hiddenoptions/markets.py`, `scripts/heldout_dedup_report.py`, `tests/test_dedup.py` and `data/firstmover/heldout_dedup_counts.csv`. That change is not in the GPT market builder used for the results presented here.

The following **full input-catalog counts**, independently checked against the local cleaned files, match Ishan's table. They are not counts of sampled markets, sampled renters, or overlap among the actual 64-listing market draws.

| Month | Analysis-catalog listings | Identifiable rows also in development | Additional identifiable repeats within held-out period | Listings after the proposed exclusions |
|---|---:|---:|---:|---:|
| Sep 2026 | 13,776 | 992 | 0 | 12,784 |
| Oct 2026 | 13,838 | 1,100 | 227 | 12,511 |
| Jun 2025 | 22,111 | 2,872 | 0 | 19,239 |
| Jul 2025 | 24,098 | 2,975 | 212 | 20,911 |
| Aug 2025 | 21,696 | 1,880 | 348 | 19,468 |

The denominator includes every row with `in_analysis_catalog=True`. A held-out row is identifiable here when `unit_key_source == "url_unit"`. Its `unit_id` is compared with the union of analysis-catalog unit IDs from June–August 2026. The cleaner constructs `unit_id` as `fmu:` plus the first 12 hexadecimal characters of the SHA-1 hash of the unit key; for `url_unit`, that key comes from the normalized building/unit URL path. The third count removes earlier-period repeats only when they were not already counted as development overlap. Counts across months can refer to the same physical unit.

Read-only reproduction with the existing cleaned inputs, run from the repository root:

```python
from pathlib import Path
import pandas as pd

base = Path("data/firstmover/clean")
def catalog(month):
    d = pd.read_csv(base / f"firstmover_listings_{month}.csv.gz",
        usecols=["unit_id", "unit_key_source", "in_analysis_catalog"])
    return d[d.in_analysis_catalog.astype(bool)]

dev = set(pd.concat([catalog(m) for m in
                    ["2026-06", "2026-07", "2026-08"]]).unit_id)
for period in [["2026-09", "2026-10"],
               ["2025-06", "2025-07", "2025-08"]]:
    earlier = set()
    for month in period:
        d = catalog(month)
        identified = d.unit_key_source.eq("url_unit")
        overlap = d.unit_id.isin(dev)
        repeat = d.unit_id.isin(earlier)
        print(month, len(d), int((identified & overlap).sum()),
              int((identified & repeat & ~overlap).sum()),
              int((~(identified & (overlap | repeat))).sum()))
        earlier.update(d.unit_id)
```

Applying the proposed exclusions now would change sampled candidates and invalidate the existing recommendation-cache identities. That would require a separate study, not a silent correction to this replay. This overlap limits claims of property-level generalization; it does not invalidate the paired policy comparisons on the reconstructed markets.

## Suggested wording

**Main narration:** “We reuse three calendar windows with modeled preferences; some homes recur across windows, so this is not independent property-level validation.”

**Appendix:** “The GPT replay preserves the original catalogs: 992 of 13,776 September and 1,100 of 13,838 October analysis-catalog listings match development unit IDs under the identifiable-URL rule; these are catalog overlaps, not measured overlap among the sampled markets.”

Keep the stronger rental coordination and individual-access results in the main talk. Jobs provide secondary evidence from published aggregates; missing jobs inputs still prevent the new coverage/retention audit. Older infrastructure branches provide code, not additional completed results to claim in the presentation.
