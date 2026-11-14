# Evidence added during the referee audit

These analyses were chosen after inspecting the original two-study paper. They do not change the frozen later-cohort sign test. The manuscript distinguishes the original directional validation from these exploratory comparisons. All counts in these files are aggregates. The PDF can be built and the reported margins checked without applicant-level records or model credentials.

- [Outcome ranks in NYC](ordinal/README.md) and [outcome ranks, seat assumptions and score comparisons in Chile](chile/README.md). Ranking is always on the original order, never the displayed rank.
- [Holding the old allocation stable](slots/SLOT_METHODS.md): a score-based policy, the exact hindsight coverage bound with fixed quota, and a full-information positive control anchored at a different allocation. Includes independent serial versus batched checks and small-market enumeration.
- [Separate NYC catalogs and renter donors](nyc-temporal/README.md): first-five versus uniform, all supported and unsupported cells, block intervals and second-implementation checks. No new model calls.

From the paper directory, run `python audit/verify_audit.py`, `python audit/slots/check_slots.py`, and `python audit/chile/verify_aggregates.py audit/chile` with NumPy and pandas. See the linked methods for regeneration from source records. For new analyses, use a checkout of the branch containing this audit (or copy the current paper directory into the upstream research checkout). Dated upstream links in the paper README identify fixed inputs; the upstream source commit predates these new scripts.
