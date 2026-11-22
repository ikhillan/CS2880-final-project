# Replaying all 1,024 training lotteries for each frozen hindsight menu

See [the definitions, proofs and reproduction commands](BOUND.md). Both lower bounds use the historical April school to choose and score improvements. A placement loss is an applicant placed by the baseline but unplaced by the candidate policy under the identical draw. Every test uses the entire national allocation, including pupils with no linked enrollment. Only pupils with a register link can gain a label hit. The study contains one distinct fixed menu per cohort and version, with positive net coverage gains reported as people per cohort. The replay counts repeated across lotteries are explicitly labeled draw totals.

If both CSV series are in the same directory, run the following after the cohort checks. This compares the two constructive lower bounds, verifies both full-support zero-exit certificates and reports the better menu per cohort.

~~~sh
python3 summarize_search.py --first aggregates --second aggregates --out selection.csv
~~~

The final mask audit reconstructs each winning and runner-up menu from its search records. It tests course eligibility, the inclusion of every common and continuity course, and exact list length for all pupils. It recomputes actual old and new historical coverage from school identifiers, counts positive hits and lost hits, verifies the reported objective, and writes the SHA-256 of each final private mask.

~~~sh
python3 score_menus.py --binary "$TMPDIR/retention-oracle" --scratch "$TMPDIR/retention-private" \
    --early "$EARLY" --late "$LATE" --features "$FEATURES" --sides "$SIDES" \
    --first "$TMPDIR/retention-results" --second "$TMPDIR/retention-results" --out "$TMPDIR/retention-results/mask-scores.csv"
~~~
