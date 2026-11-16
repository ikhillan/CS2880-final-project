# Observed bilateral preferences in the Columbia experiment

## Sources

The source population comprises the 16 waves in version 1.0 of the Bhargava–Fisman replication data, file 2509001, released under CC0. Its 7,674 directed responses yield 3,837 reciprocal meetings. The public extended version adds participant-reported overall liking and their pre-event questionnaire. We validated every one of the 7,674 decision values and genders against the author-deposited source by both people’s IDs and wave. The extended questionnaire and codebook were obtained at a fixed Git commit. We have not verified a separate redistribution grant for these two files; the archive contains download instructions and aggregate results. An overall liking score is an observed response, while a strict tie ordering and a single stable partner are counterfactual modeling choices. [Harvard Dataverse, version 1.0](https://doi.org/10.7910/DVN/27893) · [Original QJE study](https://doi.org/10.1162/qjec.2006.121.2.673) · [Questionnaire and codebook snapshot](https://github.com/cc6580/Analysis-of-Contributing-Factors-for-the-Success-of-Speed-Dating/tree/9096ea5b375011680cf23e5d2ab69dcd2b79467f)

## Replay

Python 3 with NumPy, pandas and SciPy suffices. Run from this directory in an empty output folder. Downloads are independently checksum-verified.

~~~sh
python fetch.py /tmp/ec-bilateral-input
python -B replay.py --data /tmp/ec-bilateral-input --out /tmp/ec-bilateral-results
python -B verify.py --data /tmp/ec-bilateral-input --out /tmp/ec-bilateral-results
python -B mc_precision.py --data /tmp/ec-bilateral-input --out /tmp/ec-bilateral-results
~~~

## Reading the tables

PROTOCOL.md fixes the cohort, primary policy, comparison, k, chronological split, feature restrictions, outcomes and simulation seeds before the first policy comparison. The main model is fit only to disjoint earlier people. Priorities on the receiver side and proposal order on the proposer side use overall liking among mutually affirmative pairs; each side’s ties are resolved independently and each shared refinement is held fixed across all policies. ID is used solely for joining data and array indexing. Mutual consent is not an input for the primary rule. Sensitivities give the same observed acceptable graph to every rule, alternate the proposer side and shorten the quota. All participants in the source remain in their denominator. Of 354 test-period mutual yeses, one had a missing or out-of-range score and was excluded from DA while retained for real-choice coverage. The effect of exposure on responses would require an actual exposure intervention. Ratings are evaluated under the assumption that what people recorded after the full set of encounters remains their ranking under the hypothetical smaller menu. A speed-dating match rate cannot be converted into a housing effect.

summary.csv reports equal-session rates and policy-minus-uniform differences. side=0 corresponds to female proposers, side=1 to male. known=0 scores all linked meetings using only pre-event data, then restricts to mutually assenting and rated pairs inside DA. known=1 supplies the latter graph to every ranker. Age/interest similarity equals the Pearson correlation on the pair’s commonly answered pre-event interests, less the absolute age gap in decades. The pooled prediction model imputes training means and out-of-sample data receive those same transformations. Salience weights are normalized within person because the questionnaire used different scales in different waves. Columns low and high are paired event-day percentile intervals for the difference. With just six dates these intervals should be read as descriptive; no adjustment is applied across sensitivity analyses. The uniform column for the directly observed outcome uses the same Monte Carlo slates as DA.

wave-results.csv contains only event-level means. verification.json records a separate round-based implementation confirming 8,672 entire assignment vectors. degrees.csv documents the number of proposers for whom the quota is exhaustive. REPLICATION_CHECK.md and mc-precision.json document a post-hoc, larger Monte Carlo precision check on the two primary endpoints. [Source study on order effects](https://sites.bu.edu/fisman/files/2015/11/BhargavaFisman2011_final.pdf)
