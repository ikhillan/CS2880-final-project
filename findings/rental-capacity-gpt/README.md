# Rental capacity decomposition and reserved slot for the GPT arm

This is the jobs analysis (`experiments_jobs/capacity_anchor.py`) applied to the GPT rental runs in [`../gpt-luna-rentals/`](../gpt-luna-rentals/README.md), so both NYC studies can be read the same way. It's a post hoc analysis, run on 2026-11-22 with no API calls. It uses the same markets as the rental runs and GPT's committed replies. Placement levels reproduce `../gpt-luna-rentals/accuracy.txt` exactly in all three windows.

**Method.** Change in placed = change in capacity − change in shortfall, each against random 5-listing lists on the same markets.
- **Capacity:** the largest matching the shown pairs allow.
- **Shortfall:** the part of that capacity deferred acceptance misses.
- **Reserved slot:** each list keeps 5 listings, but one slot is reserved for a listing from a largest assignment over all feasible pairs, which reuses already-shown pairs where it can. GPT's last-ranked pick gives up its slot.

Means are unweighted over markets, as in `rental_accuracy.py`. Intervals are 95% seed-block bootstraps with 10,000 draws.

## GPT 5 listings vs random 5 listings (pp; development / held out / replication)

| | Enriched | Primary |
|---|---|---|
| Placed | −2.0 / −3.7 / −3.3 | −1.2 / −1.4 / −2.2 |
| Capacity | **−8.8 / −10.0 / −10.6** | −2.7 / −2.5 / −3.4 |
| Shortfall | −6.8 / −6.3 / −7.3 | −1.4 / −1.2 / −1.2 |
| Reserved-slot gain | +3.5 / +4.0 / +4.9 | +0.7 / +0.8 / +1.0 |
| Anchored GPT vs plain random | **+1.5 [1.0, 2.1] / +0.3 [−0.3, 0.9] / +1.6 [1.1, 2.2]** | −0.5 / −0.6 / −1.3 |

**What it shows:**
- **Capacity:** GPT's lists cut capacity sharply in the enriched markets, by 9–11 pp, the same mechanism as in jobs.
- **Offset:** placement falls far less than in jobs because deferred acceptance also misses less (shortfall −6 to −7), recovering about two-thirds of the lost capacity. In jobs only about a fifth comes back (capacity −19.7, shortfall −4.1, placed −15.5). This is the same pattern the paper's archived learned model shows on rentals (`publication/research/observed/capacity/nyc-summary.csv`: capacity −10 to −12, placement −1 to −2).
- **Reserved slot:** one reserved slot per list removes the whole GPT placement loss in enriched markets. Anchored GPT places as many renters as random in the held-out window and slightly more in the other two.
  - **Lower-income renters** gain +3.6 to +5.6 pp from the reserved slot.
  - **Higher-income renters** gain +3.4 to +4.2 pp.
  - **Jobs comparison:** there the slot recovers 13–16 pp but leaves GPT 1.5–2.6 pp below random.
- **Honest ranker:** it cuts capacity much more (−45 pp in enriched), and the reserved slot brings it to +1.4 to +2.7 above random.

## Files
`<window>/capacity_levels.csv`, `capacity_decomposition.csv` and `anchor.csv`, for `stage1_pilot` (development), `heldout_primary` and `heldout_replication`.

## Reproduce
You need the FirstMover months each config uses, with hashes in `data/firstmover/raw/*.manifest.json` and cleaning by `scripts/firstmover_clean.py`, plus the ACS files.
```sh
for w in stage1_pilot heldout_primary heldout_replication; do
  gunzip -c findings/gpt-luna-rentals/$w/gpt_recs.jsonl.gz > /tmp/$w.jsonl
  python -m experiments.rental_capacity_anchor configs/$w.json --llm /tmp/$w.jsonl
done
```
